"""Step 31: deployable image-recoverability gate (offline, held-out).

The unconditional image-counterfactual steering nets ~0 on MMStar/NaturalBench because
the recoverable slice is small and steering the rest adds collateral. This builds an
actual GATE: a per-item label-free classifier that decides whether to steer, trained
to separate FIXES (steering corrects a baseline error) from BREAKS (steering breaks a
baseline-correct item). Cell + threshold selected on DEV, reported on held-out TEST.

Features (label-free, per item): base_margin, noimg_margin, ||delta(L)||, and the
image-insensitive flag (base_pred==noimg_pred). Gate fires -> use steered pred, else
baseline pred. We report baseline / unconditional / gated / oracle-gate(ceiling) on test.

    python scripts/31_recoverability_gate.py --benchmark mmstar
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
import warnings; warnings.filterwarnings("ignore")


def grouped_split(rows, seed, frac):
    g: dict = {}
    for i, r in enumerate(rows):
        g.setdefault(r.get("question_group", r["candidate_id"]), []).append(i)
    keys = sorted(g); np.random.default_rng(seed).shuffle(keys)
    tk = set(keys[:int(round(len(keys) * frac))]); m = np.zeros(len(rows), bool)
    for k, ii in g.items():
        if k in tk:
            for i in ii: m[i] = True
    return m


def _gate():
    return make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rows-jsonl", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    rows = [json.loads(l) for l in Path(args.rows_jsonl
            or f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rows])
    base = np.load(out_dir / "baseline_pred.npy")
    noimg = np.load(out_dir / "noimg_pred.npy")
    bm = np.load(out_dir / "base_margin.npy")
    nm = np.load(out_dir / "noimg_margin.npy")
    delta = np.load(out_dir / "delta.npy")
    preds = np.load(out_dir / "sweep_preds.npy")
    meta = [json.loads(l) for l in (out_dir / "sweep_pred_meta.jsonl").read_text().splitlines() if l.strip()]
    real_cells = [(i, m) for i, m in enumerate(meta) if m["tag"] == "real"]

    is_test = grouped_split(rows, args.seed, args.test_frac); dev = ~is_test
    bc = base == ci
    base_dev, base_test = bc[dev].mean(), bc[is_test].mean()

    def features(L):
        dn = np.linalg.norm(delta[:, L, :].astype(np.float32), axis=1)
        return np.column_stack([bm, nm, dn, (base == noimg).astype(float)])

    best = None
    for pi, m in real_cells:
        L, a = m["layer"], m["alpha"]
        steered = preds[pi]; sc = steered == ci
        decisive = sc != bc                                  # fix (sc=1) or break (sc=0)
        tr = dev & decisive
        if (sc[tr].sum() < 8) or ((~sc[tr]).sum() < 8):      # need both classes
            continue
        X = features(L)
        # train the gate on the decisive (fix/break) dev subset; target = "steered correct"
        clf = _gate(); clf.fit(X[tr], sc[tr].astype(int))
        p_dev = clf.predict_proba(X[dev])[:, 1]    # scalar-threshold pick (1 param, minimal overfit)
        p_test = clf.predict_proba(X[is_test])[:, 1]   # held-out test: clf never saw these

        # pick threshold on dev to maximize gated dev accuracy
        best_tau, best_devacc = None, -1
        for tau in np.linspace(0.1, 0.9, 33):
            fire = p_dev >= tau
            gated = np.where(fire, sc[dev], bc[dev])
            acc = gated.mean()
            if acc > best_devacc:
                best_devacc, best_tau = acc, tau
        fire_t = p_test >= best_tau
        gated_test = np.where(fire_t, sc[is_test], bc[is_test]).mean()
        rec = dict(L=L, a=a, tau=round(best_tau, 3), dev_acc=best_devacc, test_acc=gated_test,
                   fire_frac=float(fire_t.mean()),
                   fixes=int((fire_t & (sc[is_test] & ~bc[is_test])).sum()),
                   breaks=int((fire_t & (~sc[is_test] & bc[is_test])).sum()))
        if best is None or rec["dev_acc"] > best["dev_acc"]:
            best = rec

    print(f"=== {args.benchmark}: deployable image-recoverability gate ===")
    print(f"split dev {dev.sum()} / test {is_test.sum()} | baseline acc dev {base_dev:.3f} test {base_test:.3f}")
    if best is None:
        print("no cell had enough fix/break examples to train a gate"); return
    # references at the dev-selected cell
    sel = next(pi for pi, m in real_cells if m["layer"] == best["L"] and m["alpha"] == best["a"])
    steered = preds[sel]; sc = steered == ci
    uncond_test = sc[is_test].mean()
    oracle_test = np.where(sc[is_test] > bc[is_test], sc[is_test], bc[is_test]).mean()  # fire iff helps
    print(f"selected cell L{best['L']} a={best['a']} (gate tau={best['tau']})")
    print(f"  baseline           test {base_test:.3f}")
    print(f"  unconditional steer test {uncond_test:.3f}  ({uncond_test-base_test:+.3f})")
    print(f"  GATED              test {best['test_acc']:.3f}  ({best['test_acc']-base_test:+.3f})  "
          f"fire {best['fire_frac']:.2f}  fixes {best['fixes']}  breaks {best['breaks']}")
    print(f"  oracle gate (ceil) test {oracle_test:.3f}  ({oracle_test-base_test:+.3f})")


if __name__ == "__main__":
    main()
