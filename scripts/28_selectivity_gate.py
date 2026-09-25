"""Step 28: offline selectivity-gate prototype for image-counterfactual steering.

The unconditional image-counterfactual steering (scripts/27) recovers prior-violating
items but costs prior-consistent collateral. Selectivity = steer only the items that
need it. This prototype is PURELY OFFLINE: it reuses the saved per-cell predictions
(sweep_preds.npy) + baseline_pred.npy + delta.npy, so no GPU and any gate is free to
sweep.

Gated prediction:  pred_i = steered_i if gate fires on i, else baseline_i.

Gate signals available offline (label-free, per item):
  - dnorm : ||delta_i(L)|| -- how much the image moves the residual at the steer layer.
            We try BOTH directions (steer the highest- or lowest-influence items).

We rebuild the SAME grouped dev/test split as scripts/27, sweep the gate fraction,
select the operating point on DEV, and report it on held-out TEST.

    python scripts/28_selectivity_gate.py --benchmark vilp --layer 31 --alpha 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def grouped_split(rows, seed, test_frac):
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("question_group", r["candidate_id"]), []).append(i)
    keys = sorted(groups); np.random.default_rng(seed).shuffle(keys)
    test_keys = set(keys[:int(round(len(keys) * test_frac))])
    is_test = np.zeros(len(rows), bool)
    for g, ii in groups.items():
        if g in test_keys:
            is_test[ii] = True
    return is_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rows-jsonl", default=None)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    rows = [json.loads(l) for l in Path(args.rows_jsonl
            or f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rows])
    pv = np.array([r.get("prior_consistent") is False for r in rows])
    pc = np.array([r.get("prior_consistent") is True for r in rows])
    has_prior = pv.any() or pc.any()

    delta = np.load(out_dir / "delta.npy")                       # (N, n_layers, D)
    base_pred = np.load(out_dir / "baseline_pred.npy")
    preds = np.load(out_dir / "sweep_preds.npy")                 # (n_cells, N)
    meta = [json.loads(l) for l in (out_dir / "sweep_pred_meta.jsonl").read_text().splitlines() if l.strip()]
    ci_idx = next(i for i, m in enumerate(meta)
                  if m["tag"] == "real" and m["layer"] == args.layer and m["alpha"] == args.alpha)
    steered = preds[ci_idx]

    is_test = grouped_split(rows, args.seed, args.test_frac); dev = ~is_test
    gate_score = np.linalg.norm(delta[:, args.layer, :].astype(np.float32), axis=1)

    def acc(pred, m):
        return float((pred[m] == ci[m]).mean()) if m.sum() else float("nan")

    def gated_pred(fire):
        return np.where(fire, steered, base_pred)

    def report(pred, m, label):
        s = f"{label:16} overall={acc(pred,m):.3f}"
        if has_prior:
            s += f"  pv={acc(pred,m&pv):.3f}  pc={acc(pred,m&pc):.3f}"
        return s

    print(f"=== {args.benchmark}: selectivity gate (||delta(L{args.layer})||), steer cell L{args.layer} a={args.alpha} ===")
    print(f"split: dev {dev.sum()} / test {is_test.sum()}\n")
    print(report(base_pred, is_test, "baseline (test)"))
    print(report(steered,   is_test, "uncond (test)"))

    qs = [round(x, 2) for x in np.linspace(0, 1, 11)]
    for direction in ("high", "low"):
        print(f"\n-- gate: steer the {direction}est-||delta|| fraction --")
        print(f"{'frac':>5}{'dev_ov':>8}{'dev_pv':>8}{'dev_pc':>8}{'tst_ov':>8}{'tst_pv':>8}{'tst_pc':>8}")
        best = None
        for q in qs:
            thr = np.quantile(gate_score, 1 - q if direction == "high" else q)
            fire = gate_score >= thr if direction == "high" else gate_score <= thr
            if q == 0:
                fire = np.zeros(len(rows), bool)
            gp = gated_pred(fire)
            dov, tov = acc(gp, dev), acc(gp, is_test)
            row = (f"{q:>5.2f}{dov:>8.3f}{acc(gp,dev&pv):>8.3f}{acc(gp,dev&pc):>8.3f}"
                   f"{tov:>8.3f}{acc(gp,is_test&pv):>8.3f}{acc(gp,is_test&pc):>8.3f}") if has_prior else \
                  (f"{q:>5.2f}{dov:>8.3f}{'':>16}{tov:>8.3f}")
            print(row)
            if best is None or dov > best[1]:
                best = (q, dov, gp)
        q, _, gp = best
        print(f"  dev-selected frac={q:.2f} -> TEST: {report(gp, is_test, '')}")

    # ---- prior-reliance gate (needs no-image preds): steer items where the image
    #      did NOT change the answer (base_pred == noimg_pred) and the prior is
    #      confident (noimg_margin >= threshold) -- the text-biased subpopulation.
    npred_path = out_dir / "noimg_pred.npy"
    if not npred_path.exists():
        print("\n[prior-reliance gate skipped: noimg_pred.npy missing -> re-run --stage collect]")
        return
    noimg_pred = np.load(npred_path)
    noimg_margin = np.load(out_dir / "noimg_margin.npy")
    image_insensitive = base_pred == noimg_pred
    print(f"\n-- gate: prior-reliance (image-insensitive & confident-prior) --")
    print(f"   image-insensitive items: {image_insensitive.mean():.2f} of pool")
    print(f"{'m_thr':>6}{'fire%':>7}{'dev_ov':>8}{'dev_pv':>8}{'dev_pc':>8}{'tst_ov':>8}{'tst_pv':>8}{'tst_pc':>8}")
    best = None
    for mt in [round(x, 2) for x in np.linspace(0, 0.9, 10)]:
        fire = image_insensitive & (noimg_margin >= mt)
        gp = gated_pred(fire)
        dov = acc(gp, dev)
        row = (f"{mt:>6.2f}{fire.mean()*100:>7.1f}{dov:>8.3f}{acc(gp,dev&pv):>8.3f}{acc(gp,dev&pc):>8.3f}"
               f"{acc(gp,is_test):>8.3f}{acc(gp,is_test&pv):>8.3f}{acc(gp,is_test&pc):>8.3f}") if has_prior else \
              (f"{mt:>6.2f}{fire.mean()*100:>7.1f}{dov:>8.3f}{'':>16}{acc(gp,is_test):>8.3f}")
        print(row)
        if best is None or dov > best[1]:
            best = (mt, dov, gp)
    mt, _, gp = best
    print(f"  dev-selected m_thr={mt:.2f} -> TEST: {report(gp, is_test, '')}")


if __name__ == "__main__":
    main()
