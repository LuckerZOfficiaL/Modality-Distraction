"""Step 36: global-vector steering — build v_global and test the cosine gate (offline).

Approach 2 core analysis. Steering vector:
  v_global(L) = unit( mean over DEV vision items of (h_img - h_noimg)(L) )    [benchmark source]
i.e. the average WITHIN-ITEM image-counterfactual = the "use-the-image" direction.
NOT from de-confounded D_M: the within-item counterfactual already cancels question/
caption surface by construction, so de-confounding is unnecessary; averaging is the
de-noising (cancels per-image content, keeps the image-presence/use component).

Ablation sources for v_global: 'benchmark' (dev image-counterfactual), 'dm' (D_M
image-removal h_VT - h_T from dm_matched_deconf_v2_all), 'instruction' (scripts/20).

Core gate test (the make-or-break): rank inputs by cos(h_img, v_global); does it predict
per-item STEERABILITY (does steering fix vs break the item)? Uses the existing per-item-δ
sweep_preds as the steerability outcome (proxy until the global-vector GPU eval is run).
Reports AUC of each geometric signal vs fix/break, BOTH gate directions, per benchmark
and on ViLP prior-violating. AUC notably > 0.5 => a usable cosine threshold gate.

Needs h_img.npy (scripts/27 --stage collect, re-collected). h_noimg = h_img - delta.

    python scripts/36_global_vector_gate.py --benchmark vilp
    python scripts/36_global_vector_gate.py --benchmark mmstar --vglobal-source dm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


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


def auc1(x, y):
    try:
        a = roc_auc_score(y, x); return max(a, 1 - a)
    except ValueError:
        return float("nan")


def build_vglobal(source, L, delta, dev_mask, dm_store):
    if source == "benchmark":
        v = delta[dev_mask, L].mean(0)
    elif source == "dm":
        acts = np.load(Path(dm_store) / "activations.npy", mmap_mode="r")  # (N,3,*,d) [VT,V,T]
        v = (acts[:, 0, L].astype(np.float32) - acts[:, 2, L].astype(np.float32)).mean(0)  # image-removal
    else:
        raise ValueError(f"unknown vglobal source {source}")
    return v / (np.linalg.norm(v) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rows-jsonl", default=None)
    ap.add_argument("--layers", default="13,20,28,31")
    ap.add_argument("--vglobal-source", default="benchmark", choices=["benchmark", "dm"])
    ap.add_argument("--dm-store", default="data/activations/qwen/dm_matched_deconf_v2_all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    if not (out_dir / "h_img.npy").exists():
        raise SystemExit(f"{out_dir/'h_img.npy'} missing — re-run scripts/27 --stage collect first.")
    rows = [json.loads(l) for l in Path(args.rows_jsonl
            or f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rows])
    pv = np.array([r.get("prior_consistent") is False for r in rows])
    delta = np.load(out_dir / "delta.npy")
    h_img = np.load(out_dir / "h_img.npy")
    base = np.load(out_dir / "baseline_pred.npy")
    preds = np.load(out_dir / "sweep_preds.npy")
    meta = [json.loads(l) for l in (out_dir / "sweep_pred_meta.jsonl").read_text().splitlines() if l.strip()]

    is_test = grouped_split(rows, args.seed, args.test_frac); dev = ~is_test
    bc = base == ci

    # steerability outcome: dev-best per-item-δ real cell (proxy until global-vector GPU eval)
    real = [(i, m) for i, m in enumerate(meta) if m["tag"] == "real"]
    def cell_devacc(i):
        return (np.where(dev, preds[i] == ci, False)).sum()
    sel = max(real, key=lambda im: ((preds[im[0]][dev] == ci[dev]).mean()))[0]
    sc = preds[sel] == ci
    print(f"=== {args.benchmark} | v_global source={args.vglobal_source} | "
          f"steerability outcome = per-item-δ cell {meta[sel]} ===")

    layers = [int(x) for x in args.layers.split(",")]
    print(f"{'L':>3} {'AUC cos(h_img,vg)':>18} {'AUC ||delta||':>14} {'AUC cos(delta,vg)':>18} {'n_dec(test)':>11}")
    for L in layers:
        vg = build_vglobal(args.vglobal_source, L, delta, dev, args.dm_store)
        hN = h_img[:, L].astype(np.float32)
        cos_hv = (hN @ vg) / (np.linalg.norm(hN, axis=1) + 1e-8)
        dn = np.linalg.norm(delta[:, L].astype(np.float32), axis=1)
        d = delta[:, L].astype(np.float32)
        cos_dv = (d @ vg) / (np.linalg.norm(d, axis=1) + 1e-8)
        dec = is_test & (sc != bc)
        y = sc[dec].astype(int)
        if y.sum() < 8 or (1 - y).sum() < 8:
            print(f"{L:>3}  (test-decisive too small: {int(dec.sum())}, fix={int(y.sum())}) — using all decisive")
            dec = (sc != bc); y = sc[dec].astype(int)
        print(f"{L:>3} {auc1(cos_hv[dec], y):>18.3f} {auc1(dn[dec], y):>14.3f} "
              f"{auc1(cos_dv[dec], y):>18.3f} {int((is_test&(sc!=bc)).sum()):>11}")

    # prior-violating view (ViLP)
    if pv.any():
        L = layers[len(layers) // 2 + 1] if len(layers) > 2 else layers[-1]
        vg = build_vglobal(args.vglobal_source, L, delta, dev, args.dm_store)
        hN = h_img[:, L].astype(np.float32)
        cos_hv = (hN @ vg) / (np.linalg.norm(hN, axis=1) + 1e-8)
        decpv = pv & (sc != bc)
        if decpv.sum() >= 16:
            print(f"\nViLP prior-violating @L{L}: AUC cos(h_img,vg) vs fix/break = "
                  f"{auc1(cos_hv[decpv], sc[decpv].astype(int)):.3f}  (n_dec={int(decpv.sum())})")


if __name__ == "__main__":
    main()
