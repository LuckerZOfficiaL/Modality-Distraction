"""Step 30: per-item-alpha ceiling + recoverability predictability (offline).

Among PRIOR-ANCHORED failures (text-bias: base_pred==noimg_pred, base wrong), two
questions, both offline from the scripts/27 sweep:

(1) PER-ITEM-ALPHA CEILING: does picking the best (layer, alpha) per item recover
    more than a single fixed cell? Compare fixed-best-cell vs per-item ORACLE over the
    real grid, with a NULL-ORACLE (best over the random cells) to subtract
    multiple-comparison luck. real_oracle - null_oracle = genuine per-item headroom.

(2) PREDICTABILITY: can label-free per-item signals (||delta(L)||, base_margin,
    noimg_margin) predict which prior-anchored failures the deployable fixed cell
    recovers? If AUC >> 0.5 a gate could extract the recoverable subset.

    python scripts/30_recoverability.py --benchmark mmstar --layer 31 --alpha 2.0
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
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rows-jsonl", default=None)
    ap.add_argument("--layer", type=int, required=True, help="deployable fixed cell (dev-best)")
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    rows = [json.loads(l) for l in Path(args.rows_jsonl
            or f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rows])
    base = np.load(out_dir / "baseline_pred.npy")
    noimg = np.load(out_dir / "noimg_pred.npy")
    base_margin = np.load(out_dir / "base_margin.npy")
    noimg_margin = np.load(out_dir / "noimg_margin.npy")
    delta = np.load(out_dir / "delta.npy")
    preds = np.load(out_dir / "sweep_preds.npy")
    meta = [json.loads(l) for l in (out_dir / "sweep_pred_meta.jsonl").read_text().splitlines() if l.strip()]

    anch_fail = (base == noimg) & (base != ci)           # prior-anchored (text-bias) failures
    n = int(anch_fail.sum())
    print(f"=== {args.benchmark}: prior-anchored failures n={n} (pool {len(rows)}) ===")

    real_idx = [i for i, m in enumerate(meta) if m["tag"] == "real"]
    null_idx = [i for i, m in enumerate(meta) if m["tag"] == "random"]
    fixed_i = next(i for i, m in enumerate(meta)
                   if m["tag"] == "real" and m["layer"] == args.layer and m["alpha"] == args.alpha)

    def frac_recovered(pred_rows, mask):
        # pred_rows: list of cell-pred arrays; recovered if ANY cell is correct
        any_ok = np.zeros(len(rows), bool)
        for pi in pred_rows:
            any_ok |= (preds[pi] == ci)
        return float(any_ok[mask].mean())

    fixed = float((preds[fixed_i][anch_fail] == ci[anch_fail]).mean())
    real_oracle = frac_recovered(real_idx, anch_fail)
    null_oracle = frac_recovered(null_idx, anch_fail)
    print("\n(1) per-item-alpha ceiling on prior-anchored failures:")
    print(f"  fixed best cell (L{args.layer} a={args.alpha})   {fixed:.3f}")
    print(f"  per-item ORACLE over real grid ({len(real_idx)} cells) {real_oracle:.3f}")
    print(f"  per-item NULL-oracle (random grid)           {null_oracle:.3f}")
    print(f"  genuine per-item headroom (real-null oracle) {real_oracle-null_oracle:+.3f}")
    print(f"  headroom over fixed (real_oracle - fixed)    {real_oracle-fixed:+.3f}")

    # (2) predictability of the deployable fixed cell's recovery
    y = (preds[fixed_i][anch_fail] == ci[anch_fail]).astype(int)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        print(f"\n(2) predictability: too few in a class (rec={int(y.sum())}/{len(y)}) -- skipping AUC")
        return
    feats = {
        "dnorm_L": np.linalg.norm(delta[:, args.layer, :].astype(np.float32), axis=1)[anch_fail],
        "base_margin": base_margin[anch_fail],
        "noimg_margin": noimg_margin[anch_fail],
    }
    print(f"\n(2) predict 'fixed cell recovers it' among prior-anchored failures "
          f"(rec rate {y.mean():.3f}, n={len(y)}):")
    for name, f in feats.items():
        try:
            auc = roc_auc_score(y, f); auc = max(auc, 1 - auc)  # |direction|
        except ValueError:
            auc = float("nan")
        print(f"  univariate AUC  {name:14} {auc:.3f}")
    X = np.column_stack(list(feats.values()))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    print(f"  multivariate logistic 5-fold AUC            {roc_auc_score(y, proba):.3f}")


if __name__ == "__main__":
    main()
