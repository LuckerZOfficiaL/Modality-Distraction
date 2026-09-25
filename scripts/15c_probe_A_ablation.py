"""Step 15c: ablate what probe A (h_VT V-vs-T) actually encodes.

Probe A hits val_acc ≈ 1.000 at L20+ but the random-null shows it's not a
causal steering direction. Length was previously ruled out. This script
attacks the remaining candidates:

  (1) PER-DOMAIN: does the probe still work when restricted to a single
      source (DCI / VisText / SemArt / ROCO)? If yes, V/T is encoded
      WITHIN each domain (not a between-domain artifact).
  (2) PER-CONDITION: re-fit A using h_V (image+question, no caption) and
      h_T (caption+question, no image) instead of h_VT.
        - If h_V(V-items) vs h_V(T-items) separates → label is decodable
          from image+question alone → A reads ITEM content, not V+T
          integration.
        - If h_T(V) vs h_T(T) separates → same, from caption+question.
  (3) WITHIN-DOMAIN: combine (1)+(2). Per-source per-condition.
  (4) Length residualization on the new counterfactual pool (replicates the
      old length-residualized check for completeness).

Output: data/probes/qwen/multidomain_v1_counterfactual/probe_A_ablation.json
+ stdout summary.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit(X_tr, y_tr, X_va, y_va, C=1.0):
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_tr, y_tr)), float(clf.score(X_va, y_va))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--dm-all",
                    default="data/dm/multidomain_v1/dm_all.jsonl",
                    help="for source-domain lookup per candidate_id")
    ap.add_argument("--out-dir",
                    default="data/probes/qwen/multidomain_v1_counterfactual")
    ap.add_argument("--layers", default="13,20,23,28",
                    help="layers to ablate (subset of all 36 to keep runtime sane)")
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy")   # (N, 3, n_layers, d)
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    N = len(index)
    assert acts.shape[0] == N
    print(f"loaded {N} rows, acts shape {acts.shape}")

    # source per candidate_id from dm_all
    src_by_cid = {}
    for line in Path(args.dm_all).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            src_by_cid[r["candidate_id"]] = r.get("source", "?")
    sources = np.array([src_by_cid.get(r["candidate_id"], "?") for r in index])

    # masks
    is_train = np.array([r["split"] == "train_pass" for r in index])
    is_val   = np.array([r["split"] == "val_pass"   for r in index])
    labels   = np.array([0 if r["label"] == "vision" else 1 for r in index])
    layers = [int(x) for x in args.layers.split(",")]
    cond_names = ["VT", "V", "T"]

    print(f"\ntrain_pass: {is_train.sum()}  val_pass: {is_val.sum()}")
    print(f"sources in train_pass: {dict(zip(*np.unique(sources[is_train], return_counts=True)))}")
    print(f"sources in val_pass:   {dict(zip(*np.unique(sources[is_val], return_counts=True)))}")
    # V/T balance per source
    print("V/T balance per source (train_pass):")
    for s in sorted(set(sources[is_train])):
        m = is_train & (sources == s)
        nv = int(((labels == 0) & m).sum()); nt = int(((labels == 1) & m).sum())
        print(f"  {s:>10s}: V={nv:>4d} T={nt:>4d}")

    results = {"per_layer": {}}

    for L in layers:
        print(f"\n==== L{L} ====")
        layer_res = {}

        # (0) baseline A on full pool, three conditions
        for ci, cond in enumerate(cond_names):
            Xtr = acts[is_train, ci, L].astype(np.float32)
            ytr = labels[is_train]
            Xva = acts[is_val, ci, L].astype(np.float32)
            yva = labels[is_val]
            tr, va = fit(Xtr, ytr, Xva, yva)
            layer_res[f"full_pool_{cond}"] = {"train": tr, "val": va,
                                              "n_train": int(is_train.sum()),
                                              "n_val": int(is_val.sum())}
            print(f"  full pool, cond={cond}:  train={tr:.3f}  val={va:.3f}")

        # (1) per-domain, per-condition
        for src in sorted(set(sources)):
            for ci, cond in enumerate(cond_names):
                tr_mask = is_train & (sources == src)
                va_mask = is_val   & (sources == src)
                # need both classes present in train AND val
                if (labels[tr_mask] == 0).sum() < 5 or (labels[tr_mask] == 1).sum() < 5: continue
                if (labels[va_mask] == 0).sum() < 5 or (labels[va_mask] == 1).sum() < 5: continue
                Xtr = acts[tr_mask, ci, L].astype(np.float32)
                ytr = labels[tr_mask]
                Xva = acts[va_mask, ci, L].astype(np.float32)
                yva = labels[va_mask]
                tr, va = fit(Xtr, ytr, Xva, yva)
                layer_res[f"src={src}_cond={cond}"] = {
                    "train": tr, "val": va,
                    "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
                    "n_val_V": int(((labels==0) & va_mask).sum()),
                    "n_val_T": int(((labels==1) & va_mask).sum()),
                }
                print(f"  src={src:<8s} cond={cond}:  train={tr:.3f}  val={va:.3f}  "
                      f"(n_val={int(va_mask.sum())})")

        # (2) cross-domain transfer: train on 3 sources, test on the 4th
        # this isolates "is the label-axis source-dependent?"
        if len(set(sources)) >= 2:
            for hold in sorted(set(sources)):
                if hold == "?": continue
                tr_mask = is_train & (sources != hold)
                va_mask = is_val   & (sources == hold)
                if va_mask.sum() < 20: continue
                if (labels[va_mask] == 0).sum() < 5 or (labels[va_mask] == 1).sum() < 5: continue
                Xtr = acts[tr_mask, 0, L].astype(np.float32)  # VT
                ytr = labels[tr_mask]
                Xva = acts[va_mask, 0, L].astype(np.float32)
                yva = labels[va_mask]
                tr, va = fit(Xtr, ytr, Xva, yva)
                layer_res[f"holdout={hold}_VT"] = {
                    "train": tr, "val": va,
                    "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
                }
                print(f"  holdout={hold:<8s} (train on others, eval on {hold}):  "
                      f"train={tr:.3f}  val={va:.3f}")

        results["per_layer"][L] = layer_res

    out_path = Path(args.out_dir) / "probe_A_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
