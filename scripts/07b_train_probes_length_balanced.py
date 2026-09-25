"""Step 7b: length-balanced per-layer probe.

Same per-layer logistic regression as 07, but subsets train and val so the
caption_for_filter length distribution is matched between vision and text
classes. Confirms that probe accuracy survives when length is controlled.

Strategy: bin caption length into quintiles (over the combined pool per split);
within each bin, keep min(n_vision, n_text) of each class. The resulting
subset has identical marginal length distributions across the two classes.

Outputs to a separate dir so it doesn't clobber the unbalanced run.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def balanced_indices(rows: list[dict], lengths: np.ndarray, n_bins: int,
                     seed: int) -> list[int]:
    """Return indices of a length-balanced subset of `rows`."""
    labels = np.array([0 if r["label"] == "vision" else 1 for r in rows])
    # quantile bin edges over all rows
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = np.digitize(lengths, edges[1:-1])
    rng = random.Random(seed)
    keep: list[int] = []
    for b in range(n_bins):
        idx_b = np.where(bins == b)[0]
        idx_v = [i for i in idx_b if labels[i] == 0]
        idx_t = [i for i in idx_b if labels[i] == 1]
        k = min(len(idx_v), len(idx_t))
        if k == 0:
            continue
        rng.shuffle(idx_v); rng.shuffle(idx_t)
        keep.extend(idx_v[:k]); keep.extend(idx_t[:k])
    return sorted(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dm-dir", required=True,
                    help="needed to look up caption_for_filter per candidate")
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dm_dir = Path(args.dm_dir)

    # load activations + index
    acts = np.load(act_dir / "activations.npy")
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    n, n_layers, d = acts.shape

    # load caption_for_filter per candidate from dm_all
    dm_map = {}
    for l in (dm_dir / "dm_all.jsonl").read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        dm_map[r["candidate_id"]] = r["caption_for_filter"]

    # length per row (caption_for_filter word count)
    lengths = np.array([len(dm_map[r["candidate_id"]].split()) for r in index])
    labels = np.array([0 if r["label"] == "vision" else 1 for r in index])
    splits = np.array([r["split"] for r in index])

    # length-balance train_pass and val_pass independently
    keep_mask = np.zeros(n, dtype=bool)
    for split_name in ("train_pass", "val_pass"):
        sub_idx = np.where(splits == split_name)[0]
        sub_rows = [index[i] for i in sub_idx]
        sub_lens = lengths[sub_idx]
        kept_local = balanced_indices(sub_rows, sub_lens, args.n_bins, args.seed)
        keep_mask[sub_idx[kept_local]] = True

    # report balance stats
    print("=== length-balance report (caption_for_filter words) ===")
    for split_name in ("train_pass", "val_pass"):
        for label_val, name in [(0, "vision"), (1, "text")]:
            mask = (splits == split_name) & (labels == label_val) & keep_mask
            n_k = mask.sum()
            mean_k = lengths[mask].mean() if n_k else float("nan")
            mask_all = (splits == split_name) & (labels == label_val)
            n_all = mask_all.sum()
            mean_all = lengths[mask_all].mean() if n_all else float("nan")
            print(f"  {split_name} {name}: kept {n_k}/{n_all}, mean len {mean_k:.1f} (was {mean_all:.1f})")

    # train per-layer probe on balanced subset
    train_mask = (splits == "train_pass") & keep_mask
    val_mask   = (splits == "val_pass")   & keep_mask
    y_tr = labels[train_mask]
    y_va = labels[val_mask]
    chance_val = max((y_va == 0).mean(), (y_va == 1).mean()) if len(y_va) else float("nan")
    print(f"\ntrain n={train_mask.sum()}  val n={val_mask.sum()}  val chance={chance_val:.3f}")

    rows = []
    for L in range(n_layers):
        X_tr = acts[train_mask, L].astype(np.float32)
        X_va = acts[val_mask,   L].astype(np.float32)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        p_va = clf.predict_proba(X_va)[:, 1]
        acc = accuracy_score(y_va, p_va > 0.5)
        try:
            auc = roc_auc_score(y_va, p_va)
        except ValueError:
            auc = float("nan")
        rows.append({"layer": L, "val_acc": float(acc), "val_auc": float(auc)})
        np.savez(out_dir / f"probe_layer{L:02d}.npz",
                 coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)

    sorted_rows = sorted(rows, key=lambda r: (-r["val_acc"], -(r["val_auc"] if r["val_auc"]==r["val_auc"] else 0)))
    picked = [sorted_rows[0]["layer"], sorted_rows[1]["layer"]]

    summary = {
        "n_layers": n_layers,
        "val_chance": float(chance_val),
        "n_train_balanced": int(train_mask.sum()),
        "n_val_balanced": int(val_mask.sum()),
        "n_bins": args.n_bins,
        "per_layer": rows,
        "picked_layers": picked,
    }
    (out_dir / "per_layer.json").write_text(json.dumps(summary, indent=2))

    # plot
    fig, ax1 = plt.subplots(figsize=(9, 4))
    layers = [r["layer"] for r in rows]
    ax1.plot(layers, [r["val_acc"] for r in rows], marker="o", label="val accuracy")
    ax1.axhline(chance_val, color="grey", linestyle="--", linewidth=1, label=f"chance={chance_val:.2f}")
    ax1.set_ylabel("accuracy"); ax1.set_xlabel("layer"); ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(layers, [r["val_auc"] for r in rows], marker="x", color="orange", label="val AUROC")
    ax2.set_ylabel("AUROC"); ax2.set_ylim(0, 1.05)
    for L in picked:
        ax1.axvline(L, color="green", alpha=0.3)
    fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
    fig.tight_layout()
    fig.savefig(out_dir / "per_layer.png", dpi=120)
    print(f"\npicked layers: {picked}")
    print(f"per-layer summary -> {out_dir}/per_layer.json")
    print(f"plot              -> {out_dir}/per_layer.png")


if __name__ == "__main__":
    main()
