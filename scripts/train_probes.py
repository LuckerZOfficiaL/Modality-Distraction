"""Step 7: per-layer linear probe (vision vs text) on D_M activations.

Train on D_M,train-pass; evaluate on D_M,val-pass. Output:
  data/probes/per_layer.json    accuracy + AUROC + chosen layers
  data/probes/per_layer.png     accuracy and AUROC vs layer
  data/probes/probe_layer<L>.npz   trained sklearn probe coefficients per layer

Probe pick: top-2 layers by val accuracy, tie-broken by AUROC.

With small N (train ~48, val ~16), use logistic regression with L2 and
class_weight='balanced'. Report chance (= max class freq).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--act-dir", default=None,
                    help="override activations dir (default: data_root/activations/qwen/dm)")
    ap.add_argument("--out-dir", default=None,
                    help="override probes output dir (default: data_root/probes/qwen)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    act_dir = Path(args.act_dir) if args.act_dir else Path(cfg["paths"]["data_root"]) / "activations" / "qwen" / "dm"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["data_root"]) / "probes" / "qwen"
    out_dir.mkdir(parents=True, exist_ok=True)

    acts = np.load(act_dir / "activations.npy")  # (N, L, D)
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    n, n_layers, d = acts.shape
    print(f"loaded activations {acts.shape}  with {len(index)} index rows")

    labels = np.array([0 if r["label"] == "vision" else 1 for r in index])
    splits = np.array([r["split"] for r in index])
    train_mask = splits == "train_pass"
    val_mask   = splits == "val_pass"
    test_mask  = splits == "test"
    print(f"train_pass={train_mask.sum()}  val_pass={val_mask.sum()}  test={test_mask.sum()}")

    # chance for val
    val_y = labels[val_mask]
    chance_val = max((val_y == 0).mean(), (val_y == 1).mean())
    print(f"val chance: {chance_val:.3f}")

    rows = []
    for L in range(n_layers):
        X_tr = acts[train_mask, L].astype(np.float32)
        X_va = acts[val_mask,   L].astype(np.float32)
        y_tr = labels[train_mask]
        y_va = val_y
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        p_va = clf.predict_proba(X_va)[:, 1]
        acc = accuracy_score(y_va, p_va > 0.5)
        try:
            auc = roc_auc_score(y_va, p_va)
        except ValueError:
            auc = float("nan")
        rows.append({"layer": L, "val_acc": float(acc), "val_auc": float(auc)})
        # save coefficients
        np.savez(out_dir / f"probe_layer{L:02d}.npz",
                 coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)

    # pick top 2 layers
    sorted_rows = sorted(rows, key=lambda r: (-r["val_acc"], -(r["val_auc"] if r["val_auc"]==r["val_auc"] else 0)))
    picked = [sorted_rows[0]["layer"], sorted_rows[1]["layer"]]

    summary = {
        "n_layers": n_layers,
        "val_chance": float(chance_val),
        "per_layer": rows,
        "picked_layers": picked,
    }
    (out_dir / "per_layer.json").write_text(json.dumps(summary, indent=2))

    # plot
    fig, ax1 = plt.subplots(figsize=(9, 4))
    layers = [r["layer"] for r in rows]
    accs = [r["val_acc"] for r in rows]
    aucs = [r["val_auc"] for r in rows]
    ax1.plot(layers, accs, marker="o", label="val accuracy")
    ax1.axhline(chance_val, color="grey", linestyle="--", linewidth=1, label=f"chance={chance_val:.2f}")
    ax1.set_ylabel("accuracy"); ax1.set_xlabel("layer")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(layers, aucs, marker="x", color="orange", label="val AUROC")
    ax2.set_ylabel("AUROC"); ax2.set_ylim(0, 1.05)
    for L in picked:
        ax1.axvline(L, color="green", alpha=0.3)
    fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
    fig.tight_layout()
    fig.savefig(out_dir / "per_layer.png", dpi=120)
    print(f"per-layer summary -> {out_dir}/per_layer.json")
    print(f"plot              -> {out_dir}/per_layer.png")
    print(f"picked layers (top-2 by val_acc, AUROC tiebreak): {picked}")


if __name__ == "__main__":
    main()
