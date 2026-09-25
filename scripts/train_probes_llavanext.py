"""LLaVA-NeXT variant of 07: per-layer logistic probe on D_M last-token activations."""
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
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    act_dir = Path(cfg["paths"]["data_root"]) / "activations_llavanext" / "dm"
    out_dir = Path(cfg["paths"]["data_root"]) / "probes_llavanext"
    out_dir.mkdir(parents=True, exist_ok=True)

    acts = np.load(act_dir / "activations.npy")
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    n, n_layers, d = acts.shape
    print(f"activations {acts.shape}  index rows {len(index)}")

    labels = np.array([0 if r["label"] == "vision" else 1 for r in index])
    splits = np.array([r["split"] for r in index])
    train_mask = splits == "train_pass"
    val_mask   = splits == "val_pass"
    print(f"train_pass={train_mask.sum()}  val_pass={val_mask.sum()}  test={(splits=='test').sum()}")

    val_y = labels[val_mask]
    chance_val = max((val_y == 0).mean(), (val_y == 1).mean())
    print(f"val chance: {chance_val:.3f}")

    rows = []
    for L in range(n_layers):
        X_tr = acts[train_mask, L].astype(np.float32)
        X_va = acts[val_mask,   L].astype(np.float32)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X_tr, labels[train_mask])
        p_va = clf.predict_proba(X_va)[:, 1]
        acc = accuracy_score(val_y, p_va > 0.5)
        try:
            auc = roc_auc_score(val_y, p_va)
        except ValueError:
            auc = float("nan")
        rows.append({"layer": L, "val_acc": float(acc), "val_auc": float(auc)})
        np.savez(out_dir / f"probe_layer{L:02d}.npz",
                 coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)

    sorted_rows = sorted(rows, key=lambda r: (-r["val_acc"], -(r["val_auc"] if r["val_auc"]==r["val_auc"] else 0)))
    picked = [sorted_rows[0]["layer"], sorted_rows[1]["layer"]]
    summary = {"n_layers": n_layers, "val_chance": float(chance_val),
               "per_layer": rows, "picked_layers": picked}
    (out_dir / "per_layer.json").write_text(json.dumps(summary, indent=2))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    layers = [r["layer"] for r in rows]
    accs = [r["val_acc"] for r in rows]; aucs = [r["val_auc"] for r in rows]
    ax1.plot(layers, accs, marker="o", label="val accuracy")
    ax1.axhline(chance_val, color="grey", linestyle="--", linewidth=1, label=f"chance={chance_val:.2f}")
    ax1.axvline(24, color="red", alpha=0.3, label="L24 (SAE)")
    ax1.set_ylabel("accuracy"); ax1.set_xlabel("layer"); ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx(); ax2.plot(layers, aucs, marker="x", color="orange", label="val AUROC")
    ax2.set_ylabel("AUROC"); ax2.set_ylim(0, 1.05)
    for L in picked: ax1.axvline(L, color="green", alpha=0.3)
    fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15)); fig.tight_layout()
    fig.savefig(out_dir / "per_layer.png", dpi=120)
    print(f"summary -> {out_dir}/per_layer.json")
    print(f"plot    -> {out_dir}/per_layer.png")
    print(f"picked  : {picked}")


if __name__ == "__main__":
    main()
