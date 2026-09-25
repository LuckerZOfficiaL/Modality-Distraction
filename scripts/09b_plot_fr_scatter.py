"""Step 9b: scatter fr_v vs fr_t per layer for the trained SAEs.

Reads feature_stats.csv from step 09 and produces one scatter per layer.
Diagonal = modality-balanced; deviations = modality-preferring.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--variant", default="cc3m_1M")
    ap.add_argument("--layers", type=int, nargs="+", default=[13, 20, 28])
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    feat_root = Path(cfg["paths"]["data_root"]) / "features" / "qwen" / args.variant

    fig, axes = plt.subplots(1, len(args.layers), figsize=(5 * len(args.layers), 5), squeeze=False)
    for ax, L in zip(axes[0], args.layers):
        rows = list(csv.DictReader(open(feat_root / f"layer{L}" / "feature_stats.csv")))
        alive = [r for r in rows if r["alive"] == "True"]
        fr_v = [float(r["fire_v"]) for r in alive]
        fr_t = [float(r["fire_t"]) for r in alive]
        rho = [float(r["rho"]) for r in alive]
        ax.scatter(fr_v, fr_t, c=rho, cmap="coolwarm", s=10, alpha=0.7,
                   vmin=-max(abs(min(rho)), max(rho)), vmax=max(abs(min(rho)), max(rho)))
        # label top-N by |rho|
        ranked = sorted(zip(fr_v, fr_t, rho, [r["k"] for r in alive]),
                        key=lambda t: -abs(t[2]))[: args.top_n]
        for fv, ft, _, k in ranked:
            ax.annotate(k, (fv, ft), fontsize=7, alpha=0.8)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("fr_v (firing rate on vision rows)")
        ax.set_ylabel("fr_t (firing rate on text rows)")
        ax.set_title(f"L{L}  alive={len(alive)}")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")

    out_path = feat_root / "fr_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
