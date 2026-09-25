"""Step 9: identify modality-preferring SAE latents.

Encode D_M activations (train-pass + val-pass) through the trained SAE at the
selected layer. For each latent k, compute:
  - mu_v, mu_t: mean activation on vision-grounded vs text-grounded rows
  - rho_k = mu_v - mu_t  (signed modality preference)
  - auroc(z_k; label)    (rank-based separability)
  - fire_rate_v / fire_rate_t  (fraction of rows where z_k > 0)

Outputs:
  data/features/layer<L>/feature_stats.csv     per-latent stats (sorted by |rho_k|)
  data/features/layer<L>/top_features.json     top-N vision-prefs and text-prefs
  data/features/layer<L>/rho_hist.png          histogram of rho_k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from moground.sae import TopKSAE, SAEConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--variant", default="cc3m_1M",
                    help="SAE pretraining variant (subdir under data/sae/qwen/)")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", nargs="+", default=["train_pass"],
                    help="which split labels in index.jsonl to use for feature stats. "
                         "Default: train_pass (no val/test mixing — clean policy).")
    ap.add_argument("--act-dir", default=None,
                    help="override activations dir (default: data_root/activations/qwen/dm)")
    ap.add_argument("--out-dir", default=None,
                    help="override features output dir (default: data_root/features/qwen/<variant>/layer<L>)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])

    ckpt_path = data_root / "sae" / "qwen" / args.variant / f"layer{args.layer}" / "sae.pt"
    act_dir = Path(args.act_dir) if args.act_dir else data_root / "activations" / "qwen" / "dm"
    act_path = act_dir / "activations.npy"
    idx_path = act_dir / "index.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "features" / "qwen" / args.variant / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
    sae.load_state_dict(ck["state_dict"])
    sae.to(args.device).to(torch.float32).eval()
    print(f"loaded SAE  d_model={ck['d_model']}  d_sae={ck['d_sae']}  k={ck['k']}  layer={ck['layer']}")

    acts = np.load(act_path)  # (N, n_layers, d_model) float16
    index = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
    assert len(index) == acts.shape[0]

    keep = [i for i, r in enumerate(index) if r["split"] in args.splits]
    print(f"using {len(keep)} rows from splits={args.splits}")
    x = torch.from_numpy(acts[keep, args.layer, :]).to(args.device).float()
    labels = np.array([0 if index[i]["label"] == "vision" else 1 for i in keep])

    with torch.no_grad():
        out = sae(x)
        z = out["z"].cpu().numpy()  # (N, d_sae), zeros outside top-K
    print(f"encoded activations -> z shape {z.shape}")

    z_v = z[labels == 0]
    z_t = z[labels == 1]
    mu_v = z_v.mean(axis=0)
    mu_t = z_t.mean(axis=0)
    rho = mu_v - mu_t
    fire_v = (z_v > 0).mean(axis=0)
    fire_t = (z_t > 0).mean(axis=0)

    # auroc per latent (skip if z_k is constant)
    auroc = np.full(z.shape[1], 0.5)
    var = z.var(axis=0)
    nontrivial = np.where(var > 1e-8)[0]
    for k in nontrivial:
        try:
            auroc[k] = roc_auc_score(labels, z[:, k])
        except ValueError:
            pass

    rows = []
    for k in range(z.shape[1]):
        rows.append({
            "k": int(k), "rho": float(rho[k]), "mu_v": float(mu_v[k]), "mu_t": float(mu_t[k]),
            "fire_v": float(fire_v[k]), "fire_t": float(fire_t[k]),
            "auroc": float(auroc[k]), "alive": bool(var[k] > 1e-8),
        })
    # CSV
    import csv
    csv_path = out_dir / "feature_stats.csv"
    rows_sorted = sorted(rows, key=lambda r: -abs(r["rho"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"wrote {csv_path}")

    alive = sum(r["alive"] for r in rows)
    print(f"alive (var > 1e-8) = {alive} / {len(rows)}")

    # vision-preferring (rho > 0) and text-preferring (rho < 0), among alive
    alive_rows = [r for r in rows if r["alive"]]
    top_v = sorted(alive_rows, key=lambda r: -r["rho"])[: args.top_n]
    top_t = sorted(alive_rows, key=lambda r: r["rho"])[: args.top_n]
    print("\nTop vision-preferring:")
    print(f"{'k':>6} {'rho':>9} {'mu_v':>8} {'mu_t':>8} {'fire_v':>7} {'fire_t':>7} {'auroc':>6}")
    for r in top_v[:10]:
        print(f"{r['k']:>6} {r['rho']:>9.3f} {r['mu_v']:>8.3f} {r['mu_t']:>8.3f} "
              f"{r['fire_v']:>7.3f} {r['fire_t']:>7.3f} {r['auroc']:>6.3f}")
    print("\nTop text-preferring:")
    print(f"{'k':>6} {'rho':>9} {'mu_v':>8} {'mu_t':>8} {'fire_v':>7} {'fire_t':>7} {'auroc':>6}")
    for r in top_t[:10]:
        print(f"{r['k']:>6} {r['rho']:>9.3f} {r['mu_v']:>8.3f} {r['mu_t']:>8.3f} "
              f"{r['fire_v']:>7.3f} {r['fire_t']:>7.3f} {r['auroc']:>6.3f}")

    json.dump({
        "layer": args.layer,
        "n_rows": len(keep),
        "n_vision": int((labels == 0).sum()),
        "n_text": int((labels == 1).sum()),
        "alive": alive,
        "top_vision": top_v,
        "top_text": top_t,
    }, open(out_dir / "top_features.json", "w"), indent=2)
    print(f"wrote {out_dir/'top_features.json'}")

    # histogram of rho
    plt.figure(figsize=(8, 4))
    plt.hist([r["rho"] for r in alive_rows], bins=80, color="steelblue", edgecolor="k", linewidth=0.3)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.xlabel("rho_k = mu_vision - mu_text")
    plt.ylabel("count of latents")
    plt.title(f"layer {args.layer} | alive={alive} | N_v={int((labels==0).sum())} N_t={int((labels==1).sum())}")
    plt.tight_layout()
    plt.savefig(out_dir / "rho_hist.png", dpi=120)
    print(f"wrote {out_dir/'rho_hist.png'}")


if __name__ == "__main__":
    main()
