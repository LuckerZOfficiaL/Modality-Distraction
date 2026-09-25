"""LLaVA-NeXT variant of 09: identify modality-preferring SAE latents.

Uses the released L24 SAE (lmms-lab/llama3-llava-next-8b-hf-sae-131k)
and LLaVA-NeXT D_M activations from
data/activations/llavanext/dm/activations.npy (last input-token, all 32 layers).

Outputs to data/features/llavanext/layer24/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sparsify import Sae


SAE_HUB = "lmms-lab/llama3-llava-next-8b-hf-sae-131k"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])

    act_path = data_root / "activations" / "llavanext" / "dm" / "activations.npy"
    idx_path = data_root / "activations" / "llavanext" / "dm" / "index.jsonl"
    out_dir = data_root / "features" / "llavanext" / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = Sae.load_from_hub(SAE_HUB, hookpoint=f"model.layers.{args.layer}").to(args.device).eval()
    print(f"loaded SAE  d_in={sae.d_in}  num_latents={sae.num_latents}  k={sae.cfg.k}")

    acts = np.load(act_path)
    index = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
    assert len(index) == acts.shape[0]

    keep = [i for i, r in enumerate(index) if r["split"] in ("train_pass", "val_pass")]
    print(f"using {len(keep)} rows (train_pass + val_pass)")
    x = torch.from_numpy(acts[keep, args.layer, :]).to(args.device).float()
    labels = np.array([0 if index[i]["label"] == "vision" else 1 for i in keep])

    with torch.no_grad():
        enc = sae.encode(x)
        top_acts = enc.top_acts.cpu().numpy()       # (N, k)
        top_indices = enc.top_indices.cpu().numpy() # (N, k)

    n, k = top_acts.shape
    d_sae = sae.num_latents
    z = np.zeros((n, d_sae), dtype=np.float32)
    for i in range(n):
        z[i, top_indices[i]] = top_acts[i]
    print(f"densified z shape {z.shape}")

    z_v = z[labels == 0]; z_t = z[labels == 1]
    mu_v = z_v.mean(axis=0); mu_t = z_t.mean(axis=0)
    rho = mu_v - mu_t
    fire_v = (z_v > 0).mean(axis=0); fire_t = (z_t > 0).mean(axis=0)

    auroc = np.full(d_sae, 0.5)
    var = z.var(axis=0)
    nontrivial = np.where(var > 1e-8)[0]
    print(f"alive (var>1e-8): {len(nontrivial)}/{d_sae}")
    for kk in nontrivial:
        try:
            auroc[kk] = roc_auc_score(labels, z[:, kk])
        except ValueError:
            pass

    rows = [{
        "k": int(kk), "rho": float(rho[kk]), "mu_v": float(mu_v[kk]), "mu_t": float(mu_t[kk]),
        "fire_v": float(fire_v[kk]), "fire_t": float(fire_t[kk]),
        "auroc": float(auroc[kk]), "alive": bool(var[kk] > 1e-8),
    } for kk in range(d_sae)]
    rows_sorted = sorted(rows, key=lambda r: -abs(r["rho"]))
    with (out_dir / "feature_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        for r in rows_sorted: w.writerow(r)

    alive = sum(r["alive"] for r in rows)
    alive_rows = [r for r in rows if r["alive"]]
    top_v = sorted(alive_rows, key=lambda r: -r["rho"])[: args.top_n]
    top_t = sorted(alive_rows, key=lambda r: r["rho"])[: args.top_n]

    # Jaccard between top-N vision and top-N text fire sets
    fire_v_sets = [set(np.where(z[:, r["k"]] > 0)[0].tolist()) for r in top_v]
    fire_t_sets = [set(np.where(z[:, r["k"]] > 0)[0].tolist()) for r in top_t]
    jacc = []
    for a in fire_v_sets:
        for b in fire_t_sets:
            u = a | b
            jacc.append(len(a & b) / len(u) if u else 0.0)
    j_mean = float(np.mean(jacc)) if jacc else float("nan")

    summary = {
        "layer": args.layer, "n_rows": len(keep),
        "n_vision": int((labels == 0).sum()), "n_text": int((labels == 1).sum()),
        "alive": alive, "top_vision": top_v, "top_text": top_t,
        "jaccard_v_x_t_mean": j_mean,
    }
    json.dump(summary, open(out_dir / "top_features.json", "w"), indent=2)

    print(f"\nTop vision-preferring (top 5):")
    for r in top_v[:5]:
        print(f"  k={r['k']:>6}  rho={r['rho']:+.3f}  fire_v={r['fire_v']:.2f}  fire_t={r['fire_t']:.2f}  auroc={r['auroc']:.3f}")
    print(f"\nTop text-preferring (top 5):")
    for r in top_t[:5]:
        print(f"  k={r['k']:>6}  rho={r['rho']:+.3f}  fire_v={r['fire_v']:.2f}  fire_t={r['fire_t']:.2f}  auroc={r['auroc']:.3f}")
    print(f"\nJaccard (mean pairwise, top-{args.top_n} V × top-{args.top_n} T): {j_mean:.3f}")

    plt.figure(figsize=(8, 4))
    plt.hist([r["rho"] for r in alive_rows], bins=80, color="steelblue", edgecolor="k", linewidth=0.3)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.xlabel("rho_k = mu_vision - mu_text")
    plt.ylabel("count of latents")
    plt.title(f"layer {args.layer} | alive={alive} | N_v={(labels==0).sum()} N_t={(labels==1).sum()}")
    plt.tight_layout()
    plt.savefig(out_dir / "rho_hist.png", dpi=120)
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
