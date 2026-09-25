"""Step 17: re-score SAE features by within-item causal contribution.

Replace the discriminative score
    ρ_k = μ_v(z_k(h_VT)) - μ_t(z_k(h_VT))
which finds atoms that DISCRIMINATE V-grounded from T-grounded items in V+T
activations (= question-encoding axis per Section probe-A-ablation), with
the within-item counterfactual score
    ρ_k^causal_V = E_i [ z_k(h_V_i, L) - z_k(h_VT_i, L) ]   over V-grounded train_pass
    ρ_k^causal_T = E_i [ z_k(h_T_i, L) - z_k(h_VT_i, L) ]   over T-grounded train_pass

ρ_causal_V > 0 = atoms whose firing INCREASES when the caption is removed
from a V-grounded item = "what does the model do with image alone." These
are the atoms whose amplification (under our +α = vision convention) should
push h_VT toward h_V on V-grounded items.

Symmetric for ρ_causal_T.

The discriminative top-V atoms encode question-content. The causal top-V
atoms encode the model's image-only response. These can be very different
sets — the central hypothesis is that the causal atoms are the right targets
for SAE-mediated steering, while the discriminative ones explain the random-
null collapse of the prior SAE bcast+nm result.

Output (per layer):
  data/features/qwen/diversified_v1_causal/layer<L>/feature_stats.csv
  data/features/qwen/diversified_v1_causal/layer<L>/top_features.json
Same schema as step 09 so 13_sae_steering.py picks it up via --features-dir.
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

from moground.sae import TopKSAE, SAEConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--variant", default="diversified_v1",
                    help="existing SAE pretrain variant (subdir under data/sae/qwen/)")
    ap.add_argument("--layers", default="13,20,28",
                    help="which layers' SAEs to re-score")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-variant", default="diversified_v1_causal",
                    help="output subdir under data/features/qwen/")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy")     # (N, 3, n_layers, d) [VT, V, T]
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    print(f"loaded counterfactual activations: {acts.shape}")

    is_tp = np.array([r["split"] == "train_pass" for r in index])
    is_V  = np.array([r["label"] == "vision" for r in index]) & is_tp
    is_T  = np.array([r["label"] == "text"   for r in index]) & is_tp
    print(f"train_pass: V={int(is_V.sum())}  T={int(is_T.sum())}")

    layers = [int(x) for x in args.layers.split(",")]

    for L in layers:
        print(f"\n=== layer {L} ===")
        ckpt_path = data_root / "sae" / "qwen" / args.variant / f"layer{L}" / "sae.pt"
        if not ckpt_path.exists():
            print(f"  SAE missing at {ckpt_path} — skip")
            continue
        ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
        sae.load_state_dict(ck["state_dict"])
        sae.to(args.device).to(torch.float32).eval()
        print(f"  SAE  d_model={ck['d_model']}  d_sae={ck['d_sae']}  k={ck['k']}")

        # Encode V+T, V-only, T-only activations through SAE
        def encode(idx_mask, cond):
            X = torch.from_numpy(acts[idx_mask, cond, L].astype(np.float32)).to(args.device)
            with torch.no_grad():
                z = sae(X)["z"]  # (n, d_sae)
            return z.cpu().numpy()

        # V-grounded items: z under V+T and V-only
        zVT_V = encode(is_V, 0)   # (n_V, d_sae)
        zV_V  = encode(is_V, 1)
        # T-grounded items: z under V+T and T-only
        zVT_T = encode(is_T, 0)
        zT_T  = encode(is_T, 2)

        # Causal score: per-atom mean Δ
        rho_V_causal = (zV_V - zVT_V).mean(axis=0)   # V-grounded: caption removed
        rho_T_causal = (zT_T - zVT_T).mean(axis=0)   # T-grounded: image removed

        # Auxiliary: firing rates (any condition non-zero) for "alive" flag
        fire_VT_V = (zVT_V > 0).mean(axis=0)
        fire_V_V  = (zV_V  > 0).mean(axis=0)
        fire_VT_T = (zVT_T > 0).mean(axis=0)
        fire_T_T  = (zT_T  > 0).mean(axis=0)
        var_VT = np.concatenate([zVT_V, zVT_T], axis=0).var(axis=0)
        alive = var_VT > 1e-8

        rows = []
        for k in range(sae.cfg.d_sae):
            rows.append({
                "k": int(k),
                "rho_V_causal": float(rho_V_causal[k]),
                "rho_T_causal": float(rho_T_causal[k]),
                "fire_VT_V": float(fire_VT_V[k]),
                "fire_V_V":  float(fire_V_V[k]),
                "fire_VT_T": float(fire_VT_T[k]),
                "fire_T_T":  float(fire_T_T[k]),
                "alive": bool(alive[k]),
            })

        out_dir = data_root / "features" / "qwen" / args.out_variant / f"layer{L}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sort by |rho_V_causal| for the CSV
        rows_sorted = sorted(rows, key=lambda r: -abs(r["rho_V_causal"]))
        with open(out_dir / "feature_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            w.writeheader()
            for r in rows_sorted: w.writerow(r)

        # Top-N V-causal (highest rho_V_causal) and top-N T-causal (highest rho_T_causal)
        alive_rows = [r for r in rows if r["alive"]]
        top_V = sorted(alive_rows, key=lambda r: -r["rho_V_causal"])[: args.top_n]
        top_T = sorted(alive_rows, key=lambda r: -r["rho_T_causal"])[: args.top_n]
        # Adapt to schema 13_sae_steering.py expects: dicts with "k" and "rho" keys
        def adapt(lst, score_field):
            return [{**r, "rho": r[score_field]} for r in lst]
        json.dump({
            "layer": L,
            "scoring": "causal (within-item Δ)",
            "n_V": int(is_V.sum()),
            "n_T": int(is_T.sum()),
            "alive": int(alive.sum()),
            "top_vision": adapt(top_V, "rho_V_causal"),
            "top_text":   adapt(top_T, "rho_T_causal"),
        }, open(out_dir / "top_features.json", "w"), indent=2)
        print(f"  alive={int(alive.sum())}  best rho_V_causal={top_V[0]['rho_V_causal']:+.4f} "
              f"(k={top_V[0]['k']})  best rho_T_causal={top_T[0]['rho_T_causal']:+.4f} "
              f"(k={top_T[0]['k']})")

        # Quick comparison: overlap with discriminative top-N
        disc_path = data_root / "features" / "qwen" / args.variant / f"layer{L}" / "top_features.json"
        if disc_path.exists():
            disc = json.loads(disc_path.read_text())
            disc_V = {r["k"] for r in disc["top_vision"][: args.top_n]}
            disc_T = {r["k"] for r in disc["top_text"][: args.top_n]}
            new_V = {r["k"] for r in top_V}
            new_T = {r["k"] for r in top_T}
            print(f"  overlap with discriminative top-{args.top_n}: "
                  f"V={len(new_V & disc_V)}/{args.top_n}  "
                  f"T={len(new_T & disc_T)}/{args.top_n}")

        # Histogram of rho_V_causal on alive
        plt.figure(figsize=(8, 4))
        plt.hist([r["rho_V_causal"] for r in alive_rows], bins=80, color="steelblue",
                 edgecolor="k", linewidth=0.3)
        plt.axvline(0, color="k", linewidth=0.8)
        plt.xlabel(r"$\rho_k^{causal, V} = \overline{z_k(h_V) - z_k(h_{VT})}$")
        plt.ylabel("count of alive latents")
        plt.title(f"layer {L} causal V-score histogram")
        plt.tight_layout()
        plt.savefig(out_dir / "rho_V_causal_hist.png", dpi=120)
        plt.close()

        print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
