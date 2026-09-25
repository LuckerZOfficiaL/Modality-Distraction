"""Compute Jaccard overlap of fire sets for top modality-preferring SAE
latents on D_M (train+val+test, all 315 rows). For each layer, encode
activations through the trained SAE, take z>0 as the fire indicator, and
compute pairwise Jaccard within top-vision, within top-text, and across
the two groups. Reports mean / median / max off-diagonal Jaccard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from moground.sae import TopKSAE, SAEConfig


def jaccard_matrix(F: np.ndarray) -> np.ndarray:
    # F: [n_features, n_rows] boolean
    Fi = F.astype(np.int32)
    inter = Fi @ Fi.T
    sizes = Fi.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return J


def offdiag(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return M[mask]


def main():
    cfg = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    data_root = Path(cfg["paths"]["data_root"])
    acts = np.load(data_root / "activations/dm/activations.npy")  # (315, 36, 2048)
    print(f"D_M activations: {acts.shape}")

    layers = [7, 11, 20]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for L in layers:
        sae_dir = data_root / f"sae/layer{L:02d}"
        ckpt = torch.load(sae_dir / "sae.pt", map_location=device, weights_only=False)
        scfg = SAEConfig(d_model=ckpt["d_model"], d_sae=ckpt["d_sae"], k=ckpt["k"])
        sae = TopKSAE(scfg).to(device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()

        x = torch.from_numpy(acts[:, L, :]).to(device).float()
        with torch.inference_mode():
            res = sae(x)
            z = res["z"].cpu().numpy()  # (315, 16384)
        fired = z > 0  # (315, 16384)

        feat_dir = data_root / f"features/layer{L:02d}"
        if not feat_dir.exists():
            feat_dir = data_root / f"features/layer{L}"
        tf = json.loads((feat_dir / "top_features.json").read_text())
        ks_v = [t["k"] for t in tf["top_vision"]]
        ks_t = [t["k"] for t in tf["top_text"]]
        F_v = fired[:, ks_v].T  # (n_v, 315)
        F_t = fired[:, ks_t].T

        Jvv = jaccard_matrix(F_v)
        Jtt = jaccard_matrix(F_t)
        Fall = np.concatenate([F_v, F_t], axis=0)
        Jall = jaccard_matrix(Fall)
        n_v = F_v.shape[0]
        Jvt = Jall[:n_v, n_v:]  # cross-group, no diag

        def stats(arr):
            return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                    "max": float(arr.max()), "min": float(arr.min())}
        out[f"L{L}"] = {
            "fire_rate_vision_top": [float(F_v[i].mean()) for i in range(n_v)],
            "within_vision": stats(offdiag(Jvv)),
            "within_text": stats(offdiag(Jtt)),
            "cross_vision_text": stats(Jvt.flatten()),
            "top_vision_ks": ks_v,
            "top_text_ks": ks_t,
        }
        print(f"\nLayer {L}:")
        print(f"  within-vision  J (off-diag, n={n_v}): {out[f'L{L}']['within_vision']}")
        print(f"  within-text    J (off-diag, n={F_t.shape[0]}): {out[f'L{L}']['within_text']}")
        print(f"  cross v-vs-t   J ({n_v}x{F_t.shape[0]}):     {out[f'L{L}']['cross_vision_text']}")

    out_path = data_root / "results/jaccard_fire_sets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
