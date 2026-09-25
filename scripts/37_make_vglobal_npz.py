"""Step 37: write the D_M-derived global image-use vector as a harness npz.

v_global(L) = mean over D_M items of (h_VT - h_T)(L)  [image-removal counterfactual]
This is the cleaner "use-the-image" direction that gave the strongest cosine gate
(scripts/36: ViLP prior-violating AUC 0.872). Saved in the format scripts/19
(--method additive) consumes: keys "v_V_L{L}" for every layer, so the existing
benchmark sweep + matched-random null can run it unchanged. Gating is applied
offline afterward (scripts/38) using cos(h_img, v_global).

    python scripts/37_make_vglobal_npz.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-store", default="data/activations/qwen/dm_matched_deconf_v2_all")
    ap.add_argument("--out", default="data/steering/vglobal_dm_image.npz")
    args = ap.parse_args()

    acts = np.load(Path(args.dm_store) / "activations.npy", mmap_mode="r")  # (N,3,n_layers,d) [VT,V,T]
    n_layers = acts.shape[2]
    out = {}
    for L in range(n_layers):
        v = (acts[:, 0, L].astype(np.float32) - acts[:, 2, L].astype(np.float32)).mean(0)  # h_VT - h_T
        out[f"v_V_L{L}"] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {n_layers} layer vectors (||v||=1) -> {args.out}")
    print(f"sample norms: " + " ".join(f"L{L}={np.linalg.norm(out[f'v_V_L{L}']):.2f}" for L in (13, 20, 28, 31)))


if __name__ == "__main__":
    main()
