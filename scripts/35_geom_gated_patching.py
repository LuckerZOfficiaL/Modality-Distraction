"""Step 35: geometric-gated patching — direction (aligned vs orthogonal) + per-pool net.

Per-item selective patching (h_VT -> h_V on gated VISION items; V/T router = label,
geometry selects which V to patch). Two questions:
  1. DIRECTION: fire on most cosine-SIMILAR (aligned) or most ORTHOGONAL (large ||delta||)?
  2. WHICH POOL benefits? Report net pool accuracy on EVERY split, not just val.

Gate threshold + direction selected ONCE on --dev-split; reported on all pools.

    python scripts/35_geom_gated_patching.py --act-dir data/activations/qwen/dm_multidomain_v1_counterfactual --layer 31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--dev-split", default="val_pass", help="split used to pick the gate threshold")
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy")
    idx = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    ll = np.load(act_dir / "letter_logits.npy")
    ci = np.array([r["correct_index"] for r in idx])
    is_V = np.array([r["label"] == "vision" for r in idx])
    split = np.array([r.get("split", "?") for r in idx])
    pools = [s for s in ["train_pass", "val_pass", "val", "train"] if (split == s).any()]

    vt_pred, v_pred = ll[:, 0].argmax(1), ll[:, 1].argmax(1)
    L = args.layer
    hVT, hV = acts[:, 0, L].astype(np.float32), acts[:, 1, L].astype(np.float32)
    dnorm = np.linalg.norm(hV - hVT, axis=1)
    cos_vvt = (hV * hVT).sum(1) / (np.linalg.norm(hV, axis=1) * np.linalg.norm(hVT, axis=1) + 1e-8)

    def net(fire, mask):  # net pool acc change vs VT baseline on `mask`
        pred = np.where(fire, v_pred, vt_pred)
        return (pred[mask] == ci[mask]).mean() - (vt_pred[mask] == ci[mask]).mean()

    dev = split == args.dev_split
    print(f"L{L} | dev-split={args.dev_split} (n={int(dev.sum())}) | pools: "
          + "  ".join(f"{s}={int((split==s).sum())}" for s in pools))
    print(f"baseline VT acc per pool: " + "  ".join(f"{s} {(vt_pred[split==s]==ci[split==s]).mean():.3f}" for s in pools))

    # reference: oracle-label (patch all V), threshold-free
    print(f"\n{'gate':28}" + "".join(f"{s:>12}" for s in pools) + f"{'ALL':>12}")
    fire = is_V
    print(f"{'patch-all-V (oracle label)':28}" + "".join(f"{net(fire, split==s):>+12.3f}" for s in pools)
          + f"{net(fire, np.ones(len(ci),bool)):>+12.3f}")

    for name, score in [("cos(hV,hVT)", cos_vvt), ("||delta||", dnorm)]:
        for direction in ("aligned", "orthogonal"):
            hi = (name == "cos(hV,hVT)") == (direction == "aligned")
            qs = np.quantile(score[is_V], np.linspace(0.05, 0.95, 19))
            thr = max(qs, key=lambda t: net(is_V & (score >= t if hi else score <= t), dev))
            fire = is_V & (score >= thr if hi else score <= thr)
            row = "".join(f"{net(fire, split==s):>+12.3f}" for s in pools)
            print(f"{name+' '+direction:28}" + row + f"{net(fire, np.ones(len(ci),bool)):>+12.3f}")


if __name__ == "__main__":
    main()
