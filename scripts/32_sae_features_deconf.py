"""Step 32: re-score SAE features on the DE-CONFOUNDED data (v2 matched set).

All prior SAE feature selection used confounded D_M, so the top atoms tracked
question-TYPE. Here we re-score on dm_matched_deconf_v2_all (object-matched +
symmetric decoy; BoW question+caption ~= chance), so a V/T-separating atom must be
non-surface. Two scores per atom, per SAE layer:

  discriminative  rho_disc[k] = mean_V z_k(h_VT) - mean_T z_k(h_VT)
                  (also firing-rate diff d_fr = fr_v - fr_t)  -> candidate GATE atoms
  causal (V)      rho_cV[k]   = E_{V items}[ z_k(h_V) - z_k(h_VT) ]
                  (atoms that fire MORE when the caption is removed) -> image-routing EDIT atoms

Reports the central overlaps: disc-vs-causal (should differ), and
new(de-confounded)-vs-old(confounded) selection (should differ if de-confounding
removed question-type atoms). Writes feature_stats.csv + top_features.json.

    python scripts/32_sae_features_deconf.py            # all survivors
    python scripts/32_sae_features_deconf.py --qwen-iso # behaviorally modality-isolated subset
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from moground.sae import TopKSAE, SAEConfig


def _old_top_vision(path: Path) -> set:
    if not path.exists():
        return set()
    d = json.loads(path.read_text())
    tv = d.get("top_vision", [])
    return {int(x["k"]) if isinstance(x, dict) else int(x) for x in tv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--act-dir", default="data/activations/qwen/dm_matched_deconf_v2_all")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layers", default="13,20,28,31")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--qwen-iso", action="store_true", help="restrict to Qwen modality-isolated rows")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-variant", default="diversified_v1_deconf")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy")                       # (N,3,n_layers,d) [VT,V,T]
    idx = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    ll = np.load(act_dir / "letter_logits.npy")                      # (N,3,4)
    y = np.array([0 if r["label"] == "vision" else 1 for r in idx])
    ci = np.array([r["correct_index"] for r in idx])

    keep = np.ones(len(idx), bool)
    if args.qwen_iso:
        qa = ll.argmax(-1); vt, v, t = qa[:, 0] == ci, qa[:, 1] == ci, qa[:, 2] == ci
        keep = ((y == 0) & vt & v & ~t) | ((y == 1) & vt & ~v & t)
    is_V = (y == 0) & keep
    is_T = (y == 1) & keep
    print(f"acts {acts.shape} | scoring on {'Qwen-iso' if args.qwen_iso else 'all survivors'}: "
          f"V={int(is_V.sum())} T={int(is_T.sum())}")

    for L in [int(x) for x in args.layers.split(",")]:
        ckpt = data_root / "sae" / "qwen" / args.variant / f"layer{L}" / "sae.pt"
        if not ckpt.exists():
            print(f"\nL{L}: SAE missing at {ckpt} — skip"); continue
        ck = torch.load(ckpt, map_location=args.device, weights_only=False)
        sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
        sae.load_state_dict(ck["state_dict"]); sae.to(args.device).to(torch.float32).eval()

        def enc(mask, cond):
            X = torch.from_numpy(acts[mask, cond, L].astype(np.float32)).to(args.device)
            with torch.no_grad():
                return sae(X)["z"].cpu().numpy()

        zVT_V, zV_V = enc(is_V, 0), enc(is_V, 1)
        zVT_T = enc(is_T, 0)
        rho_disc = zVT_V.mean(0) - zVT_T.mean(0)
        fr_v, fr_t = (zVT_V > 0).mean(0), (zVT_T > 0).mean(0)
        d_fr = fr_v - fr_t
        rho_cV = (zV_V - zVT_V).mean(0)
        alive = np.concatenate([zVT_V, zVT_T], 0).var(0) > 1e-8

        order = lambda s: [int(k) for k in np.argsort(-s) if alive[k]]
        topN = args.top_n
        top_disc = order(rho_disc)[:topN]
        top_dfr = order(d_fr)[:topN]
        top_caus = order(rho_cV)[:topN]
        top_text = [int(k) for k in np.argsort(rho_disc) if alive[k]][:topN]   # most T-preferring

        old_disc = _old_top_vision(data_root / "features" / "qwen" / args.variant / f"layer{L}" / "top_features.json")
        old_caus = _old_top_vision(data_root / "features" / "qwen" / f"{args.variant}_causal" / f"layer{L}" / "top_features.json")

        def ov(a, b): return len(set(a) & set(b))
        print(f"\n=== L{L} (alive={int(alive.sum())}) top-{topN} V-pref atoms ===")
        print(f"  disc-vs-causal overlap (deconf):           {ov(top_disc, top_caus)}/{topN}")
        print(f"  deconf-disc  vs old-confounded-disc:       {ov(top_disc, old_disc)}/{topN}  (old set n={len(old_disc)})")
        print(f"  deconf-causal vs old-confounded-causal:    {ov(top_caus, old_caus)}/{topN}  (old set n={len(old_caus)})")
        print(f"  max rho_disc={rho_disc[top_disc[0]]:+.3f} (k={top_disc[0]})  "
              f"max d_fr={d_fr[top_dfr[0]]:+.3f} (k={top_dfr[0]})  "
              f"max rho_cV={rho_cV[top_caus[0]]:+.3f} (k={top_caus[0]})")

        out = data_root / "features" / "qwen" / args.out_variant / f"layer{L}"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "feature_stats.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["k", "rho_disc", "d_fr", "fr_v", "fr_t", "rho_V_causal", "alive"])
            for k in range(sae.cfg.d_sae):
                w.writerow([k, f"{rho_disc[k]:.5f}", f"{d_fr[k]:.5f}", f"{fr_v[k]:.4f}",
                            f"{fr_t[k]:.4f}", f"{rho_cV[k]:.5f}", int(alive[k])])
        (out / "top_features.json").write_text(json.dumps({
            "layer": L, "qwen_iso": args.qwen_iso, "n_V": int(is_V.sum()), "n_T": int(is_T.sum()),
            "alive": int(alive.sum()),
            # harness-compatible (scripts/19 _load_top_feature_idx): discriminative V/T atoms
            "top_vision": [{"k": k} for k in top_disc], "top_text": [{"k": k} for k in top_text],
            "top_vision_disc": top_disc, "top_vision_dfr": top_dfr,
            "top_vision_causal": top_caus}, indent=2))
    print(f"\nwrote -> data/features/qwen/{args.out_variant}/layer*/")


if __name__ == "__main__":
    main()
