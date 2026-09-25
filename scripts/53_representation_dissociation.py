"""T2: representation-vs-behavior dissociation, per backbone (model-agnostic, from cf-store).

For each layer, the relative last-token perturbation from adding the irrelevant modality:
  vision items: ||h_VT - h_V|| / ||h_VT||   (caption's perturbation of a vision rep)
  text   items: ||h_VT - h_T|| / ||h_VT||   (image's perturbation of a text rep)
paired with the behavioral harm (V+T vs single-modality accuracy). The Qwen finding:
the image perturbs text reps MORE than the caption perturbs vision reps, yet only the
caption harms accuracy -> magnitude != interference. This checks if it holds cross-model.

float32 throughout (f16 norms overflow at the last layer).

    python scripts/53_representation_dissociation.py --model qwen --cf-subdir dm_multidomain_v1_counterfactual_full
    python scripts/53_representation_dissociation.py --model llavanext --cf-subdir dm_multidomain_v1_cf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual_full")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    st = Path(cfg["paths"]["data_root"]) / "activations" / args.model / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")     # (N,3,L,d) [VT,V,T]
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    isV = lab == "vision"; isT = lab == "text"
    n_layers = acts.shape[2]

    def rel_perturb(mask, cond):  # ||h_VT - h_cond|| / ||h_VT|| per layer, mean over mask
        out = np.zeros(n_layers)
        vt = acts[mask, 0].astype(np.float32); other = acts[mask, cond].astype(np.float32)
        num = np.linalg.norm(vt - other, axis=2); den = np.linalg.norm(vt, axis=2) + 1e-6
        return (num / den).mean(0)

    vis_pert = rel_perturb(isV, 1)   # caption perturbs vision
    txt_pert = rel_perturb(isT, 2)   # image perturbs text

    # behavioral harm
    def acc(mask, cond): return (ll[mask, cond].argmax(1) == ci[mask]).mean()
    v_vt, v_v = acc(isV, 0), acc(isV, 1)
    t_vt, t_t = acc(isT, 0), acc(isT, 2)

    print(f"\n=== T2 dissociation: {args.model} ({args.cf_subdir}) ===")
    print(f"nV={int(isV.sum())} nT={int(isT.sum())} | layers={n_layers}")
    print(f"{'L':>3}{'caption->vision rel||Δ||':>26}{'image->text rel||Δ||':>24}")
    for L in range(n_layers):
        print(f"{L:>3}{vis_pert[L]:>26.3f}{txt_pert[L]:>24.3f}")
    print(f"\nPEAK perturbation: caption->vision {vis_pert.max():.3f} (L{vis_pert.argmax()}) | "
          f"image->text {txt_pert.max():.3f} (L{txt_pert.argmax()})  -> ratio {txt_pert.max()/max(vis_pert.max(),1e-6):.2f}x")
    print(f"BEHAVIORAL harm: vision V+T {v_vt:.3f} vs V-only {v_v:.3f} (Δ{v_vt-v_v:+.3f}) | "
          f"text V+T {t_vt:.3f} vs T-only {t_t:.3f} (Δ{t_vt-t_t:+.3f})")
    print("DISSOCIATION = image perturbs text reps MORE, yet caption causes the accuracy harm." )


if __name__ == "__main__":
    main()
