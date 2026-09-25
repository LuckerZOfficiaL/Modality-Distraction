"""Dump top-activating D_M examples for the L28 distraction-susceptibility features,
for auto-interp labeling. Writes a JSON the labeler (Sonnet) reads.

    python scripts/47_dump_distraction_examples.py --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_steering.sae import TopKSAE, SAEConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--feats", default="2788,653,9735,14374,15370,12769,9535")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/diagnostics/sae_distraction/layer28_examples.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    st = dr / "activations" / args.model / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    v_ok = ll[:, 1].argmax(1) == ci; vt_ok = ll[:, 0].argmax(1) == ci
    sel = np.where((lab == "vision") & v_ok)[0]
    distracted = ~vt_ok[sel]
    meta = {}
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); meta[r["candidate_id"]] = r

    ck = torch.load(dr / "sae" / args.model / args.variant / f"layer{args.layer}" / "sae.pt",
                    map_location=args.device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
    sae.load_state_dict(ck["state_dict"]); sae.to(args.device).eval()
    with torch.no_grad():
        z = sae(torch.tensor(acts[sel, 0, args.layer].astype(np.float32),
                             device=args.device))["z"].cpu().numpy()

    feats = [int(x) for x in args.feats.split(",")]
    dump = {}
    for f in feats:
        col = z[:, f]
        top = np.argsort(-col)[:args.topk]
        exs = []
        for j in top:
            cid = idx[sel[j]]["candidate_id"]; m = meta.get(cid, {})
            exs.append({"act": round(float(col[j]), 3), "distracted": bool(distracted[j]),
                        "candidate_id": cid, "source": m.get("source"),
                        "image_path": m.get("image_path"),
                        "question": m.get("question"), "options": m.get("options"),
                        "correct_index": m.get("correct_index"),
                        "caption": m.get("caption_for_filter", "") or ""})
        dump[str(f)] = {"frac_distracted_in_top": round(float(distracted[top].mean()), 2),
                        "examples": exs}
    Path(args.out).write_text(json.dumps(dump, indent=2))
    print(f"dumped {len(feats)} features x top-{args.topk} -> {args.out}")
    for f in feats:
        print(f"  feat {f}: top-{args.topk} frac distracted = {dump[str(f)]['frac_distracted_in_top']}")


if __name__ == "__main__":
    main()
