"""Step 15 (Qwen): collect top-N max-activating inputs for the SAE steering features.

Features = top 20 V-preferring + top 20 T-preferring from
  data/features/qwen/<variant>/layer<L>/top_features.json
(selected in step 09 by rho_k = mu_v - mu_t on D_M train_pass).

For each of the 40 features, encode D_M train_pass rows through the Qwen
TopKSAE at <layer> and list the N rows with the largest activation.

Output: data/interpret/qwen/<variant>/layer<L>/steering_features/top3.json

Sister scripts for the other VLMs (mirroring step 14): TBD as
  15_interpret_steering_features_llavanext.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_steering.sae import TopKSAE, SAEConfig


def load_dm_meta(dm_all_path: Path) -> dict[str, dict]:
    meta = {}
    for line in dm_all_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta[r["candidate_id"]] = r
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--dm-name", default="multidomain_v1",
                    help="subdir under data/dm/ for the dm_all.jsonl metadata")
    ap.add_argument("--act-name", default="dm_multidomain_v1",
                    help="subdir under data/activations/qwen/ for activations+index")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])

    ckpt = data_root / "sae" / "qwen" / args.variant / f"layer{args.layer}" / "sae.pt"
    act_dir = data_root / "activations" / "qwen" / args.act_name
    feat_path = data_root / "features" / "qwen" / args.variant / f"layer{args.layer}" / "top_features.json"
    dm_meta_path = data_root / "dm" / args.dm_name / "dm_all.jsonl"
    out_dir = data_root / "interpret" / "qwen" / args.variant / f"layer{args.layer}" / "steering_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top3.json"

    ck = torch.load(ckpt, map_location=args.device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
    sae.load_state_dict(ck["state_dict"])
    sae.to(args.device).to(torch.float32).eval()
    print(f"loaded SAE  d_model={ck['d_model']}  d_sae={ck['d_sae']}  k={ck['k']}  layer={ck['layer']}")

    acts = np.load(act_dir / "activations.npy")
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    assert len(index) == acts.shape[0]

    keep = [i for i, r in enumerate(index) if r["split"] == "train_pass"]
    print(f"using {len(keep)} train_pass rows (of {len(index)})")
    sub_index = [index[i] for i in keep]
    x = torch.from_numpy(acts[keep, args.layer, :]).to(args.device).float()

    with torch.no_grad():
        z = sae(x)["z"].cpu().numpy()
    print(f"encoded -> z shape {z.shape}")

    feats = json.loads(feat_path.read_text())
    dm_meta = load_dm_meta(dm_meta_path)

    out = {
        "model": "qwen2.5-vl-3b",
        "sae_variant": args.variant,
        "layer": args.layer,
        "selection": "top 20 V-pref + top 20 T-pref by rho_k = mu_v - mu_t on D_M train_pass (step 09)",
        "n_train_pass": len(keep),
        "n_vision": int(sum(1 for r in sub_index if r["label"] == "vision")),
        "n_text": int(sum(1 for r in sub_index if r["label"] == "text")),
        "groups": [],
    }

    for group_name, key in [("vision_preferring", "top_vision"),
                            ("text_preferring",   "top_text")]:
        group_out = {"group": group_name, "features": []}
        for rec in feats[key]:
            k = int(rec["k"])
            col = z[:, k]
            order = np.argsort(-col)[:args.n_samples]
            samples = []
            for ri in order:
                cid = sub_index[int(ri)]["candidate_id"]
                m = dm_meta.get(cid, {})
                samples.append({
                    "candidate_id": cid,
                    "label": sub_index[int(ri)]["label"],
                    "activation": float(col[int(ri)]),
                    "image_path": m.get("image_path"),
                    "caption": m.get("original_caption"),
                    "question": m.get("question"),
                    "options": m.get("options"),
                    "correct_index": m.get("correct_index"),
                    "source": m.get("source"),
                })
            group_out["features"].append({
                "k": k,
                "rho": float(rec["rho"]),
                "mu_v": float(rec["mu_v"]),
                "mu_t": float(rec["mu_t"]),
                "fire_v": float(rec["fire_v"]),
                "fire_t": float(rec["fire_t"]),
                "auroc": float(rec["auroc"]),
                "top_samples": samples,
            })
        out["groups"].append(group_out)
        print(f"{group_name}: {len(group_out['features'])} features")

    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
