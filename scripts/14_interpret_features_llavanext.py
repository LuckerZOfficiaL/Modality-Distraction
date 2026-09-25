"""Step 14 (LLaVA-NeXT, L24): standard SAE feature interpretation.

For all D_M rows, encode through the released lmms-lab L24 Top-K SAE,
count how often each latent appears in the per-row top-K, pick the 3
most-frequent latents, list the 5 D_M samples with the largest
activation magnitude per feature.

Output: data/interpret/llavanext/layer24/top_features_inputs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sparsify import Sae


SAE_HUB = "lmms-lab/llama3-llava-next-8b-hf-sae-131k"


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
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n-features", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--target-fire-rate", type=float, default=None,
                    help="if set, pick alive features whose fire_rate is closest to this value")
    ap.add_argument("--mode", choices=["freq", "diff"], default="freq")
    ap.add_argument("--diff-min-count", type=int, default=5)
    ap.add_argument("--diff-max-rate", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])

    act_path = data_root / "activations" / "llavanext" / "dm" / "activations.npy"
    idx_path = data_root / "activations" / "llavanext" / "dm" / "index.jsonl"
    out_dir = data_root / "interpret" / "llavanext" / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = Sae.load_from_hub(SAE_HUB, hookpoint=f"model.layers.{args.layer}").to(args.device).eval()
    K = int(sae.cfg.k)
    d_sae = int(sae.num_latents)
    print(f"loaded SAE  d_in={sae.d_in}  num_latents={d_sae}  k={K}")

    acts = np.load(act_path)
    index = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
    assert len(index) == acts.shape[0]
    print(f"D_M: {len(index)} rows; using all splits {sorted({r['split'] for r in index})}")

    x = torch.from_numpy(acts[:, args.layer, :]).to(args.device).float()
    with torch.no_grad():
        enc = sae.encode(x)
        top_acts = enc.top_acts.cpu().numpy()         # (N, K)
        top_indices = enc.top_indices.cpu().numpy()   # (N, K)
    n = top_acts.shape[0]
    print(f"encoded; top-K acts shape {top_acts.shape}")

    # frequency = how often a latent appears in any row's top-K with act > 0
    # (sparsify's TopK selects on pre-activations then applies ReLU, so top_acts
    # can contain exact zeros — exclude those for parity with Qwen/Mistral)
    fired_mask = top_acts > 0
    fired_indices = top_indices[fired_mask]
    fired_acts = top_acts[fired_mask]
    freq = np.bincount(fired_indices, minlength=d_sae)
    total_mag = np.zeros(d_sae, dtype=np.float64)
    np.add.at(total_mag, fired_indices, fired_acts.astype(np.float64))

    labels = np.array([0 if r["label"] == "vision" else 1 for r in index])
    n_v = int((labels == 0).sum()); n_t = int((labels == 1).sum())
    v_mask = (labels == 0)[:, None] & fired_mask
    t_mask = (labels == 1)[:, None] & fired_mask
    freq_v = np.bincount(top_indices[v_mask], minlength=d_sae)
    freq_t = np.bincount(top_indices[t_mask], minlength=d_sae)
    fr_v = freq_v / max(n_v, 1)
    fr_t = freq_t / max(n_t, 1)

    feat_groups: list[tuple[str, list[int]]] = []
    if args.mode == "diff":
        diff = fr_v - fr_t
        overall_rate = freq / n
        mask = ((freq_v + freq_t) >= args.diff_min_count) & (overall_rate <= args.diff_max_rate)
        diff_masked = np.where(mask, diff, np.nan)
        v_pref = np.argsort(-np.where(np.isnan(diff_masked), -np.inf, diff_masked))[: args.n_features]
        t_pref = np.argsort(np.where(np.isnan(diff_masked), np.inf, diff_masked))[: args.n_features]
        feat_groups = [("vision_preferring", v_pref.tolist()),
                       ("text_preferring", t_pref.tolist())]
        selection_desc = (f"diff mode: top {args.n_features} V-preferring + top {args.n_features} "
                          f"T-preferring by fr_v - fr_t, fire_count>={args.diff_min_count}, "
                          f"overall_rate<={args.diff_max_rate}")
        print(selection_desc)
        for g, feats in feat_groups:
            print(f"  {g}: " + ", ".join(
                f"k={k} fr_v={fr_v[k]:.3f} fr_t={fr_t[k]:.3f} d={fr_v[k]-fr_t[k]:+.3f}" for k in feats))
    elif args.target_fire_rate is None:
        order = np.lexsort((-total_mag, -freq))
        top_feats = order[: args.n_features].tolist()
        selection_desc = "top features by top-K firing frequency across all D_M rows"
        feat_groups = [("top_freq", top_feats)]
        print(f"top-{args.n_features} features by top-K frequency: {top_feats}")
    else:
        target = args.target_fire_rate
        fire_rate = freq / n
        alive = freq > 0
        dist = np.where(alive, np.abs(fire_rate - target), np.inf)
        order = np.lexsort((-total_mag, dist))
        top_feats = order[: args.n_features].tolist()
        selection_desc = f"alive features with fire_rate closest to {target}"
        feat_groups = [("near_fire_rate", top_feats)]
        print(f"features near fire_rate={target}: " +
              ", ".join(f"k={k} fr={fire_rate[k]:.3f}" for k in top_feats))

    all_feats = [fk for _, feats in feat_groups for fk in feats]
    feat_to_col = {fk: i for i, fk in enumerate(all_feats)}
    feat_acts = np.zeros((n, len(all_feats)), dtype=np.float32)
    for ri in range(n):
        for kk, idx_k in enumerate(top_indices[ri]):
            if int(idx_k) in feat_to_col:
                feat_acts[ri, feat_to_col[int(idx_k)]] = top_acts[ri, kk]

    dm_meta = load_dm_meta(data_root / "dm" / "dm_all.jsonl")

    report = {
        "sae": SAE_HUB,
        "layer": args.layer,
        "k": K,
        "d_sae": d_sae,
        "n_rows": int(n),
        "splits_used": sorted({r["split"] for r in index}),
        "selection": selection_desc,
        "features": [],
    }

    for group_name, top_feats in feat_groups:
        for fk in top_feats:
            ci = feat_to_col[int(fk)]
            col = feat_acts[:, ci]
            order_rows = np.argsort(-col)[: args.n_samples]
            samples = []
            for ri in order_rows:
                cid = index[int(ri)]["candidate_id"]
                m = dm_meta.get(cid, {})
                samples.append({
                    "candidate_id": cid,
                    "label": index[int(ri)]["label"],
                    "split": index[int(ri)]["split"],
                    "activation": float(col[int(ri)]),
                    "image_path": m.get("image_path"),
                    "caption": m.get("original_caption"),
                    "question": m.get("question"),
                    "options": m.get("options"),
                    "correct_index": m.get("correct_index"),
                })
            report["features"].append({
                "group": group_name,
                "k": int(fk),
                "fire_count": int(freq[fk]),
                "fire_rate": float(freq[fk] / n),
                "fire_rate_v": float(fr_v[fk]),
                "fire_rate_t": float(fr_t[fk]),
                "diff_v_minus_t": float(fr_v[fk] - fr_t[fk]),
                "total_magnitude": float(total_mag[fk]),
                "top_samples": samples,
            })
            print(f"\n[{group_name}] k={fk}  fr_v={fr_v[fk]:.3f}  fr_t={fr_t[fk]:.3f}  d={fr_v[fk]-fr_t[fk]:+.3f}")
            for s in samples:
                cap = (s["caption"] or "")[:80].replace("\n", " ")
                print(f"  {s['candidate_id']:<24} {s['label']:<6} act={s['activation']:.3f}  {cap}")

    if args.mode == "diff":
        suffix = "_diff"
    elif args.target_fire_rate is not None:
        suffix = f"_fr{int(round(args.target_fire_rate*100))}"
    else:
        suffix = ""
    out_path = out_dir / f"top_features_inputs{suffix}.json"
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
