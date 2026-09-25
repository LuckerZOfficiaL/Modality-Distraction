"""Step 14 (Qwen, L20): standard SAE feature interpretation.

For all D_M rows (train_pass + val_pass + test), encode through the
trained Top-K SAE, count how often each latent appears in the per-row
top-K, pick the 3 most-frequent latents, and for each list the 5 D_M
samples with the largest activation magnitude on that latent.

Output: data/interpret/qwen/layer<L>/top_features_inputs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from moground.sae import TopKSAE, SAEConfig


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
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--variant", default="cc3m_1M",
                    help="SAE pretraining variant (subdir under data/sae/qwen/)")
    ap.add_argument("--n-features", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--target-fire-rate", type=float, default=None,
                    help="if set, pick alive features whose fire_rate is closest to this value")
    ap.add_argument("--mode", choices=["freq", "diff"], default="freq",
                    help="freq: top-K firing freq (or near target if --target-fire-rate). "
                         "diff: rank by fr_v - fr_t; output 3 V-preferring + 3 T-preferring features.")
    ap.add_argument("--diff-min-count", type=int, default=5,
                    help="(diff mode) require fire_v + fire_t >= this (drops noise)")
    ap.add_argument("--diff-max-rate", type=float, default=0.9,
                    help="(diff mode) drop features with overall fire_rate above this (always-on)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", nargs="+", default=["train_pass"],
                    help="which split labels in index.jsonl to use. "
                         "Default: train_pass (no val/test mixing — clean policy).")
    ap.add_argument("--act-dir", default=None,
                    help="override D_M activations dir (default: data_root/activations/qwen/dm)")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: data_root/interpret/qwen/<variant>/layer<L>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    ckpt_path = data_root / "sae" / "qwen" / args.variant / f"layer{args.layer}" / "sae.pt"
    act_dir = Path(args.act_dir) if args.act_dir else data_root / "activations" / "qwen" / "dm"
    act_path = act_dir / "activations.npy"
    idx_path = act_dir / "index.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "interpret" / "qwen" / args.variant / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
    sae.load_state_dict(ck["state_dict"])
    sae.to(args.device).to(torch.float32).eval()
    K = int(ck["k"])
    print(f"loaded SAE  d_model={ck['d_model']}  d_sae={ck['d_sae']}  k={K}  layer={ck['layer']}")

    acts = np.load(act_path)
    full_index = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
    assert len(full_index) == acts.shape[0]
    keep = [i for i, r in enumerate(full_index) if r["split"] in args.splits]
    index = [full_index[i] for i in keep]
    acts = acts[keep]
    print(f"D_M: {len(index)} rows from splits={args.splits} (of {len(full_index)} total)")

    x = torch.from_numpy(acts[:, args.layer, :]).to(args.device).float()
    with torch.no_grad():
        out = sae(x)
        z = out["z"].cpu().numpy()  # (N, d_sae) sparse, zeros outside top-K
    print(f"encoded -> z shape {z.shape}")

    # per-row top-K indices (z is exactly sparse top-K)
    fired_mask = z > 0
    freq = fired_mask.sum(axis=0)  # (d_sae,)
    total_mag = z.sum(axis=0)
    n_rows = z.shape[0]
    labels = np.array([0 if r["label"] == "vision" else 1 for r in index])
    n_v = int((labels == 0).sum()); n_t = int((labels == 1).sum())
    freq_v = (z[labels == 0] > 0).sum(axis=0)
    freq_t = (z[labels == 1] > 0).sum(axis=0)
    fr_v = freq_v / max(n_v, 1)
    fr_t = freq_t / max(n_t, 1)

    feat_groups: list[tuple[str, list[int]]] = []  # (group_name, [feature_ids])
    if args.mode == "diff":
        diff = fr_v - fr_t
        overall_rate = freq / n_rows
        mask = ((freq_v + freq_t) >= args.diff_min_count) & (overall_rate <= args.diff_max_rate)
        diff_masked = np.where(mask, diff, np.nan)
        v_pref = np.argsort(-np.where(np.isnan(diff_masked), -np.inf, diff_masked))[: args.n_features]
        t_pref = np.argsort(np.where(np.isnan(diff_masked), np.inf, diff_masked))[: args.n_features]
        feat_groups = [("vision_preferring", v_pref.tolist()),
                       ("text_preferring", t_pref.tolist())]
        selection_desc = (f"diff mode: top {args.n_features} V-preferring + top {args.n_features} "
                          f"T-preferring by fr_v - fr_t, with fire_count>={args.diff_min_count} "
                          f"and overall_rate<={args.diff_max_rate}")
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
        fire_rate = freq / n_rows
        alive = freq > 0
        dist = np.where(alive, np.abs(fire_rate - target), np.inf)
        order = np.lexsort((-total_mag, dist))
        top_feats = order[: args.n_features].tolist()
        selection_desc = f"alive features with fire_rate closest to {target}"
        feat_groups = [("near_fire_rate", top_feats)]
        print(f"features near fire_rate={target}: " +
              ", ".join(f"k={k} fr={fire_rate[k]:.3f}" for k in top_feats))

    dm_meta = load_dm_meta(dm_dir / "dm_all.jsonl")

    report = {
        "sae": "qwen2.5-vl-3b-topk-cc3m",
        "layer": args.layer,
        "k": K,
        "d_sae": int(z.shape[1]),
        "n_rows": int(z.shape[0]),
        "splits_used": sorted({r["split"] for r in index}),
        "selection": selection_desc,
        "features": [],
    }

    for group_name, top_feats in feat_groups:
        for fk in top_feats:
            col = z[:, fk]
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
                "fire_rate": float(freq[fk] / z.shape[0]),
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
