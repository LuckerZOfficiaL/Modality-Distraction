"""Step 13: SAE-feature steering.

At each layer L (where an SAE checkpoint exists), encode h at the last
input-token position with the SAE, then multiplicatively scale modality
features in z-space:

    z'[vision_top]  = z[vision_top] * (1 + alpha)
    z'[text_top]    = z[text_top]   * (1 - alpha)
    delta           = decode(z') - decode(z)
    h_at_last       += delta

Sign convention (uniform with steps 11, 12):

    alpha > 0  pushes vision  (amplify vision feats, dampen text feats)
    alpha < 0  pushes text

alpha is multiplicative here. Default sweep covers a wide range
(±{0.5, 1, 3, 10, 30, 100}) to probe whether the previous narrow
range simply produced sub-flip deltas.

Top-N modality features come from data/features/layerLL/top_features.json
ranked by rho (vision-mean activation minus text-mean), so they match
the analysis already in pilot.tex.

Eval pool selectable via --pool {test, unseen, fails, all}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from moground.models import load_qwen_vl
from moground.sae import TopKSAE, SAEConfig
from moground.steering import (
    fix_negative_args,
    POOL_CHOICES, SAEHook, baseline_rows_needed, compute_summary,
    get_letter_token_ids, load_distraction_pool, load_oracle_passed,
    load_resume_state, print_summary, run_baseline_phase, run_hooked_phase,
    select_pool,
)


def load_sae(data_root: Path, variant: str, layer: int, device: str) -> TopKSAE:
    ckpt = torch.load(data_root / f"sae/qwen/{variant}/layer{layer}/sae.pt",
                      map_location=device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ckpt["d_model"], d_sae=ckpt["d_sae"], k=ckpt["k"]))
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device).to(torch.float32).eval()
    return sae


def load_top_features(features_dir: Path, layer: int, top_n: int) -> tuple[list[int], list[int]]:
    feats = json.loads((features_dir / f"layer{layer}" / "top_features.json").read_text())
    return ([r["k"] for r in feats["top_vision"][:top_n]],
            [r["k"] for r in feats["top_text"][:top_n]])


def main():
    fix_negative_args()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--variant", default="cc3m_1M",
                    help="SAE pretraining variant (subdir under data/sae/qwen/)")
    ap.add_argument("--layers", default="13,20,28")
    ap.add_argument("--alphas", default="-100,-30,-10,-3,-1,-0.5,0.5,1,3,10,30,100",
                    help="multiplicative; + amplifies vision feats and dampens text feats")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--pool", choices=POOL_CHOICES, default="unseen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: data_root/steering/qwen/<variant>/sae_<pool>[_bcast])")
    ap.add_argument("--features-dir", default=None,
                    help="override feature-selection dir (default: data_root/features/qwen/<variant>)")
    ap.add_argument("--broadcast", action="store_true",
                    help="apply steering to all token positions, not just the last")
    ap.add_argument("--norm-match", action="store_true",
                    help="rescale SAE-decoded delta to have norm |alpha| * ||h_last|| (per row); "
                         "alpha becomes layer-agnostic O(1). Use grid ~ -2..2")
    ap.add_argument("--random-direction", action="store_true",
                    help="null control: replace the top-N V/T features (selected by rho_k) "
                         "with random ALIVE feature indices at each layer. The SAE hook "
                         "mechanism is unchanged; only which atoms are edited is random.")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--latent-additive", action="store_true",
                    help="use additive edit in latent space (z'[idx] = z[idx] + α·target[idx]) "
                         "instead of multiplicative. Targets are read from the per-atom causal "
                         "scores in feature_stats.csv (rho_V_causal / rho_T_causal columns; "
                         "requires a causally-rescored features dir like diversified_v1_causal).")
    ap.add_argument("--distraction-pool", default=None,
                    help="optional path to data/dm/distraction_pool.jsonl (from "
                         "04d_extract_distraction_pool.py); appended to the eval pool.")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    all_rows = load_oracle_passed(data_root, baseline_rows_needed(args.pool), dm_dir=dm_dir)
    print(f"loaded {len(all_rows)} oracle-passed items "
          f"(splits: {sorted({r['__split'] for r in all_rows})})")
    if args.distraction_pool:
        dist_rows = load_distraction_pool(Path(args.distraction_pool))
        from collections import Counter as _C
        kinds = _C(r.get("distraction_kind", "?") for r in dist_rows)
        print(f"loaded {len(dist_rows)} distraction items: {dict(kinds)}")
        all_rows = all_rows + dist_rows

    bcast_tag = "_bcast" if args.broadcast else ""
    nm_tag = "_normmatch" if args.norm_match else ""
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "steering" / "qwen" / args.variant / f"sae_{args.pool}{bcast_tag}{nm_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state(out_path)
    if done:
        print(f"resume: {len(done)} records present ({len(baseline_preds)} baselines)")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    features_dir = Path(args.features_dir) if args.features_dir else data_root / "features" / "qwen" / args.variant
    saes: dict[int, TopKSAE] = {}
    feat_idx: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    feat_targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    feat_log: dict[str, list[int]] = {}
    rng = np.random.default_rng(args.random_seed)
    for L in layers:
        saes[L] = load_sae(data_root, args.variant, L, args.device)
        import csv
        rows_csv = list(csv.DictReader(open(features_dir / f"layer{L}" / "feature_stats.csv")))
        rho_by_k = {int(r["k"]): r for r in rows_csv}
        if args.random_direction:
            alive = [k for k, r in rho_by_k.items() if r.get("alive") == "True"]
            chosen = rng.choice(alive, size=2*args.top_n, replace=False).tolist()
            v, t = chosen[:args.top_n], chosen[args.top_n:]
            print(f"  L{L}: random {len(v)}/{len(t)} alive feats (of {len(alive)}, seed={args.random_seed})")
        else:
            v, t = load_top_features(features_dir, L, args.top_n)
            print(f"  L{L}: top {len(v)} vision feats, top {len(t)} text feats")
        feat_idx[L] = (torch.tensor(v, dtype=torch.long, device=args.device),
                       torch.tensor(t, dtype=torch.long, device=args.device))
        # latent-additive targets: per-atom causal magnitudes
        if args.latent_additive:
            # look for rho_V_causal / rho_T_causal columns (causal feature_stats);
            # fall back to "rho" if discriminative.
            def get_target(k_list, side):
                col = f"rho_{side}_causal" if f"rho_{side}_causal" in rows_csv[0] else "rho"
                vals = [float(rho_by_k[k][col]) for k in k_list]
                return torch.tensor(vals, dtype=torch.float32, device=args.device)
            tV = get_target(v, "V")
            tT = get_target(t, "T")
            feat_targets[L] = (tV, tT)
            print(f"    targets: ||tV||={tV.norm().item():.3f}  ||tT||={tT.norm().item():.3f}")
        feat_log[f"L{L}_vision"] = v
        feat_log[f"L{L}_text"] = t
    json.dump(feat_log, open(out_dir / "feature_indices.json", "w"), indent=2)

    out_f = out_path.open("a")
    run_baseline_phase(model=model, processor=processor, rows=all_rows, done=done,
                       out_f=out_f, baseline_preds=baseline_preds, letter_token_ids=letter_ids)

    pool = select_pool(all_rows, baseline_preds, args.pool)
    n_v = sum(1 for r in pool if r["label"] == "vision")
    n_t = sum(1 for r in pool if r["label"] == "text")
    n_test = sum(1 for r in pool if r["__split"] == "test")
    print(f"steering pool ({args.pool}): {len(pool)} items "
          f"(test={n_test}, train+val={len(pool)-n_test}; {n_v} vision, {n_t} text)")

    def hook_factory(L, a):
        v_idx, t_idx = feat_idx[L]
        kwargs = dict(broadcast=args.broadcast, norm_match=args.norm_match)
        if args.latent_additive:
            tV, tT = feat_targets[L]
            kwargs.update(latent_additive=True, v_target=tV, t_target=tT)
        return SAEHook(saes[L], v_idx, t_idx, a, **kwargs)

    delta_stats: dict[tuple[int, float], dict] = {}
    def post_call(L, a, hook):
        s = delta_stats.setdefault((L, a), {"norms": [], "zero": 0, "n": 0})
        s["norms"].extend(hook.delta_norms)
        s["zero"] += hook.zero_count
        s["n"] += len(hook.delta_norms)

    run_hooked_phase(
        model=model, processor=processor, pool=pool, layers=layers, alphas=alphas,
        done=done, out_f=out_f, letter_token_ids=letter_ids, desc="sae",
        hook_factory=hook_factory, post_call=post_call,
    )
    out_f.close()

    delta_log = {}
    print("\ndelta stats (per L, alpha):")
    for (L, a), s in sorted(delta_stats.items()):
        mean_n = float(np.mean(s["norms"])) if s["norms"] else 0.0
        delta_log[f"L{L}|alpha={a}"] = {
            "n": s["n"], "mean_delta_norm": mean_n, "zero_count": s["zero"],
        }
        print(f"  L{L} alpha={a:+g} (n={s['n']:3d}): "
              f"mean‖Δ‖={mean_n:.4f}, zero={s['zero']}")
    json.dump(delta_log, open(out_dir / "delta_stats.json", "w"), indent=2)

    summary = compute_summary(out_path, pool, layers, alphas)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir/'summary.json'}")
    print_summary(summary, layers, alphas)


if __name__ == "__main__":
    main()
