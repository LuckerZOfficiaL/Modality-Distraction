"""Step 11: probe-direction steering.

Steering vector v_L is the L2 logistic-probe coefficient at layer L
(data/probes/probe_layerLL.npz), sign-flipped (probe class 1 = text)
and unit-normalized so that the uniform sign convention holds:

    alpha > 0  pushes vision
    alpha < 0  pushes text

Eval pool selectable via --pool {test, unseen, fails, all}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_steering.models import load_qwen_vl
from sae_steering.steering import (
    POOL_CHOICES, AdditiveHook, baseline_rows_needed, compute_summary,
    fix_negative_args, get_letter_token_ids, load_distraction_pool,
    load_oracle_passed, load_resume_state, print_summary,
    run_baseline_phase, run_hooked_phase, select_pool,
)


def compute_probe_vectors(probes_dir: Path, layers, device) -> dict[int, torch.Tensor]:
    out = {}
    for L in layers:
        probe = np.load(probes_dir / f"probe_layer{L:02d}.npz")
        classes = probe["classes"].tolist()
        assert classes == [0, 1], f"unexpected classes for L{L}: {classes}"
        w = probe["coef"][0].astype(np.float32)
        n = float(np.linalg.norm(w))
        print(f"  L{L}: ‖probe_coef‖ = {n:.3f}")
        out[L] = torch.from_numpy(-w / n).to(device).to(torch.float32)
    return out


def compute_random_vectors(probes_dir: Path, layers, device, seed: int) -> dict[int, torch.Tensor]:
    """Random unit-norm vectors in R^d_model at each layer, with the same dim
    as the probe coefficient (so the hook is interchangeable). Null control."""
    rng = np.random.default_rng(seed)
    out = {}
    for L in layers:
        probe = np.load(probes_dir / f"probe_layer{L:02d}.npz")
        d = probe["coef"][0].shape[0]
        v = rng.standard_normal(d).astype(np.float32)
        v /= float(np.linalg.norm(v))
        print(f"  L{L}: random unit vector  d={d}  seed={seed}")
        out[L] = torch.from_numpy(v).to(device).to(torch.float32)
    return out


def main():
    fix_negative_args()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layers", default="13,20,23,28")
    ap.add_argument("--alphas", default="-150,-100,-50,-20,-5,-1,1,5,20,50,100,150")
    ap.add_argument("--pool", choices=POOL_CHOICES, default="unseen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probes-dir", default=None,
                    help="override probe coef dir (default: data_root/probes/qwen)")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: data_root/steering/qwen/probe_<pool>[_bcast])")
    ap.add_argument("--broadcast", action="store_true",
                    help="apply steering to all token positions, not just the last")
    ap.add_argument("--norm-match", action="store_true",
                    help="treat --alphas as fractions k of ||h_last||; injected vector "
                         "has norm k * ||h_last||. Use a smaller alpha grid (~-2..2)")
    ap.add_argument("--random-direction", action="store_true",
                    help="null control: replace the probe direction with a random unit "
                         "vector at each layer (deterministic by --random-seed). Hook "
                         "behavior and norm are unchanged; only the direction is random.")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="seed for --random-direction")
    ap.add_argument("--distraction-pool", default=None,
                    help="optional path to data/dm/distraction_pool.jsonl (from "
                         "04d_extract_distraction_pool.py); rows are appended to the "
                         "eval pool with __split='test' and flow through baseline + "
                         "steering identically to oracle survivors.")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    probes_dir = Path(args.probes_dir) if args.probes_dir else data_root / "probes" / "qwen"
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
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "steering" / "qwen" / f"probe_{args.pool}{bcast_tag}{nm_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state(out_path)
    if done:
        print(f"resume: {len(done)} records present ({len(baseline_preds)} baselines)")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    if args.random_direction:
        vectors = compute_random_vectors(probes_dir, layers, args.device, args.random_seed)
    else:
        vectors = compute_probe_vectors(probes_dir, layers, args.device)
    np.savez(out_dir / "steering_vectors.npz",
             **{f"layer{L:02d}": vectors[L].cpu().numpy() for L in layers})

    out_f = out_path.open("a")
    run_baseline_phase(model=model, processor=processor, rows=all_rows, done=done,
                       out_f=out_f, baseline_preds=baseline_preds, letter_token_ids=letter_ids)

    pool = select_pool(all_rows, baseline_preds, args.pool)
    n_v = sum(1 for r in pool if r["label"] == "vision")
    n_t = sum(1 for r in pool if r["label"] == "text")
    n_test = sum(1 for r in pool if r["__split"] == "test")
    print(f"steering pool ({args.pool}): {len(pool)} items "
          f"(test={n_test}, train+val={len(pool)-n_test}; {n_v} vision, {n_t} text)")

    run_hooked_phase(
        model=model, processor=processor, pool=pool, layers=layers, alphas=alphas,
        done=done, out_f=out_f, letter_token_ids=letter_ids, desc="probe",
        hook_factory=lambda L, a: AdditiveHook(vectors[L], a, broadcast=args.broadcast,
                                               norm_match=args.norm_match),
    )
    out_f.close()

    summary = compute_summary(out_path, pool, layers, alphas)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir/'summary.json'}")
    print_summary(summary, layers, alphas)


if __name__ == "__main__":
    main()
