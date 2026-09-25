"""Step 12: contrastive-direction steering.

Steering vector v_L for layer L is computed from D_M train-pass
activations only:

    v_L = mean(h_L | label=vision) - mean(h_L | label=text)
    v_L <- v_L / ‖v_L‖

Sign convention: alpha > 0 pushes vision, alpha < 0 pushes text.

Eval pool selectable via --pool {test, unseen, fails, all}; see
moground.steering for definitions. All 422 splits/* items are
oracle-passed by construction; oracle-failed items never reach disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from moground.models import load_qwen_vl
from moground.steering import (
    fix_negative_args,
    POOL_CHOICES, AdditiveHook, baseline_rows_needed, compute_summary,
    get_letter_token_ids, load_distraction_pool, load_oracle_passed,
    load_resume_state, print_summary, run_baseline_phase, run_hooked_phase,
    select_pool,
)


def compute_contrastive_vectors(act_dir: Path, layers, device) -> dict[int, torch.Tensor]:
    acts = np.load(act_dir / "activations.npy")
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    assert acts.shape[0] == len(index)
    train_mask = np.array([r["split"] == "train_pass" for r in index])
    vision_mask = np.array([r["label"] == "vision" for r in index]) & train_mask
    text_mask = np.array([r["label"] == "text" for r in index]) & train_mask
    print(f"contrastive source: {vision_mask.sum()} vision-train-pass, "
          f"{text_mask.sum()} text-train-pass")
    out = {}
    for L in layers:
        mu_v = acts[vision_mask, L].astype(np.float32).mean(axis=0)
        mu_t = acts[text_mask, L].astype(np.float32).mean(axis=0)
        v = mu_v - mu_t
        n = float(np.linalg.norm(v))
        print(f"  L{L}: ‖mean_vision - mean_text‖ = {n:.3f}")
        out[L] = torch.from_numpy(v / n).to(device).to(torch.float32)
    return out


def main():
    fix_negative_args()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layers", default="13,20,23,28")
    ap.add_argument("--alphas", default="-150,-100,-50,-20,-5,-1,1,5,20,50,100,150")
    ap.add_argument("--pool", choices=POOL_CHOICES, default="unseen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--act-dir", default=None,
                    help="override activations dir (default: data_root/activations/qwen/dm)")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: data_root/steering/qwen/contrastive_<pool>[_bcast])")
    ap.add_argument("--broadcast", action="store_true",
                    help="apply steering to all token positions, not just the last")
    ap.add_argument("--norm-match", action="store_true",
                    help="treat --alphas as fractions k of ||h_last||; injected vector "
                         "has norm k * ||h_last||. Use a smaller alpha grid (~-2..2)")
    ap.add_argument("--random-direction", action="store_true",
                    help="null control: replace mu_v - mu_t with a random unit vector "
                         "at each layer (deterministic by --random-seed).")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--distraction-pool", default=None,
                    help="optional path to data/dm/distraction_pool.jsonl (from "
                         "04d_extract_distraction_pool.py); appended to the eval pool.")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    act_dir = Path(args.act_dir) if args.act_dir else data_root / "activations" / "qwen" / "dm"
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
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "steering" / "qwen" / f"contrastive_{args.pool}{bcast_tag}{nm_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state(out_path)
    if done:
        print(f"resume: {len(done)} records present ({len(baseline_preds)} baselines)")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    if args.random_direction:
        # use act_dir just to find d_model from one activation row
        acts0 = np.load(act_dir / "activations.npy", mmap_mode="r")
        d = acts0.shape[-1]
        del acts0
        rng = np.random.default_rng(args.random_seed)
        vectors = {}
        for L in layers:
            v = rng.standard_normal(d).astype(np.float32)
            v /= float(np.linalg.norm(v))
            vectors[L] = torch.from_numpy(v).to(args.device).to(torch.float32)
            print(f"  L{L}: random unit vector  d={d}  seed={args.random_seed}")
    else:
        vectors = compute_contrastive_vectors(act_dir, layers, args.device)
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
        done=done, out_f=out_f, letter_token_ids=letter_ids, desc="contrastive",
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
