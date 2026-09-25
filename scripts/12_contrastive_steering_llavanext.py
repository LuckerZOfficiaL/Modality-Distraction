"""LLaVA-NeXT contrastive-direction steering (mean(vision) - mean(text))."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from moground.models_llavanext import load_llavanext
from moground.steering import (
    POOL_CHOICES, AdditiveHook, baseline_rows_needed, compute_summary,
    get_letter_token_ids, load_resume_state, print_summary,
    run_baseline_phase, run_hooked_phase, select_pool,
)
from moground.steering_llavanext import score_row, load_oracle_passed_llavanext


def compute_contrastive_vectors(data_root: Path, layers, device) -> dict[int, torch.Tensor]:
    acts = np.load(data_root / "activations_llavanext/dm/activations.npy")
    index = [json.loads(l) for l in (data_root / "activations_llavanext/dm/index.jsonl").read_text().splitlines() if l.strip()]
    train_mask = np.array([r["split"] == "train_pass" for r in index])
    vision_mask = np.array([r["label"] == "vision" for r in index]) & train_mask
    text_mask = np.array([r["label"] == "text" for r in index]) & train_mask
    print(f"contrastive source: {vision_mask.sum()} vision-train-pass, {text_mask.sum()} text-train-pass")
    out = {}
    for L in layers:
        mu_v = acts[vision_mask, L].astype(np.float32).mean(axis=0)
        mu_t = acts[text_mask, L].astype(np.float32).mean(axis=0)
        v = mu_v - mu_t
        n = float(np.linalg.norm(v))
        print(f"  L{L}: ‖μ_v - μ_t‖ = {n:.3f}")
        out[L] = torch.from_numpy(v / n).to(device).to(torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layers", default="12,20,24")
    ap.add_argument("--alphas", default="-150,-100,-50,-20,-5,-1,1,5,20,50,100,150")
    ap.add_argument("--pool", choices=POOL_CHOICES, default="unseen")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    all_rows = load_oracle_passed_llavanext(data_root, baseline_rows_needed(args.pool))
    print(f"loaded {len(all_rows)} items")

    out_dir = data_root / "steering_llavanext" / f"contrastive_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state(out_path)

    model, processor = load_llavanext(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    vectors = compute_contrastive_vectors(data_root, layers, args.device)
    np.savez(out_dir / "steering_vectors.npz",
             **{f"layer{L:02d}": vectors[L].cpu().numpy() for L in layers})

    out_f = out_path.open("a")
    run_baseline_phase(model=model, processor=processor, rows=all_rows, done=done,
                       out_f=out_f, baseline_preds=baseline_preds,
                       letter_token_ids=letter_ids, score_fn=score_row)

    pool = select_pool(all_rows, baseline_preds, args.pool)
    n_v = sum(1 for r in pool if r["label"] == "vision")
    n_t = sum(1 for r in pool if r["label"] == "text")
    n_test = sum(1 for r in pool if r["__split"] == "test")
    print(f"pool ({args.pool}): {len(pool)} (test={n_test}, train+val={len(pool)-n_test}; {n_v}v {n_t}t)")

    run_hooked_phase(
        model=model, processor=processor, pool=pool, layers=layers, alphas=alphas,
        done=done, out_f=out_f, letter_token_ids=letter_ids, desc="contrastive",
        hook_factory=lambda L, a: AdditiveHook(vectors[L], a), score_fn=score_row,
    )
    out_f.close()

    summary = compute_summary(out_path, pool, layers, alphas)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"wrote {out_dir/'summary.json'}")
    print_summary(summary, layers, alphas)


if __name__ == "__main__":
    main()
