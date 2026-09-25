"""LLaVA-NeXT probe-direction steering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_steering.models_llavanext import load_llavanext
from sae_steering.steering import (
    POOL_CHOICES, AdditiveHook, baseline_rows_needed, compute_summary,
    get_letter_token_ids, load_resume_state, print_summary,
    run_baseline_phase, run_hooked_phase, select_pool,
)
from sae_steering.steering_llavanext import score_row, load_oracle_passed_llavanext


def compute_probe_vectors(data_root: Path, layers, device) -> dict[int, torch.Tensor]:
    out = {}
    for L in layers:
        probe = np.load(data_root / "probes_llavanext" / f"probe_layer{L:02d}.npz")
        classes = probe["classes"].tolist()
        assert classes == [0, 1]
        w = probe["coef"][0].astype(np.float32)
        n = float(np.linalg.norm(w))
        print(f"  L{L}: ‖probe_coef‖ = {n:.3f}")
        out[L] = torch.from_numpy(-w / n).to(device).to(torch.float32)
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
    print(f"loaded {len(all_rows)} items (splits: {sorted({r['__split'] for r in all_rows})})")

    out_dir = data_root / "steering_llavanext" / f"probe_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state(out_path)
    if done:
        print(f"resume: {len(done)} records present ({len(baseline_preds)} baselines)")

    model, processor = load_llavanext(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    vectors = compute_probe_vectors(data_root, layers, args.device)
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
        done=done, out_f=out_f, letter_token_ids=letter_ids, desc="probe",
        hook_factory=lambda L, a: AdditiveHook(vectors[L], a), score_fn=score_row,
    )
    out_f.close()

    summary = compute_summary(out_path, pool, layers, alphas)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nwrote {out_dir/'summary.json'}")
    print_summary(summary, layers, alphas)


if __name__ == "__main__":
    main()
