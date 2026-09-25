"""LLaVA-NeXT SAE-feature steering at L24 (released lmms-lab SAE)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sparsify import Sae

from sae_steering.models_llavanext import load_llavanext
from sae_steering.steering import (
    POOL_CHOICES, baseline_rows_needed, compute_summary,
    get_letter_token_ids, load_resume_state, print_summary,
    run_baseline_phase, run_hooked_phase, select_pool,
)
from sae_steering.steering_llavanext import (
    SAEHookSparsify, score_row, load_oracle_passed_llavanext,
)


SAE_HUB = "lmms-lab/llama3-llava-next-8b-hf-sae-131k"


def load_top_features(data_root: Path, layer: int, top_n: int) -> tuple[list[int], list[int]]:
    feats = json.loads((data_root / f"features_llavanext/layer{layer}/top_features.json").read_text())
    return ([r["k"] for r in feats["top_vision"][:top_n]],
            [r["k"] for r in feats["top_text"][:top_n]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layers", default="24")
    ap.add_argument("--alphas", default="-100,-30,-10,-3,-1,-0.5,0.5,1,3,10,30,100")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--pool", choices=POOL_CHOICES, default="unseen")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    all_rows = load_oracle_passed_llavanext(data_root, baseline_rows_needed(args.pool))
    print(f"loaded {len(all_rows)} items")

    out_dir = data_root / "steering_llavanext" / f"sae_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state(out_path)

    model, processor = load_llavanext(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    saes: dict[int, Sae] = {}
    feat_idx: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    feat_log: dict[str, list[int]] = {}
    for L in layers:
        saes[L] = Sae.load_from_hub(SAE_HUB, hookpoint=f"model.layers.{L}").to(args.device).eval()
        v, t = load_top_features(data_root, L, args.top_n)
        feat_idx[L] = (torch.tensor(v, dtype=torch.long, device=args.device),
                       torch.tensor(t, dtype=torch.long, device=args.device))
        feat_log[f"L{L}_vision"] = v; feat_log[f"L{L}_text"] = t
        print(f"  L{L}: top {len(v)} vision feats, top {len(t)} text feats")
    json.dump(feat_log, open(out_dir / "feature_indices.json", "w"), indent=2)

    out_f = out_path.open("a")
    run_baseline_phase(model=model, processor=processor, rows=all_rows, done=done,
                       out_f=out_f, baseline_preds=baseline_preds,
                       letter_token_ids=letter_ids, score_fn=score_row)

    pool = select_pool(all_rows, baseline_preds, args.pool)
    n_v = sum(1 for r in pool if r["label"] == "vision")
    n_t = sum(1 for r in pool if r["label"] == "text")
    n_test = sum(1 for r in pool if r["__split"] == "test")
    print(f"pool ({args.pool}): {len(pool)} (test={n_test}, train+val={len(pool)-n_test}; {n_v}v {n_t}t)")

    def hook_factory(L, a):
        v_idx, t_idx = feat_idx[L]
        return SAEHookSparsify(saes[L], v_idx, t_idx, a)

    delta_stats: dict[tuple[int, float], dict] = {}
    def post_call(L, a, hook):
        s = delta_stats.setdefault((L, a), {"norms": [], "zero": 0, "n": 0})
        s["norms"].extend(hook.delta_norms)
        s["zero"] += hook.zero_count
        s["n"] += len(hook.delta_norms)

    run_hooked_phase(
        model=model, processor=processor, pool=pool, layers=layers, alphas=alphas,
        done=done, out_f=out_f, letter_token_ids=letter_ids, desc="sae",
        hook_factory=hook_factory, post_call=post_call, score_fn=score_row,
    )
    out_f.close()

    delta_log = {}
    for (L, a), s in sorted(delta_stats.items()):
        mean_n = float(np.mean(s["norms"])) if s["norms"] else 0.0
        delta_log[f"L{L}|alpha={a}"] = {"n": s["n"], "mean_delta_norm": mean_n, "zero_count": s["zero"]}
        print(f"  L{L} alpha={a:+g}: mean‖Δ‖={mean_n:.4f} zero={s['zero']}")
    json.dump(delta_log, open(out_dir / "delta_stats.json", "w"), indent=2)

    summary = compute_summary(out_path, pool, layers, alphas)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"wrote {out_dir/'summary.json'}")
    print_summary(summary, layers, alphas)


if __name__ == "__main__":
    main()
