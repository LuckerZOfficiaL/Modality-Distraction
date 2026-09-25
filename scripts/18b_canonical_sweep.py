"""Step 18b: canonical-setup sweep over (layer, alpha) for the additive V-causal
direction (and its matched random null).

Uses the exact same forward-pass setup as scripts/18_canonical_eval.py:
build_messages with max_pixels=1024*1024, single-item batching. Output is
one jsonl per (layer, alpha, kind) cell.

kind ∈ {"real", "random"}: real uses v_V_causal at the layer; random uses
a fresh seed-0 unit vector (one vector shared across layers for the
matched-null comparison; each layer hooks it independently). The hook is
AdditiveHook with broadcast=True, norm_match=True (bcast+normmatch).

Output: data/diagnostics/canonical_sweep/<kind>_L<L>_a<a>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sae_steering.canonical_eval import (
    load_canonical_pool,
    load_pool_rows_jsonl,
    run_canonical_forward as run_forward,
)
from sae_steering.models import load_qwen_vl
from sae_steering.steering import AdditiveHook, get_letter_token_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--pool", default="unseen", help="pool spec; or output-suffix label with --rows-jsonl")
    ap.add_argument("--rows-jsonl", default=None,
                    help="load eval pool directly from this rows JSONL (assembled/cross-domain sets)")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual",
                    help="cf-store subdir under data/activations/qwen/")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--layers", default="13,20,23,28",
                    help="comma-separated layers to sweep")
    ap.add_argument("--alphas", default="-3,-1,-0.5,-0.2,0.2,0.5,1,3")
    ap.add_argument("--kinds", default="real,random",
                    help="which directions to run: real, random, or both")
    ap.add_argument("--out-dir", default="data/diagnostics/canonical_sweep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steering-vectors",
                    default="data/steering/canonical/v_causal_vectors.npz")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # Canonical pool
    if args.rows_jsonl is not None:
        pool = load_pool_rows_jsonl(Path(args.rows_jsonl), data_root, args.cf_subdir)
        print(f"pool (rows-jsonl, tag={args.pool}): {len(pool)} cids")
    else:
        pool = load_canonical_pool(data_root, dm_dir, Path(args.distraction_pool),
                                    pool=args.pool, cf_subdir=args.cf_subdir)
        print(f"pool ({args.pool}): {len(pool)} cids")

    # Load steering vectors
    sv = np.load(args.steering_vectors)
    v_V = {L: torch.tensor(sv[f"v_V_L{L}"], device=args.device, dtype=torch.float32)
           for L in [13, 20, 23, 28]}

    # Random unit vector (seed 0), 2048-d -- shared across layers
    rng = np.random.default_rng(args.random_seed)
    v_rand_np = rng.standard_normal(2048).astype(np.float32)
    v_rand_np /= np.linalg.norm(v_rand_np)
    v_rand = torch.tensor(v_rand_np, device=args.device, dtype=torch.float32)

    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]
    print(f"Layers: {layers}, alphas: {alphas}, kinds: {kinds}")
    print(f"Total cells: {len(layers) * len(alphas) * len(kinds)}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    # Sweep
    for L in layers:
        for a in alphas:
            for kind in kinds:
                a_str = f"{a:+g}".replace("+", "p").replace("-", "m")
                suffix = "" if args.pool == "unseen" else f"_{args.pool}"
                out_path = out_dir / f"{kind}_L{L}_a{a_str}{suffix}.jsonl"

                done = set()
                if out_path.exists():
                    for line in out_path.read_text().splitlines():
                        if line.strip():
                            done.add(json.loads(line)["candidate_id"])

                vec = v_V[L] if kind == "real" else v_rand
                hook = AdditiveHook(vec, alpha=a, broadcast=True, norm_match=True)

                f = out_path.open("a")
                for row in tqdm(pool, desc=f"{kind} L{L} α={a}", leave=False):
                    cid = row["candidate_id"]
                    if cid in done: continue
                    pred = run_forward(model, processor, row, letter_ids,
                                       hook=hook, layer_idx=L)
                    f.write(json.dumps({
                        "candidate_id": cid,
                        "label": row["label"],
                        "correct_index": row["correct_index"],
                        "pred": pred,
                    }) + "\n")
                    f.flush()
                f.close()
                print(f"  done: {out_path.name}")


if __name__ == "__main__":
    main()
