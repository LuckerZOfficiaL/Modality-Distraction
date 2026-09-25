"""Template for evaluating a new steering method canonically.

To use:
  1. Copy this file to scripts/<new_method_name>.py.
  2. Replace the body of make_hook() to construct your hook.
  3. (Optional) Pick layer_idx and any cell-specific config.
  4. The script will write data/diagnostics/canonical_<method>.jsonl alongside
     the existing canonical_eval/ files, directly comparable to the baseline.

This template enforces the canonical setup so your numbers are within-script
clean against the canonical baseline (pool acc 0.671 on 948 cids).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from sae_steering.canonical_eval import (
    load_canonical_pool,
    load_cf_activations,
    random_matched_unit_vector,
    run_canonical_forward,
    compare_to_baseline,
)
from sae_steering.models import load_qwen_vl
from sae_steering.steering import (
    AdditiveHook,
    get_letter_token_ids,
)


def make_hook(args, *, kind: str, cf_activations, cid_to_cf, device: str):
    """Return (hook, layer_idx) for the given kind. Override this for new methods.

    kind ∈ {"real", "random"}. Always implement the random branch with the same
    structural hook so it serves as a matched-magnitude null.
    """
    # ----- EXAMPLE: additive V-direction at L28, alpha=+1, bcast+normmatch -----
    # Replace this section with your method's hook construction.

    import numpy as np
    # Real direction would normally come from a steering vector file; this
    # example uses the v_V_L28 vector from the project's cf-contrastive run.
    sv = np.load("data/steering/canonical/v_causal_vectors.npz")
    v_real = torch.tensor(sv["v_V_L28"], device=device, dtype=torch.float32)
    layer_idx = 28
    alpha = 1.0

    if kind == "real":
        hook = AdditiveHook(v_real, alpha=alpha, broadcast=True, norm_match=True)
    elif kind == "random":
        v_rand = random_matched_unit_vector(2048, seed=0, device=device)
        hook = AdditiveHook(v_rand, alpha=alpha, broadcast=True, norm_match=True)
    else:
        raise ValueError(kind)
    return hook, layer_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--out-name", required=True,
                    help="output file name stem; writes data/diagnostics/canonical_<NAME>_{real,random}.jsonl")
    ap.add_argument("--kinds", default="real,random",
                    help="comma-separated; always run both real and random for the matched null")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # ---- Canonical pool + cf-store ----
    pool = load_canonical_pool(data_root, dm_dir, Path(args.distraction_pool))
    print(f"Pool: {len(pool)} cids")

    cf_activations, cid_to_cf = load_cf_activations(data_root)

    # ---- Model ----
    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    out_dir = Path("data/diagnostics"); out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [k.strip() for k in args.kinds.split(",")]

    for kind in kinds:
        hook, layer_idx = make_hook(args, kind=kind, cf_activations=cf_activations,
                                     cid_to_cf=cid_to_cf, device=args.device)
        out_path = out_dir / f"canonical_{args.out_name}_{kind}.jsonl"
        done = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip(): done.add(json.loads(line)["candidate_id"])

        f = out_path.open("a")
        for row in tqdm(pool, desc=f"{args.out_name} ({kind})", leave=False):
            cid = row["candidate_id"]
            if cid in done: continue
            pred = run_canonical_forward(model, processor, row, letter_ids,
                                          hook=hook, layer_idx=layer_idx)
            f.write(json.dumps({
                "candidate_id": cid,
                "label": row["label"],
                "correct_index": row["correct_index"],
                "pred": pred,
            }) + "\n")
            f.flush()
        f.close()

        # Final readout
        summary = compare_to_baseline(out_path)
        print(f"\n{kind}: pool={summary['method_pool_acc']:.4f}  "
              f"Δ={summary['delta_pool']:+.4f}  (baseline {summary['baseline_pool_acc']:.4f}, n={summary['n']})")


if __name__ == "__main__":
    main()
