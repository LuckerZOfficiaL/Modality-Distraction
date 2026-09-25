"""Step 18c: canonical-setup full-pool patching sweep.

Re-runs unconditional V-patch and T-patch at the plateau layers (27, 29, 31,
33, 35) under the canonical forward-pass setup (build_messages with
max_pixels=1024*1024). Required because the original 16_patching_diagnostic.py
runs used the pre-canonical setup, which produces ~7% script-drift vs the
canonical baseline.

For each (direction, layer), the script writes one jsonl:
  data/diagnostics/canonical_patching/{direction}_L{L}.jsonl
where direction ∈ {V, T}.

Each row: cid, pred, label, correct_index.
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
    SetActivationHook,
    load_canonical_pool,
    load_pool_rows_jsonl,
    load_cf_activations,
    run_canonical_forward as run_forward,
)
from sae_steering.models import load_qwen_vl
from sae_steering.steering import get_letter_token_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--pool", default="unseen", help="pool spec; or output-suffix label with --rows-jsonl")
    ap.add_argument("--rows-jsonl", default=None,
                    help="load eval pool directly from this rows JSONL (assembled/cross-domain sets)")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual",
                    help="cf-store subdir under data/activations/qwen/")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--directions", default="V,T", help="V,T or just one")
    ap.add_argument("--layers", default="27,29,31,33,35")
    ap.add_argument("--out-dir", default="data/diagnostics/canonical_patching")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # Pool + cf-store
    if args.rows_jsonl is not None:
        pool = load_pool_rows_jsonl(Path(args.rows_jsonl), data_root, args.cf_subdir)
        print(f"Pool (rows-jsonl, tag={args.pool}): {len(pool)}")
    else:
        pool = load_canonical_pool(data_root, dm_dir, Path(args.distraction_pool),
                                    pool=args.pool, cf_subdir=args.cf_subdir)
        print(f"Pool ({args.pool}): {len(pool)}")
    activations, cid_to_cf = load_cf_activations(data_root, cf_subdir=args.cf_subdir)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    directions = [d.strip() for d in args.directions.split(",")]
    layers = [int(L) for L in args.layers.split(",")]

    # condition idx in cf-store: VT=0, V=1, T=2
    cond_for_dir = {"V": 1, "T": 2}

    for direction in directions:
        cond_idx = cond_for_dir[direction]
        for L in layers:
            suffix = "" if args.pool == "unseen" else f"_{args.pool}"
            out_path = out_dir / f"{direction}_L{L}{suffix}.jsonl"
            done = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    if line.strip(): done.add(json.loads(line)["candidate_id"])

            f = out_path.open("a")
            for row in tqdm(pool, desc=f"{direction}-patch L{L}", leave=False):
                cid = row["candidate_id"]
                if cid in done: continue
                target_np = activations[cid_to_cf[cid], cond_idx, L].astype(np.float32)
                target = torch.tensor(target_np, device=args.device, dtype=torch.float32)
                hook = SetActivationHook(target)
                pred = run_forward(model, processor, row, letter_ids, hook=hook, layer_idx=L)
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
