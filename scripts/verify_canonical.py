"""Quick sanity check: confirm a fresh canonical baseline forward pass matches
the saved baseline on a small sample of items.

Run after any environment change (new transformers version, new model load
pattern, switched attention impl). If agreement drops below 95% on the
sample, the canonical setup has drifted; investigate before trusting any
new method's numbers.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from sae_steering.canonical_eval import (
    load_canonical_pool,
    run_canonical_forward,
)
from sae_steering.models import load_qwen_vl
from sae_steering.steering import get_letter_token_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--baseline-file",
                    default="data/diagnostics/canonical_eval/baseline.jsonl")
    ap.add_argument("--n-sample", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    pool = load_canonical_pool(data_root, dm_dir, Path(args.distraction_pool))
    saved = {r["candidate_id"]: r["pred"] for r in
             (json.loads(l) for l in Path(args.baseline_file).read_text().splitlines() if l.strip())}

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n_sample, len(pool)))

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    agree = 0
    diffs = []
    for row in sample:
        cid = row["candidate_id"]
        if cid not in saved: continue
        pred = run_canonical_forward(model, processor, row, letter_ids)
        if pred == saved[cid]:
            agree += 1
        else:
            diffs.append((cid, pred, saved[cid]))

    n = len(sample)
    print(f"Sampled {n} cids, agree {agree}/{n} = {agree/n:.4f}")
    if agree / n < 0.95:
        print(f"\n*** WARNING: canonical setup may have drifted. ***")
        print(f"First few disagreements (cid, fresh_pred, saved_pred):")
        for d in diffs[:10]: print(f"  {d}")
    elif agree / n < 0.99:
        print(f"Note: <99% agreement is unusual. Mild drift may have occurred.")
    else:
        print(f"OK: canonical setup is bit-aligned (within 1.4% irreducible floor).")


if __name__ == "__main__":
    main()
