"""Step 19a: materialize external MCQ benchmarks into canonical rows + images.

Downloads MMStar (full val, 1500), ViLP (full, ~300 questions x 3 variants),
and a fixed-seed group-preserving subsample of NaturalBench (~250 yes/no groups
= ~1000 items), writing each to data/benchmarks/<source>/{rows.jsonl,images/}.

CPU + network only; no model. Run once.

    python scripts/19a_prep_benchmarks.py
    python scripts/19a_prep_benchmarks.py --only mmstar
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

from sae_steering.benchmarks import prep_mmstar, prep_naturalbench, prep_vilp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/benchmarks")
    ap.add_argument("--only", default="all",
                    help="comma-separated subset of mmstar,vilp,naturalbench")
    ap.add_argument("--mmstar-max", type=int, default=0, help="0 = full val (1500)")
    ap.add_argument("--nb-groups", type=int, default=250,
                    help="number of NaturalBench yes/no groups (x4 items)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    which = {w.strip() for w in args.only.split(",")} if args.only != "all" \
        else {"mmstar", "vilp", "naturalbench"}

    if "mmstar" in which:
        print("\n=== MMStar ===")
        ds = load_dataset("Lin-Chen/MMStar", split="val")
        it = ds if args.mmstar_max <= 0 else ds.select(range(args.mmstar_max))
        prep_mmstar(it, out_root / "mmstar")

    if "vilp" in which:
        print("\n=== ViLP ===")
        ds = load_dataset("ViLP/ViLP", split="train")
        prep_vilp(ds, out_root / "vilp")

    if "naturalbench" in which:
        print("\n=== NaturalBench ===")
        # stream and keep the first N yes/no groups (deterministic, no full download)
        stream = load_dataset("BaiqiL/NaturalBench", split="train", streaming=True)
        kept, buf = 0, []
        for r in stream:
            if str(r.get("Question_Type", "")).lower() != "yes_no":
                continue
            buf.append(r)
            kept += 1
            if kept >= args.nb_groups:
                break
        prep_naturalbench(buf, out_root / "naturalbench")

    print("\nDone.")
    # streaming datasets spawn worker threads that crash on interpreter
    # finalization; files are already flushed, so exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
