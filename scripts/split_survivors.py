"""Stratified train/val/test split of an existing survivors.jsonl.

Use when oracle filtering already ran and you just want to materialize
splits/{train,val,test}.jsonl matching the project's standard 60/20/20
seeded-shuffle policy.

Example:
    .venv/bin/python scripts/split_survivors.py \\
        --survivors data/dm/gemini/survivors.jsonl \\
        --out-dir   data/dm/gemini/splits
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.6, 0.2, 0.2],
                    help="train val test")
    args = ap.parse_args()

    survivors = [json.loads(l) for l in Path(args.survivors).read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label in ("vision", "text"):
        rows = [r for r in survivors if r["label"] == label]
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * args.ratios[0])
        n_val   = int(n * args.ratios[1])
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        lc = Counter(r["label"] for r in rows)
        print(f"  {name}: {len(rows)} ({lc['vision']}v + {lc['text']}t) -> {path}")
    print(f"total: {len(survivors)}")


if __name__ == "__main__":
    main()
