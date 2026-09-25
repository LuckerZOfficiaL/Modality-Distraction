"""Filter an assembled (cross-domain) dataset dir to the top-K% by retrieval_score.

Reads <src_dir>/dm_all.jsonl, sorts by retrieval_score (desc), keeps the top
--top-frac fraction, then writes a sibling <src_dir>_filtered/ with:
  - dm_all.jsonl       (filtered rows)
  - splits/{train,val,test}.jsonl   (re-split 60/20/20 deterministically)
  - candidates.jsonl   (copy of filtered dm_all rows for downstream tools)

Usage:
  python scripts/04j_filter_top_similarity.py --src-dir data/dm/text_qa/openbookqa --top-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-fracs", default="0.6,0.2,0.2",
                    help="train,val,test fractions of the filtered set")
    args = ap.parse_args()

    src = Path(args.src_dir)
    dst = src.with_name(src.name + "_filtered")
    rows = [json.loads(l) for l in (src / "dm_all.jsonl").read_text().splitlines() if l.strip()]
    n_in = len(rows)
    rows.sort(key=lambda r: r["retrieval_score"], reverse=True)
    k = int(round(n_in * args.top_frac))
    kept = rows[:k]
    print(f"{src}: {n_in} -> {len(kept)} (top {args.top_frac:.0%}, score >= {kept[-1]['retrieval_score']:.4f})")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    ft, fv, fte = (float(x) for x in args.split_fracs.split(","))
    n_train = int(round(len(kept) * ft))
    n_val = int(round(len(kept) * fv))
    splits = {
        "train": kept[:n_train],
        "val":   kept[n_train:n_train + n_val],
        "test":  kept[n_train + n_val:],
    }

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "splits").mkdir(exist_ok=True)
    with (dst / "dm_all.jsonl").open("w") as f:
        for r in kept: f.write(json.dumps(r) + "\n")
    with (dst / "candidates.jsonl").open("w") as f:
        for r in kept: f.write(json.dumps(r) + "\n")
    for name, rows_ in splits.items():
        with (dst / "splits" / f"{name}.jsonl").open("w") as f:
            for r in rows_: f.write(json.dumps(r) + "\n")
        print(f"  splits/{name}.jsonl: {len(rows_)}")
    print(f"wrote -> {dst}")


if __name__ == "__main__":
    main()
