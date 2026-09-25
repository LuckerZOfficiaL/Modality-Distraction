"""Step 4i (post-processing): drop rows whose `caption_for_filter` (the text
passage) is empty from a text-QA D_M dataset directory. Re-registers
dm_all.jsonl, candidates.jsonl, and splits/.

Use this after 04g/04h on any text-QA source. It's idempotent: running on a
dataset whose passages are all non-empty is a no-op (no rows dropped).

Use case: ARC-Challenge and MMLU-hard provide no supporting passage by source
design, so 100% of their constructed rows have empty passages. This script
keeps the rest.

Usage:
  # one dir
  python scripts/filter_passage_present.py --dm-dir data/dm/text_qa/race_high

  # all subdirs of a root
  python scripts/filter_passage_present.py --dm-root data/dm/text_qa

The seed and split ratios are read from --config (default configs/pilot.yaml)
or use the built-in defaults (seed=0, splits 60/20/20).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _filter_file(path: Path) -> tuple[int, int]:
    """Drop rows with empty caption_for_filter. Returns (kept, dropped)."""
    if not path.is_file():
        return 0, 0
    rows = []
    dropped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("caption_for_filter") or "").strip():
            rows.append(r)
        else:
            dropped += 1
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows), dropped


def _regen_splits(dm_dir: Path, seed: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    rows = [json.loads(l) for l in (dm_dir / "dm_all.jsonl").read_text().splitlines() if l.strip()]
    splits_dir = dm_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    labels = sorted({r.get("label", "?") for r in rows})
    for lbl in labels:
        rs = [r for r in rows if r.get("label") == lbl]
        rng.shuffle(rs)
        n = len(rs)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        splits["train"].extend(rs[:n_train])
        splits["val"].extend(rs[n_train:n_train + n_val])
        splits["test"].extend(rs[n_train + n_val:])
    for name, rs in splits.items():
        with (splits_dir / f"{name}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    return {k: len(v) for k, v in splits.items()}


def filter_one(dm_dir: Path, seed: int, ratios: tuple[float, float, float]) -> None:
    kept_dm, dropped_dm = _filter_file(dm_dir / "dm_all.jsonl")
    kept_cand, dropped_cand = _filter_file(dm_dir / "candidates.jsonl")
    if kept_dm == 0 and dropped_dm == 0 and kept_cand == 0 and dropped_cand == 0:
        print(f"  {dm_dir}: no dm_all.jsonl or candidates.jsonl, skipped")
        return
    if dropped_dm:
        splits_summary = _regen_splits(dm_dir, seed, ratios)
        print(f"  {dm_dir}: kept {kept_dm} / dropped {dropped_dm} (dm_all)  +  "
              f"kept {kept_cand} / dropped {dropped_cand} (candidates); "
              f"splits regenerated -> {splits_summary}")
    else:
        print(f"  {dm_dir}: all rows already have passages — no-op")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-dir", default=None,
                    help="single dataset dir to filter")
    ap.add_argument("--dm-root", default=None,
                    help="root dir; every subdir containing dm_all.jsonl gets filtered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="0.6,0.2,0.2",
                    help="train,val,test fractions for split regeneration")
    args = ap.parse_args()
    if not (args.dm_dir or args.dm_root):
        ap.error("pass --dm-dir or --dm-root")
    ratios = tuple(float(x) for x in args.split.split(","))
    assert abs(sum(ratios) - 1.0) < 1e-6, "split must sum to 1.0"

    if args.dm_dir:
        targets = [Path(args.dm_dir)]
    else:
        root = Path(args.dm_root)
        targets = sorted(p.parent for p in root.glob("**/dm_all.jsonl"))

    if not targets:
        print("no dataset dirs found")
        return
    print(f"filtering {len(targets)} dataset dir(s):")
    for t in targets:
        filter_one(t, args.seed, ratios)


if __name__ == "__main__":
    main()
