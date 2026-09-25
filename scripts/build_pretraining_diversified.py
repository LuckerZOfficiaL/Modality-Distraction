"""Merge per-source pretraining indices into a single diversified_v1 corpus.

Inputs (each {"key", "image_path", "caption"} per line):
  data/pretraining/cc3m_subset/index.jsonl
  data/pretraining/plotqa/index.jsonl
  data/pretraining/wikiart/index.jsonl
  data/pretraining/openi/index.jsonl

Output:
  data/pretraining/diversified_v1/index.jsonl   (interleaved random order)
  data/pretraining/diversified_v1/source_stats.json
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data/pretraining"
SOURCES = [
    ("cc3m",    DATA_ROOT / "cc3m_subset" / "index.jsonl",  None),    # None = use all
    ("plotqa",  DATA_ROOT / "plotqa"      / "index.jsonl",  None),
    ("wikiart", DATA_ROOT / "wikiart"     / "index.jsonl",  None),
    ("openi",   DATA_ROOT / "openi"       / "index.jsonl",  None),
]
OUT_DIR = DATA_ROOT / "diversified_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rng = random.Random(0)
    all_rows: list[dict] = []
    stats: dict[str, int] = {}
    for name, idx, cap in SOURCES:
        if not idx.exists():
            print(f"  WARN: {idx} not found, skipping")
            stats[name] = 0
            continue
        rows = [json.loads(l) for l in idx.read_text().splitlines() if l.strip()]
        # tag with source for analysis
        for r in rows:
            r["source"] = name
        if cap is not None and len(rows) > cap:
            rng.shuffle(rows)
            rows = rows[:cap]
        stats[name] = len(rows)
        all_rows.extend(rows)
        print(f"  {name}: {len(rows)}")

    rng.shuffle(all_rows)
    out_idx = OUT_DIR / "index.jsonl"
    with out_idx.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    (OUT_DIR / "source_stats.json").write_text(json.dumps({
        "total": len(all_rows),
        "per_source": stats,
    }, indent=2))
    print(f"\ntotal: {len(all_rows)} -> {out_idx}")


if __name__ == "__main__":
    main()
