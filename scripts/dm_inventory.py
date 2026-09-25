"""Print a per-source inventory of D_M survivors (vision + text grounded MCQs).

Walks data/dm/ and data/dm/vistext/ for every dm_all.jsonl or survivors.jsonl
and tallies label counts. Aggregate directories (those whose seeds.jsonl is a
union of other batches) are skipped to avoid double-counting — heuristic:
batch dirs whose names contain a hyphen or underscore-range (e.g. batch_1-3).

Usage:
  python scripts/dm_inventory.py
  python scripts/dm_inventory.py --root data
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


AGGREGATE_NAME_RE = re.compile(r"batch[_-]?\d+[-_]\d+", re.IGNORECASE)


def count(p: Path) -> tuple[int, int, int]:
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    c = Counter(r.get("label", "?") for r in rows)
    return len(rows), c.get("vision", 0), c.get("text", 0)


def is_aggregate(p: Path) -> bool:
    return any(AGGREGATE_NAME_RE.search(part) for part in p.parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data",
                    help="root containing dm/ (default: data)")
    ap.add_argument("--include-aggregates", action="store_true",
                    help="include aggregate batch dirs like batch_1-3 (default: skip)")
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[tuple[str, int, int, int]] = []
    for name in ("dm_all.jsonl", "survivors.jsonl"):
        for p in sorted((root / "dm").rglob(name)):
            if not args.include_aggregates and is_aggregate(p):
                continue
            try:
                n, v, t = count(p)
            except Exception as e:
                print(f"  ! failed {p}: {e}")
                continue
            rows.append((str(p), n, v, t))

    if not rows:
        print(f"no dm_all.jsonl / survivors.jsonl found under {root}")
        return

    width = max(len(r[0]) for r in rows)
    print(f"{'PATH':<{width}}  {'TOTAL':>7} {'V':>5} {'T':>5}")
    print("-" * (width + 22))
    by_domain: dict[str, tuple[int, int, int]] = {}
    for path, n, v, t in rows:
        print(f"{path:<{width}}  {n:>7} {v:>5} {t:>5}")
        domain = path.split("/", 2)[1] if "/" in path else "?"  # data/<domain>/...
        a, b, c = by_domain.get(domain, (0, 0, 0))
        by_domain[domain] = (a + n, b + v, c + t)

    print("-" * (width + 22))
    for domain, (n, v, t) in by_domain.items():
        print(f"{domain + ' (sum)':<{width}}  {n:>7} {v:>5} {t:>5}")
    total = sum(n for _, n, _, _ in rows)
    total_v = sum(v for _, _, v, _ in rows)
    total_t = sum(t for _, _, _, t in rows)
    print(f"{'GRAND TOTAL':<{width}}  {total:>7} {total_v:>5} {total_t:>5}")
    if not args.include_aggregates:
        print("(aggregate dirs like batch_1-3 skipped; pass --include-aggregates to include)")


if __name__ == "__main__":
    main()
