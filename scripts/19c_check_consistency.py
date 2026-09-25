"""Step 19c: consistency checker for benchmark_eval method sweeps.

Guarantees that, for each pool tag, EVERY method/cell file shares the exact same
candidate_id set as that pool's baseline -- so cross-method comparisons are
apples-to-apples and a stray pool (e.g. two different rows.jsonl run under the
same --tag, or an over-broad analysis glob) is caught loudly.

For each tag (derived from baseline_<tag>.jsonl) it checks every *_<tag>.jsonl:
  - identical candidate_id SET to the baseline (no missing / no extra),
  - no duplicate candidate_ids within the file,
  - same row count as the baseline.

Exit code is non-zero if any check fails. CPU only, no model.

    python scripts/19c_check_consistency.py
    python scripts/19c_check_consistency.py --dir data/diagnostics/benchmark_eval --tags unseen,vilp
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def cid_list(path: Path) -> list[str]:
    return [json.loads(l)["candidate_id"]
            for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/diagnostics/benchmark_eval")
    ap.add_argument("--tags", default=None,
                    help="comma-separated subset; default = all baseline_<tag>.jsonl found")
    args = ap.parse_args()

    d = Path(args.dir)
    baselines = sorted(d.glob("baseline_*.jsonl"))
    tags = [b.name[len("baseline_"):-len(".jsonl")] for b in baselines]
    if args.tags:
        want = {t.strip() for t in args.tags.split(",")}
        tags = [t for t in tags if t in want]
    if not tags:
        print(f"no baseline_<tag>.jsonl in {d}")
        sys.exit(1)

    ok_all = True
    for tag in tags:
        base = d / f"baseline_{tag}.jsonl"
        b_list = cid_list(base)
        B = set(b_list)
        n = len(b_list)
        # all files for this tag except the baseline itself
        cells = [f for f in sorted(d.glob(f"*_{tag}.jsonl")) if f.name != base.name]
        print(f"\n=== {tag}  (baseline n={n}, {len(B)} unique; {len(cells)} cell files) ===")
        if len(B) != n:
            dups = [c for c, k in Counter(b_list).items() if k > 1]
            print(f"  !! BASELINE has {n-len(B)} duplicate cids, e.g. {dups[:3]}")
            ok_all = False
        bad = 0
        for f in cells:
            fl = cid_list(f)
            F = set(fl)
            problems = []
            if len(F) != len(fl):
                problems.append(f"{len(fl)-len(F)} dup cids")
            if F - B:
                problems.append(f"{len(F-B)} extra (not in baseline)")
            if B - F:
                problems.append(f"{len(B-F)} missing")
            if len(fl) != n:
                problems.append(f"n={len(fl)}!={n}")
            if problems:
                bad += 1; ok_all = False
                ex = next(iter((F - B) | (B - F)), None)
                print(f"  FAIL {f.name}: {'; '.join(problems)}" + (f"  e.g. {ex}" if ex else ""))
        if bad == 0:
            print(f"  OK — all {len(cells)} cell files match the baseline cid set exactly")
        else:
            print(f"  {bad}/{len(cells)} files inconsistent")

    print("\n" + ("ALL CONSISTENT" if ok_all else "INCONSISTENCIES FOUND"))
    sys.exit(0 if ok_all else 2)


if __name__ == "__main__":
    main()
