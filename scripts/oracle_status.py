"""Report per-chunk progress of any oracle work pool and list incomplete chunks.

Usage:
  python scripts/oracle_status.py --work-dir data/dm/work/gen
  python scripts/oracle_status.py --work-dir data/dm/work/gen --json
  python scripts/oracle_status.py --work-dir data/dm/work/gen --pending  # prints chunk indices needing work, one per line
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def chunk_status(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    out: list[dict] = []
    for ch in manifest["chunks"]:
        in_path = Path(ch["input"])
        out_path = Path(ch["output"])
        n_in = sum(1 for l in in_path.read_text().splitlines() if l.strip())
        done_ids: set[str] = set()
        errors = 0
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("seed_id") or row.get("candidate_id")
                if sid:
                    done_ids.add(sid)
                if "error" in row:
                    errors += 1
        out.append({
            "index": ch["index"],
            "input": str(in_path),
            "output": str(out_path),
            "n_input": n_in,
            "n_done": len(done_ids),
            "n_remaining": n_in - len(done_ids),
            "n_errors": errors,
            "complete": len(done_ids) >= n_in,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pending", action="store_true",
                    help="print only indices of chunks with n_remaining > 0, one per line")
    args = ap.parse_args()
    manifest = Path(args.work_dir) / "manifest.json"
    rows = chunk_status(manifest)

    if args.pending:
        for r in rows:
            if not r["complete"]:
                print(r["index"])
        return
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    total_in = sum(r["n_input"] for r in rows)
    total_done = sum(r["n_done"] for r in rows)
    total_err = sum(r["n_errors"] for r in rows)
    print(f"{Path(args.work_dir)}: {total_done}/{total_in} rows done ({total_err} errors) across {len(rows)} chunks")
    for r in rows:
        mark = "\u2713" if r["complete"] else "."
        print(f"  [{mark}] chunk {r['index']:03d}: {r['n_done']:>3}/{r['n_input']:<3} "
              f"(remaining {r['n_remaining']}, errors {r['n_errors']})")


if __name__ == "__main__":
    main()
