"""Materialize a canonical pool to disk (cid list + row data).

Useful for:
  - Reproducibility: pin the exact cid set used by an experiment.
  - Documentation: have a named pool definition that other scripts can read.
  - Extending the cf-store: pass --emit-row-jsonl to write rows that 06c-style
    activation collectors can consume.

Examples
--------

  # The default headline pool (val_all + train_fails + distraction):
  python scripts/collect_pool_cids.py --pool unseen --out data/pools/unseen

  # Held-out test:
  python scripts/collect_pool_cids.py --pool test_all --out data/pools/test

  # Composite (any combination):
  python scripts/collect_pool_cids.py \\
      --pool "val_all + test_all + distraction" --out data/pools/val_test_distraction

  # Only the Qwen-modality-isolated val subset:
  python scripts/collect_pool_cids.py --pool val_pass_all --out data/pools/val_pass

Output
------
  <out>/cids.txt         one candidate_id per line
  <out>/rows.jsonl       full row dicts (image_path, caption, question, options, ...)
  <out>/SPEC.txt         human-readable pool spec for provenance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from moground.canonical_eval import load_canonical_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--pool", required=True,
                    help="pool spec: named alias (unseen/test/all/val/train/etc) "
                         "or composite like 'val_all + train_fails + distraction'")
    ap.add_argument("--out", required=True,
                    help="output directory (created if absent)")
    ap.add_argument("--ignore-cf-store", action="store_true",
                    help="don't filter to cf-store; useful when collecting a pool "
                         "you'll later run 06c on to extend the cf-store")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    if args.ignore_cf_store:
        # Bypass the cf-store filter by pointing at a temporary fake subdir, or
        # by editing load_canonical_pool. Simpler: do a manual run that skips
        # the cf-store filter.
        from moground.canonical_eval import _parse_pool_spec
        from moground.steering import load_distraction_pool
        components = _parse_pool_spec(args.pool)
        print(f"[collect] components={components}")

        rows: list[dict] = []
        base_splits = set()
        needs_dist = "distraction" in components
        for c in components:
            if c == "distraction": continue
            split = c.rpartition("_")[0]
            base = split.removesuffix("_pass") if split.endswith("_pass") else split
            base_splits.add(base)
        for sp in base_splits:
            for line in (dm_dir / "splits" / f"{sp}.jsonl").read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line); r["__split"] = sp; rows.append(r)
        if needs_dist:
            rows = rows + load_distraction_pool(Path(args.distraction_pool))
        # Naive: select everything from the unioned splits and distraction.
        # User running with --ignore-cf-store should know what they're asking for.
        pool = rows
        print(f"[collect] --ignore-cf-store: returning all {len(pool)} rows from "
              f"requested splits (no filter on canonical baseline either; this is "
              f"a row dump for cf-store extension, not an evaluation pool).")
    else:
        pool = load_canonical_pool(data_root, dm_dir, Path(args.distraction_pool),
                                   pool=args.pool)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cids.txt").write_text("\n".join(r["candidate_id"] for r in pool) + "\n")
    with (out_dir / "rows.jsonl").open("w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    (out_dir / "SPEC.txt").write_text(
        f"pool spec: {args.pool}\n"
        f"cf-store filter: {'OFF' if args.ignore_cf_store else 'ON'}\n"
        f"size: {len(pool)} cids\n"
    )
    print(f"\nWrote {len(pool)} cids to {out_dir}/")
    print(f"  cids.txt  ({len(pool)} lines)")
    print(f"  rows.jsonl  ({len(pool)} rows)")
    print(f"  SPEC.txt")


if __name__ == "__main__":
    main()
