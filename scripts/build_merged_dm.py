"""Build a balanced V+T merged D_M dir from two single-modality sources.

Sub-samples each side to the requested per-class-per-split counts (deterministic
RNG keyed by --seed), concatenates into dm_all.jsonl, writes splits/{train,val,test}.jsonl,
and (optionally) seeds pass/{train,val}.jsonl as copies of the splits so 06a/07 can
operate without a separate Qwen-filter pass first (--seed-pass-from-splits).

Usage:
  python scripts/build_merged_dm.py \
      --v-source data/dm/v_mcq/aokvqa \
      --t-source data/dm/text_qa/race_high \
      --out-dir data/dm/merged_aokvqa_racehigh \
      --counts 1500,500,500 \
      --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def _load_split(dir_: Path, split: str) -> list[dict]:
    p = dir_ / "splits" / f"{split}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-source", required=True, help="dir with splits/{train,val,test}.jsonl, all label='vision'")
    ap.add_argument("--t-source", required=True, help="dir with splits/{train,val,test}.jsonl, all label='text'")
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--counts",   default="1500,500,500",
                    help="train,val,test PER CLASS (per modality). E.g. 1500,500,500 yields "
                         "3000 train, 1000 val, 1000 test in total (half V, half T)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-pass-from-splits", action="store_true",
                    help="also write pass/{train,val}.jsonl as copies of the splits "
                         "(so 06a can run without a separate step-04 Qwen filter — "
                         "use this when training a probe on construction-validated data)")
    args = ap.parse_args()

    n_train, n_val, n_test = (int(x) for x in args.counts.split(","))
    counts = {"train": n_train, "val": n_val, "test": n_test}
    rng = random.Random(args.seed)

    v_root, t_root, out_root = Path(args.v_source), Path(args.t_source), Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(exist_ok=True)
    if args.seed_pass_from_splits:
        (out_root / "pass").mkdir(exist_ok=True)

    merged_all = []
    for split in ("train", "val", "test"):
        v_rows = _load_split(v_root, split)
        t_rows = _load_split(t_root, split)
        need = counts[split]
        for rows, kind in [(v_rows, "vision"), (t_rows, "text")]:
            if len(rows) < need:
                print(f"  ! {split}/{kind}: only {len(rows)} available, need {need}; using all")
        rng.shuffle(v_rows); rng.shuffle(t_rows)
        v_pick = v_rows[:need]
        t_pick = t_rows[:need]
        # ensure label is set correctly (overwrite in case sources had different labels)
        for r in v_pick: r["label"] = "vision"
        for r in t_pick: r["label"] = "text"
        rows = v_pick + t_pick
        rng.shuffle(rows)
        out = out_root / "splits" / f"{split}.jsonl"
        with out.open("w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        merged_all.extend(rows)
        lc = Counter(r["label"] for r in rows)
        print(f"  {split}: wrote {len(rows)} ({dict(lc)}) -> {out}")
        # seed pass/ from splits if requested (skip test — never VLM-filtered by convention)
        if args.seed_pass_from_splits and split in ("train", "val"):
            shutil.copy(out, out_root / "pass" / f"{split}.jsonl")

    with (out_root / "dm_all.jsonl").open("w") as f:
        for r in merged_all:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(merged_all)} total rows -> {out_root / 'dm_all.jsonl'}")
    if args.seed_pass_from_splits:
        print("pass/{train,val}.jsonl seeded from splits/ (skip step 04 if you trust the "
              "construction-validation; otherwise run 04_vlm_behavioral_filter.py on this dir).")


if __name__ == "__main__":
    main()
