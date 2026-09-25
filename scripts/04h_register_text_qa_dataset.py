"""Step 4h: register a text-QA-retrieved candidates.jsonl as a D_M dataset
directory that downstream scripts (06_collect_dm_activations, 07_train_probes,
11/12/13 steering) can consume directly.

Writes:
  <out-dir>/dm_all.jsonl           — every candidate as a survivor row
  <out-dir>/splits/{train,val,test}.jsonl  — 60/20/20 split per label

Each row carries `validation_regime: "construction"` to distinguish these
items from oracle-3pass-validated D_M survivors at downstream analysis time.

This script does NOT run any model. The items are valid T-grounded by
construction (the text-QA dataset guarantees T-only answerability; the
cross-domain image pool guarantees V-only insufficiency). Qwen behavioral
filtering is deferred to evaluation time (existing 04 / 06 / 11-13 scripts).

Usage:
  python scripts/04h_register_text_qa_dataset.py \
      --candidates data/dm/text_qa/race_high/candidates.jsonl \
      --out-dir data/dm/text_qa/race_high \
      --seed 0 --split 0.6,0.2,0.2
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="candidates.jsonl from 04g_retrieve_text_qa_candidates.py")
    ap.add_argument("--out-dir", required=True,
                    help="dataset dir; dm_all.jsonl + splits/ written here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="0.6,0.2,0.2",
                    help="comma-separated train,val,test fractions")
    ap.add_argument("--regime-tag", default="construction",
                    help="value of `validation_regime` on each emitted row")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(exist_ok=True)

    rows = [json.loads(l) for l in Path(args.candidates).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} candidates from {args.candidates}")

    # Annotate. Keep all original fields; add validation_regime + null pass_* slots
    # so the schema is union-compatible with oracle-validated dm_all.jsonl rows.
    enriched = []
    for r in rows:
        enriched.append({
            **r,
            "pass_vt": None,
            "pass_v":  None,
            "pass_t":  None,
            "validation_regime": args.regime_tag,
        })

    dm_all_path = out_dir / "dm_all.jsonl"
    with dm_all_path.open("w") as f:
        for r in enriched:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(enriched)} rows -> {dm_all_path}")

    # Per-label 60/20/20 split, same convention as 03d / 05.
    ratios = [float(x) for x in args.split.split(",")]
    assert abs(sum(ratios) - 1.0) < 1e-6, "split fractions must sum to 1.0"
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    labels_present = sorted({r.get("label", "?") for r in enriched})
    for label in labels_present:
        rows_l = [r for r in enriched if r.get("label") == label]
        rng.shuffle(rows_l)
        n = len(rows_l)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        splits["train"].extend(rows_l[:n_train])
        splits["val"].extend(rows_l[n_train:n_train + n_val])
        splits["test"].extend(rows_l[n_train + n_val:])
    for name, rs in splits.items():
        path = out_dir / "splits" / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        lc = Counter(r.get("label", "?") for r in rs)
        print(f"  {name}: {len(rs)} ({dict(lc)}) -> {path}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
