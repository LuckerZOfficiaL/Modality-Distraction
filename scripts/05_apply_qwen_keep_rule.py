"""Step 5: join qwen_candidates.jsonl with qwen_v_only.jsonl and apply the
per-label V-only majority filter to produce final survivors.jsonl + splits.

Per-label keep rule:
  text-grounded:   drop if Qwen V-only majority CORRECT  (Qwen didn't need caption)
  vision-grounded: drop if Qwen V-only majority WRONG    (Qwen can't see it)

Survivors carry pass_t, pass_vt, pass_v (the V-only majority outcome).

Works for any oracle backend (Gemini / Sonnet subagent) — both write
qwen_candidates.jsonl in the same format.

Usage:
  python scripts/05_apply_qwen_keep_rule.py --out-dir data/dm/gemini/batch2
  python scripts/05_apply_qwen_keep_rule.py --out-dir data/dm/sonnet_hardT/batch7
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="dir containing qwen_candidates.jsonl + qwen_v_only.jsonl "
                         "(same dir oracle pipeline wrote survivors.jsonl to)")
    ap.add_argument("--config", default=None,
                    help="optional yaml config for seed + split ratios "
                         "(defaults: seed=0, split=[0.6, 0.2, 0.2])")
    ap.add_argument("--candidates", default="qwen_candidates.jsonl")
    ap.add_argument("--qwen", default="qwen_v_only.jsonl")
    ap.add_argument("--survivors", default="survivors.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seed = 0
    split_ratios = [0.6, 0.2, 0.2]
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        seed = int(cfg.get("seed", 0))
        split_ratios = cfg.get("dm", {}).get("split", split_ratios)

    candidates = [
        json.loads(l) for l in (out_dir / args.candidates).read_text().splitlines() if l.strip()
    ]
    qwen_by_id: dict[str, dict] = {}
    for l in (out_dir / args.qwen).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        qwen_by_id[r["candidate_id"]] = r

    survivors = []
    skipped = 0
    for c in candidates:
        cid = c["candidate_id"]
        q = qwen_by_id.get(cid)
        if q is None or q.get("majority") is None:
            skipped += 1
            continue
        majority = int(q["majority"])
        v_ok = majority == c["correct_index"]
        if c["label"] == "text" and v_ok:
            continue
        if c["label"] == "vision" and not v_ok:
            continue
        survivors.append({
            **c,
            "pass_v": v_ok,
            "qwen_v_only_predictions": q.get("predictions", []),
            "qwen_v_only_majority": majority,
        })

    surv_path = out_dir / args.survivors
    with surv_path.open("w") as f:
        for r in survivors:
            f.write(json.dumps(r) + "\n")

    # per-label split, same logic as 03d
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label in ("vision", "text"):
        rows = [r for r in survivors if r["label"] == label]
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:])
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, rows in splits.items():
        (splits_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))

    label_counts = Counter(r["label"] for r in survivors)
    print(f"qwen_candidates: {len(candidates)} | skipped (no qwen majority): {skipped}")
    print(f"survivors: {len(survivors)} ({label_counts['vision']}v + {label_counts['text']}t)")
    for name, rows in splits.items():
        lc = Counter(r["label"] for r in rows)
        print(f"  {name}: {len(rows)} ({lc['vision']}v + {lc['text']}t)")
    print(f"-> {surv_path}")
    print(f"-> {splits_dir}/")


if __name__ == "__main__":
    main()
