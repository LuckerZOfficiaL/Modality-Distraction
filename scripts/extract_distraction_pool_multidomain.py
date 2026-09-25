"""Step 4e: distraction pool extractor for multidomain D_M (train+val only).

Same logic as 04d but restricted to ONE dm dir (multidomain_v1) and to the
candidate_ids present in splits/train.jsonl ∪ splits/val.jsonl. Test is held
out from steering iteration, so test candidates never enter the distraction
pool.

A distraction event is a Qwen 3-pass prediction where joint V+T is wrong but
one single modality is right:
  - v_distracted (V✓ ∧ VT✗): adding caption distracts → steer toward V (label="vision")
  - t_distracted (T✓ ∧ VT✗): adding image distracts → steer toward T (label="text")
  - both alone right + joint wrong: defaulted to v_distracted (rare)

Output rows carry `label`, `distraction_kind`, audit-trail predictions, and
`source` (so per-domain breakdown still works). `__split` is set here for
consistency, but `load_distraction_pool` in steering.py overrides it to
"distraction".

Usage:
  python scripts/extract_distraction_pool_multidomain.py \
      --dm-dir data/dm/multidomain_v1 \
      --out    data/dm/multidomain_v1/distraction_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-dir", default="data/dm/multidomain_v1")
    ap.add_argument("--out", default=None,
                    help="default: <dm-dir>/distraction_pool.jsonl")
    ap.add_argument("--include-splits", nargs="+", default=["train", "val"],
                    help="which splits to draw candidates from (default: train val; test excluded)")
    args = ap.parse_args()

    dm_dir = Path(args.dm_dir)
    out_path = Path(args.out) if args.out else dm_dir / "distraction_pool.jsonl"

    # candidate_ids allowed by the new policy (train ∪ val)
    allowed: set[str] = set()
    for sub in args.include_splits:
        sp = dm_dir / "splits" / f"{sub}.jsonl"
        for l in sp.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            allowed.add(r["candidate_id"])
    print(f"allowed cids from splits {args.include_splits}: {len(allowed)}")

    # candidates (full row content) and qwen 3-pass predictions
    cands: dict[str, dict] = {}
    for l in (dm_dir / "candidates.jsonl").read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        cands[r["candidate_id"]] = r

    preds: dict[str, dict] = {}
    for l in (dm_dir / "pass_qwen_candidates" / "predictions.jsonl").read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        preds[r["candidate_id"]] = r

    out_rows: list[dict] = []
    kind_counter = Counter()
    per_source = Counter()

    for cid, p in preds.items():
        if cid not in allowed:
            continue
        if cid not in cands:
            continue
        ci = p["correct_index"]
        vt_ok = p["vt"] == ci
        v_ok  = p["v"]  == ci
        t_ok  = p["t"]  == ci
        if vt_ok:
            continue
        if v_ok and not t_ok:
            kind, label = "v_distracted", "vision"
        elif t_ok and not v_ok:
            kind, label = "t_distracted", "text"
        elif v_ok and t_ok:
            kind, label = "v_distracted", "vision"
        else:
            continue
        c = cands[cid]
        out_rows.append({
            "candidate_id": cid,
            "seed_id": c.get("seed_id"),
            "source": c.get("source"),
            "image_path": c["image_path"],
            "original_caption": c.get("original_caption"),
            "label": label,
            "question": c["question"],
            "options": c["options"],
            "correct_index": c["correct_index"],
            "caption_for_filter": c["caption_for_filter"],
            "rationale": c.get("rationale", ""),
            "distraction_kind": kind,
            "qwen_pred_vt": p["vt"],
            "qwen_pred_v":  p["v"],
            "qwen_pred_t":  p["t"],
            "__split": "distraction",
            "source_dm_dir": str(dm_dir),
        })
        kind_counter[kind] += 1
        per_source[(c.get("source", "?"), kind)] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(out_rows)} rows -> {out_path}")
    print(f"  v_distracted: {kind_counter['v_distracted']}")
    print(f"  t_distracted: {kind_counter['t_distracted']}")
    print("\nper-source × kind:")
    for (src, kind), n in sorted(per_source.items()):
        print(f"  {src:<10s} {kind:<14s} {n}")


if __name__ == "__main__":
    main()
