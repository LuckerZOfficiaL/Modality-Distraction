"""Step 3d: apply 3-pass oracle keep-rule and split into train/val/test.

Keep-rule:
  vision-grounded: V+T correct AND V-only correct AND T-only wrong
  text-grounded:   V+T correct AND V-only wrong  AND T-only correct

Writes:
  data/dm/dm_all.jsonl          — all survivors, with pass_vt/v/t booleans added
  data/dm/splits/train.jsonl
  data/dm/splits/val.jsonl
  data/dm/splits/test.jsonl     (60/20/20, fixed seed)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml


def load_predictions(work_dir: Path) -> dict[str, int]:
    preds: dict[str, int] = {}
    for f in sorted(work_dir.glob("chunk_*.result.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "predicted_index" in r:
                preds[r["candidate_id"]] = r["predicted_index"]
    return preds


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    seed = int(cfg["seed"])
    split_ratios = cfg["dm"]["split"]

    candidates = [
        json.loads(l)
        for l in (dm_dir / "candidates.jsonl").read_text().splitlines()
        if l.strip()
    ]

    vt = load_predictions(dm_dir / "work" / "ans_vt")
    v  = load_predictions(dm_dir / "work" / "ans_v")
    t  = load_predictions(dm_dir / "work" / "ans_t")

    survivors: list[dict] = []
    n_skip = 0
    for c in candidates:
        cid = c["candidate_id"]
        ci = c["correct_index"]
        if cid not in vt or cid not in v or cid not in t:
            n_skip += 1
            continue
        vt_ok = vt[cid] == ci
        v_ok  = v[cid]  == ci
        t_ok  = t[cid]  == ci
        if c["label"] == "vision" and vt_ok and v_ok and not t_ok:
            survivors.append({**c, "pass_vt": True, "pass_v": True, "pass_t": False})
        elif c["label"] == "text" and vt_ok and not v_ok and t_ok:
            survivors.append({**c, "pass_vt": True, "pass_v": False, "pass_t": True})

    # write dm_all.jsonl
    dm_all_path = dm_dir / "dm_all.jsonl"
    with dm_all_path.open("w") as f:
        for row in survivors:
            f.write(json.dumps(row) + "\n")

    # split per label to preserve balance
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

    splits_dir = dm_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, rows in splits.items():
        path = splits_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in survivors)
    print(f"candidates: {len(candidates)} | skipped (missing preds): {n_skip}")
    print(f"survivors: {len(survivors)} ({label_counts['vision']} vision, {label_counts['text']} text)")
    for name, rows in splits.items():
        lc = Counter(r["label"] for r in rows)
        print(f"  {name}: {len(rows)} ({lc['vision']}v + {lc['text']}t)")
    print(f"dm_all -> {dm_all_path}")
    print(f"splits -> {splits_dir}/")


if __name__ == "__main__":
    main()
