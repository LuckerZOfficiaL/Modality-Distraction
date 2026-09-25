"""Restore the 2,500 CC3M distractor images referenced by the assembled pool
(merged_aokvqa_racehigh text items), which were lost in the 2026-07-14 disk cleanup.

Streams pixparse/cc3m-wds (same source + key scheme as scripts/download_cc3m_subset.py), keeps ONLY the needed
keys, writes them back to the original paths so dm_all.jsonl image_path fields resolve
again. Resume-safe (skips files already on disk); stops as soon as all keys are found.

    python scripts/restore_merged_cc3m_images.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "data/dm/merged_aokvqa_racehigh/dm_all.jsonl").read_text().splitlines()
            if l.strip()]
    need = {}  # key -> target path
    for r in rows:
        p = Path(r["image_path"])
        if "cc3m_subset" in str(p) and not p.exists():
            need[p.stem] = p
    print(f"need {len(need)} images")
    if not need:
        print("nothing to restore"); return
    for p in need.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)
    found = 0
    pbar = tqdm(ds, desc="scanning cc3m-wds")
    for ex in pbar:
        key = ex.get("__key__")
        if key in need:
            img = ex["jpg"]
            img.convert("RGB").save(need[key], "JPEG", quality=95)
            found += 1
            pbar.set_postfix(found=found, missing=len(need) - found)
            del need[key]
            if not need:
                break
    print(f"restored {found}; still missing {len(need)}")
    if need:
        (ROOT / "data/dm/merged_aokvqa_racehigh/missing_cc3m_keys.json").write_text(
            json.dumps(sorted(need.keys()), indent=1))
        print("missing keys written to data/dm/merged_aokvqa_racehigh/missing_cc3m_keys.json")


if __name__ == "__main__":
    main()
