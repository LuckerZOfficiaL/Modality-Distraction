"""Step 5: Stream a subset of CC3M (image, caption) pairs to disk.

Source: pixparse/cc3m-wds (HF dataset, webdataset-packaged, includes JPEGs).

Outputs:
  data/cc3m_subset/images/<key>.jpg     # JPEG-encoded image
  data/cc3m_subset/index.jsonl          # one row: {"key", "image_path", "caption"}

Resume-safe: re-running picks up where it left off by reading the existing
index.jsonl. Streams in order, so HF's deterministic shard order plus our
existing-key set means no duplicate downloads. D_M seeds come from COCO
val2017, not CC3M, so no cross-dataset dedup needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--target", type=int, default=200_000, help="number of pairs to retain")
    ap.add_argument("--min-side", type=int, default=128, help="reject images with min(W,H) below this")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["paths"]["cc3m_dir"])
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    have: set[str] = set()
    if index_path.exists():
        for l in index_path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            have.add(r["key"])
    print(f"resuming with {len(have)} existing pairs; target {args.target}")
    if len(have) >= args.target:
        print("already at target.")
        return

    ds = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)
    pbar = tqdm(total=args.target, initial=len(have), desc="cc3m")
    n_skip_size = n_skip_existing = n_skip_decode = 0
    with index_path.open("a") as idx:
        for sample in ds:
            if len(have) >= args.target:
                break
            key = sample["__key__"]
            if key in have:
                n_skip_existing += 1
                if n_skip_existing % 1000 == 0:
                    pbar.set_postfix(skipped_existing=n_skip_existing)
                continue
            try:
                img: Image.Image = sample["jpg"]
            except Exception:
                n_skip_decode += 1
                continue
            if min(img.size) < args.min_side:
                n_skip_size += 1
                continue
            caption = sample["txt"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_path = img_dir / f"{key}.jpg"
            try:
                img.save(img_path, format="JPEG", quality=90)
            except Exception:
                n_skip_decode += 1
                continue
            idx.write(json.dumps({
                "key": key,
                "image_path": str(img_path),
                "caption": caption,
            }) + "\n")
            idx.flush()
            have.add(key)
            pbar.update(1)
    pbar.close()
    print(f"done. {len(have)} pairs at {out_dir}")
    print(f"skipped: existing={n_skip_existing} small={n_skip_size} decode-fail={n_skip_decode}")


if __name__ == "__main__":
    main()
