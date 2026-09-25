"""Step 1 (VisText variant): sample (chart-image, L2/L3 caption) seeds for D_M.

VisText = Tang et al. 2023, ACL. Chart-caption corpus over Statista charts.
We use the L2/L3 caption (human-written summary of trends/statistics) as the
seed; L1 is auto-generated and would leak chart-construction facts into the
T-only condition.

Acquisition (manual, one-time):
  cd data/raw/vistext/
  wget https://vis.csail.mit.edu/vistext/tabular.zip && unzip tabular.zip
  wget https://vis.csail.mit.edu/vistext/images.zip  && unzip images.zip
This script reads:
  data/raw/vistext/data_train.json
  data/raw/vistext/images/<img_id>.png

Outputs:
  data/dm/vistext/seeds.jsonl

Row schema:
  {"seed_id": "vistext_<img_id>", "image_path": <abs>, "caption": <L2L3>,
   "source": "vistext", "image_id": <img_id>}

Multiple captions per image exist in VisText (caption_id like "1_01", "1_02").
We deduplicate by img_id and keep the first caption (lowest caption_id),
giving ~7K unique image seeds from data_train.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_vistext.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    dm_dir.mkdir(parents=True, exist_ok=True)

    num_seeds = int(cfg["seeds"]["num_seeds"])
    seed = int(cfg["seed"])

    raw = data_root / "raw" / "vistext"
    train_json = raw / "data_train.json"
    images_dir = raw / "images"
    if not train_json.exists():
        sys.exit(f"missing {train_json} - see module docstring for download steps")
    if not images_dir.exists():
        sys.exit(f"missing {images_dir} - see module docstring for download steps")

    records = json.loads(train_json.read_text())
    print(f"loaded {len(records)} VisText train records", file=sys.stderr)

    by_img: dict[str, dict] = {}
    for r in records:
        img_id = str(r["img_id"])
        cap_id = str(r.get("caption_id", ""))
        if img_id in by_img and cap_id >= str(by_img[img_id].get("caption_id", "")):
            continue
        by_img[img_id] = r
    print(f"deduped to {len(by_img)} unique images", file=sys.stderr)

    seeds = list(by_img.values())
    rng = random.Random(seed)
    rng.shuffle(seeds)

    out = dm_dir / "seeds.jsonl"
    n_written = 0
    n_skipped_no_image = 0
    n_skipped_no_caption = 0
    with out.open("w") as f:
        for r in tqdm(seeds, desc="seeds"):
            if n_written >= num_seeds:
                break
            img_id = str(r["img_id"])
            img_path = images_dir / f"{img_id}.png"
            if not img_path.exists():
                n_skipped_no_image += 1
                continue
            cap = (r.get("caption_L2L3") or "").strip()
            if not cap:
                n_skipped_no_caption += 1
                continue
            row = {
                "seed_id": f"vistext_{img_id}",
                "image_path": str(img_path.resolve()),
                "caption": cap,
                "source": "vistext",
                "image_id": img_id,
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1

    print(f"wrote {n_written} seeds -> {out}")
    if n_skipped_no_image:
        print(f"skipped {n_skipped_no_image} (image png missing)")
    if n_skipped_no_caption:
        print(f"skipped {n_skipped_no_caption} (no L2L3 caption)")


if __name__ == "__main__":
    main()
