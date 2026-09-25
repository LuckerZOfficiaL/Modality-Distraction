"""Step 1: sample (image, caption) seeds for D_M from DCI (Densely Captioned Images).

DCI = Urbanek et al. 2023, "A Picture is Worth More Than 77 Text Tokens".
~7,805 SA-1B images densely annotated. We use `extra_caption` (long
descriptive paragraph) as the seed — it's dense enough that appending a
fabricated fact for text-grounded augmentation will not be a length /
sentence-count tell vs the unaugmented vision-grounded version.

Acquisition:
  - Annotations: auto-download dci.tar.gz from fbaipublicfiles (public).
  - Images: SA-1B shard sa_000138.tar must be downloaded MANUALLY from Meta
    (accept SA-1B license at https://ai.meta.com/datasets/segment-anything-downloads/).
    Place sa_000138.tar at: data/raw/dci/sa_000138.tar
    The script will extract it on first run.

Outputs:
  data/raw/dci/annotations/   (jsons, one per image)
  data/raw/dci/photos/        (extracted SA-1B images)
  data/dm/seeds.jsonl

Row schema:
  {"seed_id": "dci_<image_stem>", "image_path": <abs>, "caption": <str>,
   "source": "dci", "image_id": <stem>}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml
from tqdm import tqdm

DCI_ANN_URL = "https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz"
SA_SHARD_NAME = "sa_000138.tar"


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dst}", file=sys.stderr)
    last = [0]

    def hook(n, bs, total):
        done = n * bs
        pct = 100 * done / max(total, 1)
        if pct - last[0] >= 5:
            print(f"  {pct:5.1f}%", file=sys.stderr)
            last[0] = pct

    urlretrieve(url, dst, reporthook=hook)


def _ensure_dci(data_root: Path) -> tuple[Path, Path]:
    raw = data_root / "raw" / "dci"
    raw.mkdir(parents=True, exist_ok=True)

    ann_tgz = raw / "dci.tar.gz"
    ann_dir = raw / "annotations"
    if not ann_dir.exists():
        _download(DCI_ANN_URL, ann_tgz)
        print("extracting DCI annotations", file=sys.stderr)
        with tarfile.open(ann_tgz) as t:
            t.extractall(raw)
        # the tarball extracts into a folder; normalize to raw/annotations
        candidates = [p for p in raw.iterdir() if p.is_dir() and p.name != "photos" and p.name != "annotations"]
        for c in candidates:
            inner_anns = list(c.rglob("*.json"))
            if inner_anns:
                ann_dir.mkdir(exist_ok=True)
                for j in inner_anns:
                    j.rename(ann_dir / j.name)
                break

    photos_dir = raw / "photos"
    sa_tar = raw / SA_SHARD_NAME
    if not photos_dir.exists():
        if not sa_tar.exists():
            print(f"\nERROR: {sa_tar} not found.", file=sys.stderr)
            print("Manually download sa_000138.tar from Meta SA-1B after accepting the license:", file=sys.stderr)
            print("  https://ai.meta.com/datasets/segment-anything-downloads/", file=sys.stderr)
            print(f"and place at: {sa_tar}", file=sys.stderr)
            sys.exit(1)
        photos_dir.mkdir()
        print(f"extracting {sa_tar} -> {photos_dir}", file=sys.stderr)
        with tarfile.open(sa_tar) as t:
            t.extractall(photos_dir)

    return ann_dir, photos_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--caption-field", default="extra_caption",
                    choices=["extra_caption", "short_caption", "summary_base"],
                    help="which DCI caption to use as the seed")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    dm_images = Path(cfg["paths"]["dm_images"])
    dm_dir.mkdir(parents=True, exist_ok=True)
    dm_images.mkdir(parents=True, exist_ok=True)

    num_seeds = int(cfg["seeds"]["num_seeds"])
    seed = int(cfg["seed"])

    ann_dir, photos_dir = _ensure_dci(data_root)

    link = dm_images / "dci"
    if not link.exists():
        link.symlink_to(photos_dir)

    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"found {len(ann_files)} DCI annotations", file=sys.stderr)

    rng = random.Random(seed)
    rng.shuffle(ann_files)

    out = dm_dir / "seeds.jsonl"
    n_written = 0
    n_skipped_no_image = 0
    n_skipped_no_caption = 0
    with out.open("w") as f:
        for ann_path in tqdm(ann_files, desc="seeds"):
            if n_written >= num_seeds:
                break
            try:
                ann = json.loads(ann_path.read_text())
            except Exception:
                continue
            img_rel = ann.get("image", "")
            img_path = photos_dir / Path(img_rel).name
            if not img_path.exists():
                n_skipped_no_image += 1
                continue

            if args.caption_field == "extra_caption":
                cap = ann.get("extra_caption", "")
            elif args.caption_field == "short_caption":
                cap = ann.get("short_caption", "")
            else:
                summ = ann.get("summaries", {}).get("base", [])
                cap = summ[0] if summ else ""
            cap = (cap or "").strip()
            if not cap:
                n_skipped_no_caption += 1
                continue

            stem = Path(img_rel).stem or ann_path.stem
            row = {
                "seed_id": f"dci_{stem}",
                "image_path": str(img_path.resolve()),
                "caption": cap,
                "source": "dci",
                "image_id": stem,
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1

    print(f"wrote {n_written} seeds -> {out}")
    if n_skipped_no_image:
        print(f"skipped {n_skipped_no_image} (image missing in shard)")
    if n_skipped_no_caption:
        print(f"skipped {n_skipped_no_caption} (no caption in field {args.caption_field})")


if __name__ == "__main__":
    main()
