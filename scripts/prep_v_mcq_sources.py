"""Load V-grounded (image-required) MCQ datasets from HuggingFace and dump them
to the standardized jsonl shape that 04g_retrieve_text_qa_candidates.py with
--mode v can consume: {id, question, options[4], correct_index, image_path}.

Supported sources:
  - aokvqa      : A-OKVQA (~25k items, 4-way MCQ, world-knowledge + image)
  - scienceqa   : ScienceQA (~21k items, ~half with images, 2-5 way MCQ — filter to 4-way + image-required)

Images are saved to data/raw/v_mcq/<source>/<id>.<ext> (cache-aware: skip if file
already exists). The output jsonl row points to that on-disk path so that the
04g retrieval and downstream Qwen calls can read the image directly.

Usage:
  python scripts/prep_v_mcq_sources.py --source aokvqa     --out data/raw/aokvqa.jsonl
  python scripts/prep_v_mcq_sources.py --source scienceqa  --out data/raw/scienceqa.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _save_image(img, out_path: Path) -> bool:
    """Save a PIL.Image to out_path. Returns True if a new file was written."""
    if out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out_path, format="JPEG", quality=92)
    return True


def load_aokvqa(image_root: Path):
    from datasets import load_dataset
    rows = []
    n_new = 0
    for split in ("train", "validation"):
        try:
            ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split)
        except Exception as e:
            print(f"  ! HF load failed for HuggingFaceM4/A-OKVQA[{split}]: {e}")
            continue
        for r in ds:
            choices = r.get("choices") or []
            if len(choices) != 4:
                continue
            ci = r.get("correct_choice_idx")
            if ci is None or not (0 <= int(ci) < 4):
                continue
            qid = r.get("question_id") or r.get("id")
            img = r.get("image")
            if img is None or qid is None:
                continue
            out_path = image_root / "aokvqa" / f"{qid}.jpg"
            n_new += int(_save_image(img, out_path))
            rows.append({
                "id": f"aokvqa_{qid}",
                "question": r["question"],
                "options": list(choices),
                "correct_index": int(ci),
                "image_path": str(out_path),
                "source": "aokvqa",
                "split_orig": split,
            })
    print(f"  saved {n_new} new images (existing files reused)")
    return rows


def load_scienceqa(image_root: Path):
    from datasets import load_dataset
    rows = []
    n_new = 0
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset("derek-thomas/ScienceQA", split=split)
        except Exception as e:
            print(f"  ! HF load failed for derek-thomas/ScienceQA[{split}]: {e}")
            continue
        for i, r in enumerate(ds):
            img = r.get("image")
            choices = r.get("choices") or []
            if img is None or len(choices) != 4:
                continue
            ci = r.get("answer")
            if ci is None or not (0 <= int(ci) < 4):
                continue
            qid = f"{split}_{i}"
            out_path = image_root / "scienceqa" / f"{qid}.jpg"
            n_new += int(_save_image(img, out_path))
            rows.append({
                "id": f"scienceqa_{qid}",
                "question": r["question"],
                "options": list(choices),
                "correct_index": int(ci),
                "image_path": str(out_path),
                "source": "scienceqa",
                "split_orig": split,
            })
    print(f"  saved {n_new} new images (existing files reused)")
    return rows


LOADERS = {
    "aokvqa":     load_aokvqa,
    "scienceqa":  load_scienceqa,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(LOADERS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default="data/raw/v_mcq",
                    help="root dir for cached images")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional row cap for quick probe")
    args = ap.parse_args()

    print(f"loading {args.source} from HF (will save images to {args.image_root}/{args.source}/)")
    rows = LOADERS[args.source](Path(args.image_root))
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
