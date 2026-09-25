"""Fetch + materialize additional general-capability MCQ benchmarks into the same rows.jsonl schema
as data/benchmarks/{mmstar,naturalbench} (image_path, question, options, correct_index), so
scripts/78 can evaluate them. General multimodal MCQ, MMStar-like:
  mmbench    lmms-lab/MMBench (en, dev; gold answers)   ~4300 general perception+reasoning
  seedbench  lmms-lab/SEED-Bench (image subset)         subsampled to --seed-max
  scienceqa  lmms-lab/ScienceQA (ScienceQA-IMG, test)   ~2000 science reasoning w/ image

    python scripts/79_prep_more_benchmarks.py --only all
    python scripts/79_prep_more_benchmarks.py --only mmbench,scienceqa
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def _save(img, path: Path):
    if isinstance(img, list):
        img = img[0]
    if img.mode != "RGB":
        img = img.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=95)


def _write(rows, out_dir: Path):
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with (out_dir / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n"); n += 1
    print(f"  -> {out_dir}/rows.jsonl  ({n} items)")


def _isnan(v):
    return v is None or str(v).strip().lower() in ("nan", "none", "")


def _collect(pairs, out_dir: Path, build):
    """Loop (index, record) pairs with per-record fault tolerance (one bad record can't kill prep)."""
    rows, skipped, first_err = [], 0, None
    for i, r in pairs:
        try:
            row = build(i, r)
            if row is not None:
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
    if skipped:
        print(f"  skipped {skipped} bad records (first: {first_err})")
    _write(rows, out_dir)


def prep_mmbench(out_dir: Path):
    ds = load_dataset("lmms-lab/MMBench", "en", split="dev")

    def build(i, r):
        opts = [r[L] for L in ("A", "B", "C", "D") if not _isnan(r.get(L))]
        if len(opts) < 2 or _isnan(r.get("answer")):
            return None
        ci = "ABCD".index(r["answer"].strip())
        if ci >= len(opts):
            return None
        ip = out_dir / "images" / f"{i}.jpg"; _save(r["image"], ip)
        q = r["question"] if _isnan(r.get("hint")) else f"{r['hint']}\n{r['question']}"
        return {"candidate_id": f"mmbench_{i}", "image_path": str(ip.resolve()),
                "question": q, "options": opts, "correct_index": ci}
    _collect(enumerate(ds), out_dir, build)


def prep_seedbench(out_dir: Path, seed_max: int):
    ds = load_dataset("lmms-lab/SEED-Bench", split="test")
    # data_type is a scalar column -> reading it does NOT decode images (fast); only the selected
    # rows are then materialized via random access, avoiding a full ~14k-image decode scan.
    idx = [i for i, dt in enumerate(ds["data_type"]) if dt == "image"]
    random.Random(0).shuffle(idx)
    if seed_max:
        idx = idx[:seed_max]
    idx = sorted(idx)

    def build(i, r):
        opts = [r["choice_a"], r["choice_b"], r["choice_c"], r["choice_d"]]
        ci = "ABCD".index(r["answer"].strip())
        ip = out_dir / "images" / f"{i}.jpg"; _save(r["image"], ip)
        return {"candidate_id": f"seed_{i}", "image_path": str(ip.resolve()),
                "question": r["question"], "options": opts, "correct_index": ci}
    _collect(((oi, ds[oi]) for oi in idx), out_dir, build)


def prep_scienceqa(out_dir: Path):
    ds = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split="test")

    def build(i, r):
        if r.get("image") is None or len(r["choices"]) < 2:
            return None
        ci = int(r["answer"])
        if ci >= len(r["choices"]):
            return None
        ip = out_dir / "images" / f"{i}.jpg"; _save(r["image"], ip)
        return {"candidate_id": f"sqa_{i}", "image_path": str(ip.resolve()),
                "question": r["question"], "options": list(r["choices"]), "correct_index": ci}
    _collect(enumerate(ds), out_dir, build)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/benchmarks")
    ap.add_argument("--only", default="all")
    ap.add_argument("--seed-max", type=int, default=3000, help="SEED-Bench subsample cap (0=full)")
    args = ap.parse_args()
    which = {w.strip() for w in args.only.split(",")} if args.only != "all" else {"mmbench", "seedbench", "scienceqa"}
    root = Path(args.out_root)
    if "mmbench" in which:
        print("mmbench:"); prep_mmbench(root / "mmbench")
    if "seedbench" in which:
        print("seedbench:"); prep_seedbench(root / "seedbench", args.seed_max)
    if "scienceqa" in which:
        print("scienceqa:"); prep_scienceqa(root / "scienceqa")


if __name__ == "__main__":
    main()
