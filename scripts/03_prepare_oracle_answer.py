"""Step 3 prep: chunk candidates.jsonl into work orders for one of three
oracle answering passes:

  --mode vt   image + caption  (image_path + caption_for_filter)
  --mode v    image only       (image_path)
  --mode t    caption only     (caption_for_filter)

Outputs go to data/dm/work/ans_<mode>/:
  chunk_NNN.jsonl              # one row per candidate
  chunk_NNN.result.jsonl       # agents append predictions here
  manifest.json                # chunk index + agent prompt template

Each chunk row schema (input):
  {"candidate_id": ..., "question": ..., "options": [...],
   "image_path": <only when mode in {vt,v}>,
   "caption":    <only when mode in {vt,t}>}

Each chunk row schema (output, written by agent):
  {"candidate_id": ..., "predicted_index": <0|1|2|3>}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from moground.oracle_prompts import ANSWER_INSTRUCTIONS


MODE_DESC = {
    "vt": "image + caption",
    "v": "image only (no caption)",
    "t": "caption only (no image)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--mode", choices=["vt", "v", "t"], required=True)
    ap.add_argument("--chunk-size", type=int, default=25)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    candidates_path = dm_dir / "candidates.jsonl"
    work_dir = dm_dir / "work" / f"ans_{args.mode}"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in candidates_path.read_text().splitlines() if l.strip()]

    # build per-row work payload depending on mode
    work_rows: list[dict] = []
    for r in rows:
        item: dict = {
            "candidate_id": r["candidate_id"],
            "question": r["question"],
            "options": r["options"],
        }
        if args.mode in ("vt", "v"):
            item["image_path"] = r["image_path"]
        if args.mode in ("vt", "t"):
            item["caption"] = r["caption_for_filter"]
        work_rows.append(item)

    chunks = []
    for i in range(0, len(work_rows), args.chunk_size):
        batch = work_rows[i : i + args.chunk_size]
        idx = i // args.chunk_size
        in_path = work_dir / f"chunk_{idx:03d}.jsonl"
        out_path = work_dir / f"chunk_{idx:03d}.result.jsonl"
        with in_path.open("w") as f:
            for it in batch:
                f.write(json.dumps(it) + "\n")
        out_path.touch(exist_ok=True)
        chunks.append({
            "index": idx,
            "input": str(in_path),
            "output": str(out_path),
            "n_items": len(batch),
        })

    template = (
        ANSWER_INSTRUCTIONS
        + "\n\n"
        + f"This is the {args.mode.upper()} pass: {MODE_DESC[args.mode]}.\n"
        + "\nYour task (RESUMABLE):\n"
        + "1. Read the input file at {INPUT_PATH} (one JSON object per line).\n"
        + "2. Read the output file at {OUTPUT_PATH} if it exists. Build the set of candidate_ids ALREADY PRESENT there. Skip those.\n"
        + "3. For EACH remaining input row:\n"
        + ("   - Read the image at image_path with the Read tool.\n" if args.mode in ("vt", "v") else "")
        + ("   - Treat the provided 'caption' as the only textual context.\n" if args.mode in ("vt", "t") else "")
        + (f"   - You ARE NOT given a caption in this pass. Answer purely from the image.\n" if args.mode == "v" else "")
        + (f"   - You ARE NOT given an image in this pass. Answer purely from the caption text.\n" if args.mode == "t" else "")
        + "   - Pick the single best option for the MCQ. Do not abstain.\n"
        + "   - APPEND one line to {OUTPUT_PATH} via:\n"
        + "       python3 -c 'import sys; open(sys.argv[1],\"a\").write(sys.argv[2]+\"\\n\")' {OUTPUT_PATH} '<compact-json>'\n"
        + "     where <compact-json> is exactly: {\"candidate_id\":\"<id>\",\"predicted_index\":<0|1|2|3>}\n"
        + "     Do NOT use the Write tool.\n"
        + "4. On per-row failure, append {\"candidate_id\":\"<id>\",\"error\":\"<reason>\"} and continue.\n"
        + "5. Stop only when every input candidate_id appears in {OUTPUT_PATH}.\n"
        + "6. Never rewrite or truncate {OUTPUT_PATH} \u2014 only append.\n"
        + "7. Target: {N_ITEMS} rows.\n"
    )

    manifest = {
        "mode": args.mode,
        "n_candidates": len(work_rows),
        "n_chunks": len(chunks),
        "chunks": chunks,
        "agent_prompt_template": template,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared mode={args.mode}: {len(chunks)} chunks covering {len(work_rows)} candidates -> {work_dir}")


if __name__ == "__main__":
    main()
