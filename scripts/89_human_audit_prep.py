"""P1: build the certification-audit sample + annotator sheets.

Stratified sample of D_M items (default 30 vision + 30 text per domain = 240) for
(a) the human validation study, (b) an independent second-oracle re-judgment (Gemini or
Sonnet). Produces, under data/human_audit/:
  sample.jsonl            full item records (id, image_path, caption, question, options, label)
  sheet_T.csv             text-only pass:  caption + question + options   (no image)
  sheet_V.csv             vision-only pass: image + question + options    (no caption)
  sheet_VT.csv            both-modalities pass + degenerate-subtype tags
  INSTRUCTIONS.md         protocol (3 passes, ordering, tag definitions)
Per-annotator protocol: do the T pass for ALL items first, then V, then VT (so seeing the
image never contaminates a text-only judgment); different annotators may take different
condition orders to balance carryover.

    python scripts/89_human_audit_prep.py --per-cell 30
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMS = ["dci", "vistext", "semart", "roco"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in (ROOT / "data/dm/multidomain_v1/dm_all.jsonl").read_text().splitlines()
            if l.strip()]
    out = ROOT / "data/human_audit"; out.mkdir(exist_ok=True)
    sample = []
    for d in DOMS:
        for lab in ("vision", "text"):
            pool = [r for r in rows if r["source"] == d and r["label"] == lab]
            sample += rng.sample(pool, min(args.per_cell, len(pool)))
    rng.shuffle(sample)
    with (out / "sample.jsonl").open("w") as f:
        for r in sample:
            f.write(json.dumps({k: r[k] for k in
                    ("candidate_id", "image_path", "caption_for_filter", "question",
                     "options", "correct_index", "label", "source")}) + "\n")

    def sheet(name, fields, extra=()):
        with (out / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields + [f"opt_{c}" for c in "ABCD"] + list(extra) + ["answer (A-D or SKIP)"])
            for i, r in enumerate(sample):
                base = {"idx": i, "id": r["candidate_id"], "question": r["question"],
                        "caption": r["caption_for_filter"], "image": r["image_path"]}
                w.writerow([base[k.split(":")[0]] for k in fields] + r["options"][:4]
                           + [""] * len(extra) + [""])

    sheet("sheet_T.csv", ["idx", "id", "caption", "question"])
    sheet("sheet_V.csv", ["idx", "id", "image", "question"])
    sheet("sheet_VT.csv", ["idx", "id", "image", "caption", "question"],
          extra=("tag_burned_in_text (y/blank)", "tag_annotation_reading (y/blank)",
                 "tag_near_tie_options (y/blank)", "tag_ill_posed (y/blank)"))

    (out / "INSTRUCTIONS.md").write_text("""# D_M certification audit — annotator protocol

240 items, 3 passes. **Do the full T pass first, then V, then VT** (never look at an image
before finishing that item's T judgment). Each pass: pick the single best option (A-D), or
SKIP if genuinely unanswerable from the information shown.

- **T pass (sheet_T.csv):** caption + question only. No image.
- **V pass (sheet_V.csv):** image + question only. Ignore any prior memory of the caption.
- **VT pass (sheet_VT.csv):** image + caption + question. Also tag (leave blank if no):
  - `burned_in_text`: the answer is read from text/numbers printed inside the image (OCR).
  - `annotation_reading`: the answer is read off an arrow/marker/legend annotation.
  - `near_tie_options`: two or more options are so close the distinction feels arbitrary.
  - `ill_posed`: question ambiguous / multiple defensible answers / no correct option.

What we compute: human V/T/VT accuracy per grounding label; agreement with the oracle
certificate (vision-grounded items should be answerable in V and VT but not T; symmetric for
text-grounded); prevalence of degenerate subtypes; law robustness excluding tagged items.
""")
    n = len(sample)
    print(f"sample n={n} -> {out} (sheets T/V/VT + INSTRUCTIONS.md)")


if __name__ == "__main__":
    main()
