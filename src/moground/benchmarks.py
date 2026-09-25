"""External MCQ benchmark adapters for the canonical steering harness.

Each loader materializes a benchmark into the canonical row schema consumed by
``moground.steering.build_messages``:

    {
      "candidate_id": str,            # globally unique
      "image_path":   str,            # path to a materialized RGB JPEG
      "question":     str,
      "options":      list[str],      # 2..6 options
      "correct_index": int,
      "label":        "vision",       # all external items are image-required
      "source":       "mmstar"|"vilp"|"naturalbench",
      ...source-specific stratification tags...
    }

There is NO ``caption_for_filter`` key -- external benchmarks have no separable
text modality, so ``build_messages`` drops the caption block. Structurally this
is the V-only prompt form; the steering question is whether the offline
``v_V_causal`` direction (and, later, the L31 SAE) still helps when the model's
only text is the question + options it can answer from language priors.

Images are saved once to ``<out_dir>/images/`` and referenced by path so the
rest of the pipeline (path-based ``build_messages``) is unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


def _save_image(img: "Image.Image", path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=95)


# --------------------------------------------------------------------------
# MMStar  (Lin-Chen/MMStar, split "val", 1500 items)
#   options are embedded in the question string:
#     "<stem>\nOptions: A: <opt>, B: <opt>, C: <opt>, D: <opt>"
#   `answer` is the gold letter.
# --------------------------------------------------------------------------

# inline comma format: "A: opt, B: opt, ..."
_MMSTAR_INLINE_RE = re.compile(r"([A-F]):\s*(.*?)(?=(?:,\s*[A-F]:)|$)", re.S)
# line-per-option format: "(A) opt" / "A. opt" / "A: opt"
_MMSTAR_LINE_RE = re.compile(r"\(?([A-F])\)?\s*[:.)]?\s*(.+)$")


def parse_mmstar_question(question: str) -> tuple[str, list[str], list[str]]:
    """Return (stem, options, letters). Handles MMStar's two prompt formats:

      (1) "<stem>\\nOptions: A: <o>, B: <o>, ..."           (inline, comma-sep)
      (2) "Hint: ...\\nQuestion: <stem>\\nChoices:\\n(A) <o>\\n(B) <o> ..." (per-line)

    Raises ValueError if unparseable."""
    q = re.sub(r"^\s*Hint:[^\n]*\n", "", question)   # drop instruction scaffold
    q = re.sub(r"^\s*Question:\s*", "", q)
    parts = re.split(r"\n\s*(?:Options?|Choices?)\s*:\s*", q, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("no Options/Choices block")
    stem, optstr = parts[0].strip(), parts[1].strip()

    letters: list[str] = []
    options: list[str] = []
    if "\n" in optstr:  # line-per-option
        for line in optstr.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = _MMSTAR_LINE_RE.match(line)
            if m:
                letters.append(m.group(1))
                options.append(m.group(2).strip())
    else:  # inline comma-separated
        for m in _MMSTAR_INLINE_RE.findall(optstr):
            letters.append(m[0])
            options.append(m[1].strip().rstrip(",").strip())

    if len(options) < 2:
        raise ValueError(f"parsed <2 options: {options}")
    return stem, options, letters


def prep_mmstar(hf_iter, out_dir: Path) -> int:
    """Materialize MMStar rows + images. `hf_iter` yields MMStar records.
    Returns the number of rows written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    rows_path = out_dir / "rows.jsonl"
    n, skipped = 0, 0
    with rows_path.open("w") as f:
        for r in hf_iter:
            idx = str(r["index"])
            try:
                stem, options, letters = parse_mmstar_question(r["question"])
                correct_index = letters.index(r["answer"].strip())
            except (ValueError, KeyError):
                skipped += 1
                continue
            img_path = img_dir / f"{idx}.jpg"
            _save_image(r["image"], img_path)
            try:
                meta = json.loads(r["meta_info"].replace("'", '"'))
            except Exception:
                meta = {}
            f.write(json.dumps({
                "candidate_id": f"mmstar_{idx}",
                "image_path": str(img_path.resolve()),
                "question": stem,
                "options": options,
                "correct_index": correct_index,
                "label": "vision",
                "source": "mmstar",
                "category": r.get("category"),
                "l2_category": r.get("l2_category"),
                "orig_source": meta.get("source"),
            }) + "\n")
            n += 1
    if skipped:
        print(f"[mmstar] skipped {skipped} unparseable items")
    print(f"[mmstar] wrote {n} rows -> {rows_path}")
    return n


# --------------------------------------------------------------------------
# ViLP  (ViLP/ViLP, split "train", ~300 questions)
#   Each record: question + (image1,answer1),(image2,answer2),(image3,answer3).
#   answer1 is the language-prior ("typical") answer; image1 is prior-consistent.
#   We make each (question, image_k) a 3-way MCQ with options {answer1,2,3}
#   (deterministically shuffled), so the prior answer is always a distractor.
# --------------------------------------------------------------------------

def prep_vilp(hf_iter, out_dir: Path, seed: int = 0) -> int:
    import random
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    rows_path = out_dir / "rows.jsonl"
    n, skipped = 0, 0
    with rows_path.open("w") as f:
        for qi, r in enumerate(hf_iter):
            answers = [str(r["answer1"]).strip(), str(r["answer2"]).strip(),
                       str(r["answer3"]).strip()]
            if len(set(answers)) != 3:
                skipped += 1
                continue
            rng = random.Random(f"{seed}-{qi}")
            order = [0, 1, 2]
            rng.shuffle(order)
            options = [answers[j] for j in order]
            for k in (1, 2, 3):  # image variant 1=prior-consistent, 2/3=prior-violating
                img = r[f"image{k}"]
                gold = answers[k - 1]
                img_path = img_dir / f"{qi}_v{k}.jpg"
                _save_image(img, img_path)
                f.write(json.dumps({
                    "candidate_id": f"vilp_{qi}_v{k}",
                    "image_path": str(img_path.resolve()),
                    "question": r["question"].strip(),
                    "options": options,
                    "correct_index": options.index(gold),
                    "label": "vision",
                    "source": "vilp",
                    "variant": k,
                    "prior_consistent": (k == 1),
                    "prior_answer": answers[0],
                    "question_group": f"vilp_{qi}",
                }) + "\n")
                n += 1
    if skipped:
        print(f"[vilp] skipped {skipped} questions with non-distinct answers")
    print(f"[vilp] wrote {n} rows -> {rows_path}")
    return n


# --------------------------------------------------------------------------
# NaturalBench  (BaiqiL/NaturalBench, yes/no balanced groups)
#   Each group: 2 images x 2 questions -> 4 (image,question) yes/no items with
#   alternating answers, so a language-prior model that ignores the image scores
#   at chance. We materialize a fixed-seed subsample preserving whole groups so
#   both per-item accuracy and official group-accuracy are computable.
# --------------------------------------------------------------------------

def prep_naturalbench(hf_iter, out_dir: Path) -> int:
    """`hf_iter` yields already-selected NaturalBench group records (caller does
    the group subsampling). Only yes_no items are materialized; other types are
    skipped and counted."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    rows_path = out_dir / "rows.jsonl"
    n, skipped = 0, 0
    with rows_path.open("w") as f:
        for r in hf_iter:
            if str(r.get("Question_Type", "")).lower() != "yes_no":
                skipped += 1
                continue
            gid = str(r["Index"])
            img_paths = {}
            for slot in (0, 1):
                p = img_dir / f"{gid}_img{slot}.jpg"
                _save_image(r[f"Image_{slot}"], p)
                img_paths[slot] = str(p.resolve())
            for slot in (0, 1):
                for qi in (0, 1):
                    ans = str(r[f"Image_{slot}_Question_{qi}"]).strip()
                    correct_index = 0 if ans.lower() == "yes" else 1
                    f.write(json.dumps({
                        "candidate_id": f"nb_{gid}_i{slot}_q{qi}",
                        "image_path": img_paths[slot],
                        "question": r[f"Question_{qi}"].strip(),
                        "options": ["Yes", "No"],
                        "correct_index": correct_index,
                        "label": "vision",
                        "source": "naturalbench",
                        "group": f"nb_{gid}",
                        "orig_source": r.get("Source"),
                    }) + "\n")
                    n += 1
    if skipped:
        print(f"[naturalbench] skipped {skipped} non-yes_no items")
    print(f"[naturalbench] wrote {n} rows -> {rows_path}")
    return n
