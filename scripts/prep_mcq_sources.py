"""Load MCQ-native text QA datasets from HuggingFace `datasets` and dump them
to the standardized jsonl shape that 04g_retrieve_text_qa_candidates.py
expects: {id, question, options[4], correct_index, passage}.

Supported sources:
  - race-high        : RACE high-school reading comprehension (28k items, has passage)
  - openbookqa       : OpenBookQA core (5k items, has supporting fact)
  - arc-challenge    : ARC Challenge science MCQ (1.1k items, no passage)
  - mmlu-hard        : MMLU restricted to hard subjects (~3k items, no passage)
  - commonsenseqa    : CommonsenseQA (12k items, no passage; 5-way native — drops the
                        least-plausible distractor to fit 4-way)

Outputs all splits (train + val + test) concatenated to a single jsonl, since
construction-time partitioning is handled by register_text_qa_dataset.py.

Usage:
  python scripts/prep_mcq_sources.py --source race-high   --out data/raw/race_high.jsonl
  python scripts/prep_mcq_sources.py --source openbookqa  --out data/raw/openbookqa.jsonl
  python scripts/prep_mcq_sources.py --source arc-challenge --out data/raw/arc_challenge.jsonl
  python scripts/prep_mcq_sources.py --source mmlu-hard   --out data/raw/mmlu_hard.jsonl
  python scripts/prep_mcq_sources.py --source commonsenseqa --out data/raw/commonsenseqa.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Hand-picked hard MMLU subjects: ones where strong-model accuracy sits ~30-55%.
# Source: MMLU eval reports across multiple papers; conservative selection.
MMLU_HARD_SUBJECTS = [
    "professional_law",
    "abstract_algebra",
    "college_chemistry",
    "college_physics",
    "college_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_european_history",
    "high_school_world_history",
    "high_school_us_history",
    "high_school_chemistry",
    "high_school_physics",
    "high_school_mathematics",
    "machine_learning",
    "philosophy",
    "professional_medicine",
    "world_religions",
]


def load_race_high():
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", "high")
    rows = []
    letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for split in ("train", "validation", "test"):
        for r in ds[split]:
            if len(r["options"]) != 4:
                continue
            ci = letter2idx.get(r["answer"])
            if ci is None:
                continue
            rows.append({
                "id": f"race-h_{r['example_id']}_{r['question'][:24].replace(' ', '_')}",
                "question": r["question"],
                "options": r["options"],
                "correct_index": ci,
                "passage": r["article"],
                "source": "race-high",
                "split_orig": split,
            })
    return rows


def load_openbookqa():
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "additional")
    rows = []
    letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for split in ("train", "validation", "test"):
        for r in ds[split]:
            choices_text = r["choices"]["text"]
            choices_label = r["choices"]["label"]
            if len(choices_text) != 4:
                continue
            ci = letter2idx.get(r["answerKey"])
            if ci is None:
                continue
            # use the gold supporting fact as the passage
            passage = r.get("fact1") or ""
            rows.append({
                "id": f"openbookqa_{r['id']}",
                "question": r["question_stem"],
                "options": choices_text,
                "correct_index": ci,
                "passage": passage,
                "source": "openbookqa",
                "split_orig": split,
            })
    return rows


def load_arc_challenge():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    rows = []
    for split in ("train", "validation", "test"):
        for r in ds[split]:
            texts = r["choices"]["text"]
            labels = r["choices"]["label"]
            if len(texts) != 4:
                continue  # skip 3- or 5-option items
            label2idx = {lab: i for i, lab in enumerate(labels)}
            ci = label2idx.get(r["answerKey"])
            if ci is None:
                continue
            rows.append({
                "id": f"arc-ch_{r['id']}",
                "question": r["question"],
                "options": texts,
                "correct_index": ci,
                "passage": "",  # no native passage in ARC; image will be pure distractor
                "source": "arc-challenge",
                "split_orig": split,
            })
    return rows


def load_mmlu_hard():
    from datasets import load_dataset
    rows = []
    for subject in MMLU_HARD_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject)
        except Exception as e:
            print(f"  ! could not load mmlu/{subject}: {e}")
            continue
        for split in ("test", "validation", "dev"):
            if split not in ds:
                continue
            for i, r in enumerate(ds[split]):
                if len(r["choices"]) != 4:
                    continue
                rows.append({
                    "id": f"mmlu_{subject}_{split}_{i}",
                    "question": r["question"],
                    "options": list(r["choices"]),
                    "correct_index": int(r["answer"]),
                    "passage": "",
                    "source": f"mmlu-{subject}",
                    "split_orig": split,
                })
    return rows


def load_commonsenseqa():
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa")
    rows = []
    # CSQA is 5-way (labels A-E); drop the least-plausible-looking distractor
    # heuristically: drop the option whose text length is most outlier vs others.
    letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for split in ("train", "validation"):
        for r in ds[split]:
            texts = r["choices"]["text"]
            if len(texts) != 5:
                continue
            ci = letter2idx.get(r["answerKey"])
            if ci is None:
                continue
            # drop the option index whose text length is most outlier (excluding the
            # correct one). Simple, deterministic, no LLM call.
            wrong_idxs = [i for i in range(5) if i != ci]
            target_len = len(texts[ci])
            wrong_idxs.sort(key=lambda i: abs(len(texts[i]) - target_len), reverse=True)
            drop = wrong_idxs[0]
            kept = [t for i, t in enumerate(texts) if i != drop]
            new_ci = ci if ci < drop else ci - 1
            rows.append({
                "id": f"csqa_{r['id']}",
                "question": r["question"],
                "options": kept,
                "correct_index": new_ci,
                "passage": "",
                "source": "commonsenseqa",
                "split_orig": split,
            })
    return rows


LOADERS = {
    "race-high":      load_race_high,
    "openbookqa":     load_openbookqa,
    "arc-challenge":  load_arc_challenge,
    "mmlu-hard":      load_mmlu_hard,
    "commonsenseqa":  load_commonsenseqa,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(LOADERS.keys()))
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--limit", type=int, default=None, help="optional row cap for quick probe")
    args = ap.parse_args()

    rows = LOADERS[args.source]()
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
