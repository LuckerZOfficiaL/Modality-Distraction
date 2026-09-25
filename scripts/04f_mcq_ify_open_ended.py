"""Step 4f: convert open-ended QA (HotpotQA, TriviaQA, NaturalQuestions, etc.)
to 4-way MCQ by generating 3 plausible distractors per item with the Gemini
oracle client.

Input rows must have at least:
  {"id": ..., "question": ..., "answer": "...", optional "passage": "..."}
Output rows:
  {"id": ..., "question": ..., "passage": ...,
   "options": [...4...], "correct_index": <0-3>}

Resume-safe (skips ids already in output).

Cost guidance: Gemini Flash @ ~1¢/1k input tokens, ~250 tokens/item
  → ~$2-3 per 10k items. Cheap.

Usage:
  python scripts/04f_mcq_ify_open_ended.py \
      --input data/raw/hotpotqa.jsonl \
      --output data/raw/hotpotqa_mcq.jsonl \
      --max-workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from moground.oracle_clients import make_client


SYSTEM = (
    "You generate plausible incorrect answer options for a multiple-choice "
    "question. Given the question and the correct answer, propose exactly 3 "
    "distractors that are of the same category and style as the correct answer, "
    "but factually wrong. Distractors must be:\n"
    "  - the same type (a date if answer is a date, a person name if a person, etc.)\n"
    "  - of similar length to the correct answer\n"
    "  - distinguishable from each other and from the correct answer\n"
    "  - plausible enough that a reader without the supporting passage might "
    "    pick any of them.\n"
    "Output ONLY a JSON object: {\"distractors\": [\"d1\", \"d2\", \"d3\"]}"
)


def _strip_json(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()


def _user_prompt(question: str, answer: str, passage: str | None) -> str:
    parts = [f"Question: {question}", f"Correct answer: {answer}"]
    if passage:
        parts.append(f"Supporting passage: {passage[:1500]}")  # cap to keep tokens reasonable
    parts.append("\nProduce 3 plausible incorrect options. JSON only.")
    return "\n".join(parts)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _dedupe(correct: str, distractors: list[str]) -> list[str] | None:
    """Reject if any distractor matches the correct answer or duplicates another."""
    seen = {_normalize(correct)}
    out = []
    for d in distractors:
        n = _normalize(d)
        if not n or n in seen:
            return None
        seen.add(n)
        out.append(d.strip())
    return out if len(out) == 3 else None


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="jsonl with {id, question, answer, [passage]}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the per-item shuffling of correct/distractor positions")
    args = ap.parse_args()

    rows = [
        json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()
    ]
    done = _load_done(Path(args.output))
    todo = [r for r in rows if str(r["id"]) not in done]
    print(f"mcq-ify: {len(rows)} total, {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    client = make_client(args.backend, model=args.model, key_path=args.key_path)

    def call_one(row):
        try:
            txt = client.complete(
                system=SYSTEM,
                user=_user_prompt(row["question"], row["answer"], row.get("passage")),
                images=(),
                temperature=0.0,
                response_mime_type="application/json",
            )
            obj = json.loads(_strip_json(txt))
            distractors = obj.get("distractors") or []
            cleaned = _dedupe(row["answer"], distractors)
            if cleaned is None:
                return {"id": row["id"], "error": "distractor-dedupe-failed",
                        "raw_distractors": distractors}
            # Stable shuffle for reproducibility.
            seed_str = f"{args.seed}|{row['id']}"
            rng = random.Random(int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16))
            options = [row["answer"]] + cleaned
            order = list(range(4))
            rng.shuffle(order)
            options_shuf = [options[i] for i in order]
            correct_index = order.index(0)
            return {
                "id": str(row["id"]),
                "question": row["question"],
                "passage": row.get("passage", ""),
                "options": options_shuf,
                "correct_index": correct_index,
                "answer": row["answer"],
                "source": row.get("source", ""),
            }
        except Exception as e:
            return {"id": str(row["id"]),
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    with Path(args.output).open("a") as f, ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for i, res in enumerate(ex.map(call_one, todo), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
