"""Step 4: Qwen2.5-VL-3B V-only majority-vote pass over qwen_candidates.jsonl.

Runs N samples per candidate at non-zero temperature, takes the modal
predicted_index. Drops the per-candidate noise that a single-shot V-only
inference would carry into the keep rule.

Resumable: skips candidate_ids already present in the output.

Usage (RUN ON GPU):
  python scripts/04_qwen_v_only_majority.py \
      --input  data/dm/gemini/batch2/qwen_candidates.jsonl \
      --output data/dm/gemini/batch2/qwen_v_only.jsonl \
      --num-samples 5 --temperature 0.7
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
    "You are answering a multiple-choice question about an image. "
    "Look at the image, choose the single best option (0, 1, 2, or 3), "
    "and reply with ONLY valid JSON: {\"predicted_index\": <0|1|2|3>}. "
    "Do not abstain; if uncertain, pick the most likely option."
)


def _format_user(c: dict) -> str:
    parts = [f"question: {c['question']}", "options:"]
    for i, opt in enumerate(c["options"]):
        parts.append(f"  {i}. {opt}")
    parts.append("Answer with a JSON object only.")
    return "\n".join(parts)


def _parse_index(text: str) -> int | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        obj = json.loads(t.strip())
        pi = int(obj.get("predicted_index"))
        return pi if pi in (0, 1, 2, 3) else None
    except Exception:
        # last-ditch: scan for a bare 0-3 digit near the start.
        for ch in t.strip():
            if ch in "0123":
                return int(ch)
        return None


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                if "majority" in r and "candidate_id" in r:
                    done.add(r["candidate_id"])
            except json.JSONDecodeError:
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="qwen_candidates.jsonl from cascade")
    ap.add_argument("--output", required=True, help="qwen_v_only.jsonl to write")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Non-zero so the N samples diversify; 0.7 is a sane default.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        json.loads(l) for l in in_path.read_text().splitlines() if l.strip()
    ]
    done = _load_done(out_path)
    todo = [c for c in candidates if c["candidate_id"] not in done]
    print(f"qwen-v-only: {len(done)}/{len(candidates)} done; {len(todo)} to go "
          f"(N={args.num_samples} samples each, temp={args.temperature})")
    if not todo:
        print("nothing to do.")
        return

    # Lazy import so this script imports cleanly without GPU deps when not running.
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"loading {args.model} on {args.device}…")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    with out_path.open("a") as f:
        for ci, c in enumerate(todo, 1):
            try:
                image = Image.open(c["image_path"]).convert("RGB")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  _format_user(c)},
                    ]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)

                preds: list[int | None] = []
                for _ in range(args.num_samples):
                    with torch.inference_mode():
                        out = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=0.95,
                            max_new_tokens=args.max_new_tokens,
                        )
                    gen = out[:, inputs["input_ids"].shape[1]:]
                    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
                    preds.append(_parse_index(decoded))

                valid = [p for p in preds if p is not None]
                if not valid:
                    row = {"candidate_id": c["candidate_id"],
                           "predictions": preds, "majority": None,
                           "error": "no valid predictions"}
                else:
                    majority, _ = Counter(valid).most_common(1)[0]
                    row = {"candidate_id": c["candidate_id"],
                           "predictions": preds, "majority": int(majority),
                           "n_valid": len(valid)}
                f.write(json.dumps(row) + "\n"); f.flush()
                if ci % 25 == 0 or ci == len(todo):
                    print(f"  {ci}/{len(todo)}")
            except Exception as e:
                row = {"candidate_id": c["candidate_id"],
                       "predictions": [], "majority": None,
                       "error": f"{type(e).__name__}: {str(e)[:200]}"}
                f.write(json.dumps(row) + "\n"); f.flush()

    print(f"done -> {out_path}")
    print(f"\nNEXT: python scripts/05_apply_qwen_keep_rule.py --out-dir {in_path.parent}")


if __name__ == "__main__":
    main()
