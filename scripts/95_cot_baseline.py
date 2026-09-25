"""Chain-of-thought prompting baseline for the mitigation comparison (S7).

Unlike scripts/61 (single forward pass, letter-logit argmax), CoT requires GENERATION:
the model is instructed to reason step by step and finish with 'Answer: X'. Same three
input conditions (VT / V / T), same pools, same jsonl schema as t8 plus the raw
generations and a parse flag, so the standard vd() scoring applies unchanged
(v-distraction = P(VT wrong | V correct), conditioned on the CoT run's own V pass).

Prompt = the family's standard eval prompt with ONLY the final instruction line swapped
for the CoT instruction, so the comparison isolates the reasoning directive.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/95_cot_baseline.py --model qwen3b --pool both
  # smoke: --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MAXPX = 1024 * 1024

REGISTRY = {  # tag -> (family, model_id, qwen_cls)
    "qwen3b":  ("qwen", "Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl"),
    "qwen7b":  ("qwen", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
    "qwen2b":  ("qwen", "Qwen/Qwen2-VL-2B-Instruct",   "qwen2_vl"),
    "next":    ("hf",   "llava-hf/llama3-llava-next-8b-hf", None),
    "mistral": ("hf",   "llava-hf/llava-v1.6-mistral-7b-hf", None),
    "internvl": ("hf",  "OpenGVLab/InternVL3-8B-hf", None),
    "llavaov": ("hf",   "llava-hf/llava-onevision-qwen2-7b-ov-hf", None),
    "llava15": ("hf",   "llava-hf/llava-1.5-7b-hf", None),
    "qwen3vl30b": ("qwen", "Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen3_vl_moe"),  # MoE: 30B total / 3B active
    "internvl14b": ("hf", "OpenGVLab/InternVL3-14B-hf", None),
    "mistral24b": ("hf", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", None),
}

COT_INSTR = ("Let's think step by step. First reason briefly about the question and the "
             "available evidence, then conclude your reply with a final line in the exact "
             "format 'Answer: X' where X is one of A, B, C, or D.")

LETTERS = "ABCDEF"


def cot_instr(n: int = 4) -> str:
    """CoT instruction for an n-option item. n=4 reproduces COT_INSTR byte-for-byte."""
    ls = LETTERS[:n]
    tail = ", ".join(ls[:-1]) + f", or {ls[-1]}" if n > 2 else f"{ls[0]} or {ls[1]}"
    return COT_INSTR.replace("A, B, C, or D", tail)

COND = [("vt", True, True), ("v", True, False), ("t", False, True)]  # (name, use_img, use_cap)


def parse_letter(text: str, n: int = 4) -> int:
    """Extract the final answer letter; -1 if unparseable (scored as wrong)."""
    hi = LETTERS[n - 1]
    hits = re.findall(rf"answer[:\s\*]*\(?([A-{hi}])\)?", text, flags=re.IGNORECASE)
    if hits:
        return ord(hits[-1].upper()) - ord("A")
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.findall(rf"\b([A-{hi}])\b", tail)
    if m:
        return ord(m[-1].upper()) - ord("A")
    return -1


def load_rows(pool: str, cfg: dict, split: str) -> list[dict]:
    if pool == "assembled":
        p = Path(cfg["paths"]["data_root"]) / "dm/merged_aokvqa_racehigh/dm_all.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    dm = Path(cfg["paths"]["dm_dir"]); rows = []
    splits = {"test": ("test",), "val": ("val",), "valtest": ("val", "test"),
              "all": ("train", "val", "test")}[split]
    for sp in splits:
        f = dm / "splits" / f"{sp}.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def make_qwen(model_id, cls_tag, device):
    from qwen_vl_utils import process_vision_info
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen2_vl": Qwen2VLForConditionalGeneration,
           "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[cls_tag]
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    def build(row, ui, uc):
        opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, row["options"]))
        content = []
        if uc:
            content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
        if ui:
            content.append({"type": "image", "image": row["image_path"], "max_pixels": MAXPX})
        content.append({"type": "text", "text": f"Question: {row['question']}\n{opts}\n\n{cot_instr(len(row['options']))}"})
        return [{"role": "user", "content": content}]

    def gen(row, ui, uc, max_new):
        msgs = build(row, ui, uc)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        if imgs is not None and len(imgs) == 0:
            imgs = None
        inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(device)
        out = model.generate(**inp, do_sample=False, max_new_tokens=max_new,
                             pad_token_id=proc.tokenizer.eos_token_id)
        new = out[0, inp["input_ids"].shape[1]:]
        return proc.tokenizer.decode(new, skip_special_tokens=True)
    return gen


def make_hf(model_id, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", proc)

    def build(row, ui, uc):
        opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, row["options"]))
        content = []
        if uc:
            content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
        if ui:
            content.append({"type": "image"})
        content.append({"type": "text", "text": f"Question: {row['question']}\n{opts}\n\n{cot_instr(len(row['options']))}"})
        return [{"role": "user", "content": content}]

    def gen(row, ui, uc, max_new):
        prompt = proc.apply_chat_template(build(row, ui, uc), add_generation_prompt=True,
                                          tokenize=False)
        img = Image.open(row["image_path"]).convert("RGB") if ui else None
        inp = proc(text=[prompt], images=[img] if img else None, return_tensors="pt").to(device)
        out = model.generate(**inp, do_sample=False, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id)
        new = out[0, inp["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True)
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--pool", default="both", choices=["dm", "assembled", "both"])
    ap.add_argument("--dm-split", default="test", choices=["test", "val", "valtest", "all"],
                    help="dm pool scope; 'test' matches every S7 in-style number and is 5x cheaper")
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--out-suffix", default="_cot", help="output key suffix ({model}{suffix}.jsonl)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (smoke-test with e.g. 20)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    family, model_id, cls_tag = REGISTRY[args.model]
    gen = (make_qwen(model_id, cls_tag, args.device) if family == "qwen"
           else make_hf(model_id, args.device))
    key = f"{args.model}{args.out_suffix}"
    pools = ["dm", "assembled"] if args.pool == "both" else [args.pool]

    with torch.inference_mode():
        for pool in pools:
            rows = load_rows(pool, cfg, args.dm_split)
            if args.limit:
                rows = rows[:args.limit]
            out = (Path(cfg["paths"]["data_root"])
                   / ("eval/t8_assembled" if pool == "assembled" else "eval/t8") / f"{key}.jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(l)["candidate_id"] for l in out.read_text().splitlines()
                    if l.strip()} if out.exists() else set()
            pending = [r for r in rows if r["candidate_id"] not in done]
            print(f"[{pool}] {len(done)} done, {len(pending)} pending of {len(rows)}")
            nbad = 0
            with out.open("a") as f:
                for r in tqdm(pending, desc=f"{key}/{pool}"):
                    rec = {"candidate_id": r["candidate_id"], "source": r.get("source"),
                           "label": r["label"], "correct_index": int(r["correct_index"])}
                    for name, ui, uc in COND:
                        text = gen(r, ui, uc, args.max_new_tokens)
                        pred = parse_letter(text)
                        rec[f"pred_{name}"] = pred
                        rec[f"gen_{name}"] = text
                        rec[f"parsed_{name}"] = pred >= 0
                        nbad += (pred < 0)
                    f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"  -> {out}  (unparseable generations: {nbad})")


if __name__ == "__main__":
    main()
