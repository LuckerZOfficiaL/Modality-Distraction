"""General-capability check for the distraction-aware finetunes: plain single-condition MCQ accuracy
(image + question + options, NO caption) on external held-out VLM benchmarks, for base vs a LoRA
adapter, across all backbones. Answers "did finetuning cost general multimodal capability?" -- which
the D_M / assembled preservation metrics (A-OKVQA, RACE) do not cover.

Letter-logit argmax (forward once, argmax over the option-letter tokens), handles 2-6 options.

Families:
  qwen  : Qwen2.5-VL / Qwen2-VL (qwen_vl_utils + process_vision_info)

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_capability.py --model qwen2.5-vl-3b \
        --benchmark mmstar --key qwen3b_base
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_capability.py --model llavanext \
        --benchmark mmstar --adapter runs/ft_kl2.0_llavanext_s0 --key next_kl2.0
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from moground.steering import get_letter_token_ids, special_token_kwargs

ROOT = Path(__file__).resolve().parent.parent
MAXPX = 1024 * 1024
LETTERS = "ABCDEF"

# key -> (family, model_id).  qwen family also needs a class tag.
REGISTRY = {
    "qwen2.5-vl-3b": ("qwen", "Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl"),
    "qwen2.5-vl-7b": ("qwen", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
    "qwen2-vl-2b":   ("qwen", "Qwen/Qwen2-VL-2B-Instruct", "qwen2_vl"),
    "qwen3-vl-30b":  ("qwen", "Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen3_vl_moe"),  # MoE: 30B total / 3B active
    "llavanext":     ("hf", "llava-hf/llama3-llava-next-8b-hf", None),
    "internvl3-8b":  ("hf", "OpenGVLab/InternVL3-8B-hf", None),
    "internvl3-14b": ("hf", "OpenGVLab/InternVL3-14B-hf", None),
    "llava-onevision-7b": ("hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf", None),
    "llava-1.5-7b":  ("hf", "llava-hf/llava-1.5-7b-hf", None),
    "mistral-small-3.1-24b": ("hf", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", None),  # 3rd large model
}


PREPEND = ""          # set by --prepend: a policy instruction placed before the question


def prompt_text(row):
    opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, row["options"]))
    head = f"{PREPEND}\n" if PREPEND else ""
    return f"{head}{row['question']}\n{opts}\nAnswer with the option's letter only."


def scale_adapter(model, w):
    """Multiply every LoRA layer's effective update by w (w=0 -> base, w=1 -> as-trained, w>1 -> over-edit).
    This is the continuous 'intervention strength' axis for the correctability phase diagram."""
    if w == 1.0:
        return
    n = 0
    for m in model.modules():
        if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict):
            for k in list(m.scaling.keys()):
                m.scaling[k] *= w
            n += 1
    print(f"scaled {n} LoRA layers by w={w}")


def load_qwen(model_id, cls_tag, adapter, device, scale=1.0):
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
    from qwen_vl_utils import process_vision_info
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen2_vl": Qwen2VLForConditionalGeneration,
           "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[cls_tag]
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
        scale_adapter(model, scale)
    proc = AutoProcessor.from_pretrained(model_id)
    lids = get_letter_token_ids(proc, device, n_letters=6)

    def predict(row):
        msg = [{"role": "user", "content": [
            {"type": "image", "image": row["image_path"], "max_pixels": MAXPX},
            {"type": "text", "text": prompt_text(row)}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msg)
        inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(device)
        li = int(inp.attention_mask[0].sum().item() - 1)
        k = len(row["options"])
        return int(model(**inp).logits[0, li][lids[:k]].argmax().item())
    return predict


def load_hf(model_id, adapter, device, scale=1.0):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
        scale_adapter(model, scale)
    proc = AutoProcessor.from_pretrained(model_id)
    lids = get_letter_token_ids(proc, device, n_letters=6)
    stk = special_token_kwargs(model_id)   # {} for every pre-existing backbone

    def predict(row):
        msg = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_text(row)}]}]
        prompt = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        img = Image.open(row["image_path"]).convert("RGB")
        inp = proc(text=[prompt], images=[img], return_tensors="pt", **stk).to(device)
        li = int(inp["attention_mask"][0].sum().item() - 1)
        k = len(row["options"])
        return int(model(**inp).logits[0, li][lids[:k]].argmax().item())
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry key (qwen2.5-vl-3b, llavanext, ...)")
    ap.add_argument("--benchmark", required=True, help="mmstar, naturalbench, mmbench, seedbench, scienceqa")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base)")
    ap.add_argument("--adapter-scale", type=float, default=1.0,
                    help="scale the LoRA update by this (0=base, 1=as-trained, >1=over-edit); phase-diagram y-axis")
    ap.add_argument("--key", required=True, help="output label, e.g. qwen3b_kl2.0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N rows (report slice = --skip 400 --limit 400)")
    ap.add_argument("--prepend", default="", help="policy instruction prepended to every prompt "
                    "(use to score the prompt-instruction baseline on capability benchmarks)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global PREPEND
    PREPEND = args.prepend
    if args.model not in REGISTRY:
        raise SystemExit(f"unknown --model {args.model!r}; known: {sorted(REGISTRY)}")
    family, model_id, cls_tag = REGISTRY[args.model]
    rows = [json.loads(l) for l in (ROOT / f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    if args.skip:
        rows = rows[args.skip:]
    if args.limit:
        rows = rows[:args.limit]

    predict = (load_qwen(model_id, cls_tag, args.adapter, args.device, args.adapter_scale) if family == "qwen"
               else load_hf(model_id, args.adapter, args.device, args.adapter_scale))

    correct = 0
    preds = []
    with torch.inference_mode():
        for r in tqdm(rows, desc=f"{args.benchmark}/{args.key}"):
            p = predict(r)
            preds.append(int(p))
            correct += int(p == r["correct_index"])
    acc = correct / len(rows)
    out = ROOT / "data/diagnostics/capability"; out.mkdir(parents=True, exist_ok=True)
    res = {"benchmark": args.benchmark, "model": args.model, "key": args.key, "adapter": args.adapter,
           "n": len(rows), "acc": acc, "se": float(np.sqrt(acc * (1 - acc) / len(rows))),
           "skip": args.skip, "preds": preds,
           "correct_index": [int(r["correct_index"]) for r in rows]}
    (out / f"{args.benchmark}_{args.key}.json").write_text(json.dumps(res, indent=2))
    print(f"\n{args.benchmark} [{args.key}]  acc={acc:.4f} ± {res['se']:.4f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
