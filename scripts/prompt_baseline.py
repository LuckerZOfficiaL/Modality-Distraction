"""Prompt-instruction baseline for the mitigation comparison (S7), reproduced.

Single forward pass, letter-logit argmax (same readout as scripts/behavioral_eval.py, NOT generation),
with the verbatim S7 instruction prepended to the prompt in all three input conditions.
Writes {tag}_promptbase.jsonl with the exact schema of the existing 5-model runs
(candidate_id, source, label, correct_index, pred_vt/pred_v/pred_t) so vd() scores it unchanged.

The original generator was not kept in-repo; this reconstructs it. VALIDATE first by
re-running an existing model (e.g. qwen3b) and diffing against its stored _promptbase.jsonl
before trusting the three new backbones.

  # validate the harness reproduces an existing model:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/prompt_baseline.py --model qwen3b --pool both --out-suffix _promptbase_check
  # then the three missing models:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/prompt_baseline.py --model internvl --pool both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MAXPX = 1024 * 1024

INSTR = ("The caption may be irrelevant; judge from the image when they disagree, "
         "use the caption only if it clearly helps.")

REGISTRY = {
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
COND = [("vt", True, True), ("v", True, False), ("t", False, True)]


def load_rows(pool: str, cfg: dict, dm_split: str = "all") -> list[dict]:
    if pool == "assembled":
        p = Path(cfg["paths"]["data_root"]) / "dm/merged_aokvqa_racehigh/dm_all.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    dm = Path(cfg["paths"]["dm_dir"]); rows = []
    splits = ("test",) if dm_split == "test" else ("train", "val", "test")
    for sp in splits:
        f = dm / "splits" / f"{sp}.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def make_qwen(model_id, cls_tag, device):
    from qwen_vl_utils import process_vision_info
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
    from moground.steering import get_letter_token_ids
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen2_vl": Qwen2VLForConditionalGeneration,
           "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[cls_tag]
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    lids = get_letter_token_ids(proc, device, n_letters=6)

    def build(row, ui, uc):
        opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
        content = []
        if uc:
            content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
        if ui:
            content.append({"type": "image", "image": row["image_path"], "max_pixels": MAXPX})
        content.append({"type": "text", "text":
                        f"Question: {row['question']}\n{opts}\n\n{INSTR}\nReply with one letter (A-D)."})
        return [{"role": "user", "content": content}]

    def predict3(row):
        preds = []
        for _, ui, uc in COND:
            msgs = build(row, ui, uc)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            if imgs is not None and len(imgs) == 0:
                imgs = None
            inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(device)
            li = int(inp["attention_mask"][0].sum().item() - 1)
            k = len(row["options"])
            preds.append(int(model(**inp).logits[0, li][lids[:k]].argmax().item()))
        return preds
    return predict3


def make_hf(model_id, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from moground.steering import get_letter_token_ids
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    lids = get_letter_token_ids(proc, device, n_letters=6)

    def build(row, ui, uc):
        opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
        content = []
        if uc:
            content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
        if ui:
            content.append({"type": "image"})
        content.append({"type": "text", "text":
                        f"Question: {row['question']}\n{opts}\n\n{INSTR}\nReply with one letter (A-D)."})
        return [{"role": "user", "content": content}]

    def predict3(row):
        preds = []
        for _, ui, uc in COND:
            prompt = proc.apply_chat_template(build(row, ui, uc), add_generation_prompt=True, tokenize=False)
            img = Image.open(row["image_path"]).convert("RGB") if ui else None
            inp = proc(text=[prompt], images=[img] if img else None, return_tensors="pt").to(device)
            li = int(inp["attention_mask"][0].sum().item() - 1)
            k = len(row["options"])
            preds.append(int(model(**inp).logits[0, li][lids[:k]].argmax().item()))
        return preds
    return predict3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--pool", default="both", choices=["dm", "assembled", "both"])
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--out-suffix", default="_promptbase")
    ap.add_argument("--dm-split", default="all", choices=["all", "test"],
                    help="dm pool scope; 'test' = the 685 test items (enough to validate vd_test)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    family, model_id, cls_tag = REGISTRY[args.model]
    predict3 = (make_qwen(model_id, cls_tag, args.device) if family == "qwen"
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
            with out.open("a") as f:
                for r in tqdm(pending, desc=f"{key}/{pool}"):
                    p = predict3(r)
                    f.write(json.dumps({
                        "candidate_id": r["candidate_id"], "source": r.get("source"),
                        "label": r["label"], "correct_index": int(r["correct_index"]),
                        "pred_vt": p[0], "pred_v": p[1], "pred_t": p[2]}) + "\n"); f.flush()
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
