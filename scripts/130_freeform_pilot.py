"""Free-form pilot on MoGround's certified vision items.

The paper measures distraction with multiple-choice items. A reviewer will ask whether the effect
is an artifact of the MCQ format, since VLMs are mostly used in free-form generation. This script
re-runs the SAME certified items with the options removed, asking for a short free-form answer
under the same three input conditions.

Nothing else changes: same images, same captions, same questions, same certification. Only the
answer format differs, so any difference in the measured flip rate is attributable to the format.

    CUDA_VISIBLE_DEVICES=2 python scripts/130_freeform_pilot.py --model qwen2.5-vl-3b --n 420

Writes data/eval/freeform/{key}.jsonl, one row per item with the raw generation per condition.
Resumable (skips candidate_ids already present).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/eval/freeform"
MAXPX = 1024 * 1024
CONDS = [("VT", True, True), ("V", True, False), ("T", False, True)]

REGISTRY = {
    "qwen2.5-vl-3b": {"type": "qwen", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "cls": "qwen2_5_vl"},
    "qwen2.5-vl-7b": {"type": "qwen", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "cls": "qwen2_5_vl"},
    "qwen2-vl-2b":   {"type": "qwen", "model_id": "Qwen/Qwen2-VL-2B-Instruct",   "cls": "qwen2_vl"},
    "llava-1.5-7b":  {"type": "hf",   "model_id": "llava-hf/llava-1.5-7b-hf"},
    "llava-onevision-7b": {"type": "hf", "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf"},
    "internvl3-8b":  {"type": "hf",   "model_id": "OpenGVLab/InternVL3-8B-hf"},
    # LLaVA-NeXT is not loadable through AutoModelForImageTextToText in this env, so it reuses the
    # project's own loader, exactly as scripts/61 does for the MCQ eval.
    "llavanext":     {"type": "hf",   "model_id": "llava-hf/llama3-llava-next-8b-hf"},
}

# The free-form instruction mirrors the MCQ one minus the option list. "a few words" keeps the
# output scorable against the gold option string without forcing single-word answers, which would
# re-introduce a closed format through the back door.
INSTR = "Answer the question in a few words, based only on the information given."


def build_parts(row, ui, uc):
    parts = []
    if uc:
        parts.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
    if ui:
        parts.append({"type": "image", "image": row["image_path"], "max_pixels": MAXPX})
    parts.append({"type": "text", "text": f"Question: {row['question']}\n\n{INSTR}"})
    return [{"role": "user", "content": parts}]


def _attach(model, spec):
    """Load the robustness task vector and scale its effective update by w, exactly as
    scripts/61_behavioral_eval.py does, so the free-form numbers sit on the same intervention."""
    if not spec.get("adapter"):
        return model
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, spec["adapter"]).eval()
    w = spec.get("adapter_scale", 1.0)
    if w != 1.0:
        n = 0
        for m in model.modules():
            if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict):
                for k in list(m.scaling.keys()):
                    m.scaling[k] *= w
                n += 1
        print(f"scaled {n} LoRA layers by w={w}", flush=True)
    return model


def make_qwen(spec, device, max_new):
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration)
    from qwen_vl_utils import process_vision_info
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen2_vl": Qwen2VLForConditionalGeneration}[spec["cls"]]
    model = cls.from_pretrained(spec["model_id"], torch_dtype=torch.bfloat16, device_map=device).eval()
    model = _attach(model, spec)
    proc = AutoProcessor.from_pretrained(spec["model_id"])

    @torch.inference_mode()
    def gen3(row):
        outs = []
        for name, ui, uc in CONDS:
            msgs = build_parts(row, ui, uc)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            if imgs is not None and len(imgs) == 0:
                imgs = None
            inp = proc(text=[text], images=imgs, videos=vids, padding=True,
                       return_tensors="pt").to(model.device)
            ids = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
            new = ids[0][inp.input_ids.shape[1]:]
            outs.append(proc.decode(new, skip_special_tokens=True).strip())
        return outs
    return gen3


def make_hf(spec, device, max_new):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    model = AutoModelForImageTextToText.from_pretrained(
        spec["model_id"], dtype=torch.bfloat16, device_map=device, trust_remote_code=True).eval()
    model = _attach(model, spec)
    proc = AutoProcessor.from_pretrained(spec["model_id"], trust_remote_code=True)

    @torch.inference_mode()
    def gen3(row):
        outs = []
        for name, ui, uc in CONDS:
            content = []
            if uc:
                content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
            if ui:
                content.append({"type": "image"})
            content.append({"type": "text", "text": f"Question: {row['question']}\n\n{INSTR}"})
            prompt = proc.apply_chat_template([{"role": "user", "content": content}],
                                              add_generation_prompt=True, tokenize=False)
            img = Image.open(row["image_path"]).convert("RGB") if ui else None
            inp = proc(text=prompt, images=img, return_tensors="pt").to(model.device, torch.bfloat16)
            ids = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
            new = ids[0][inp.input_ids.shape[1]:]
            outs.append(proc.decode(new, skip_special_tokens=True).strip())
        return outs
    return gen3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--key", default=None, help="output file stem (default: model name)")
    ap.add_argument("--n", type=int, default=0, help="0 = all certified vision test items")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--pool", default="test", choices=("test", "human", "assembled"))
    ap.add_argument("--labels", default="vision", choices=("vision", "text", "both"),
                    help="vision items give v-distraction, text items give t-distraction")
    ap.add_argument("--adapter", default=None, help="LoRA dir, e.g. runs/ft_kl2.0_qwen7b_s0")
    ap.add_argument("--adapter-scale", type=float, default=1.0, help="w; the paper's default is 0.5")
    args = ap.parse_args()

    # the paper's three reporting pools, same filters as scripts/61 and make_table_pareto.py
    if args.pool == "test":
        src = ROOT / "data/dm/multidomain_v1/splits/test.jsonl"; keep = viol = None
    elif args.pool == "human":
        src = ROOT / "data/dm/human_moground/frozen_20260801.jsonl"; keep = viol = None
    else:
        src = ROOT / "data/dm/merged_aokvqa_racehigh/dm_all.jsonl"
        keep = set(json.loads((ROOT / "data/dm/merged_aokvqa_racehigh/"
                               "assembled_dev_heldout_split.json").read_text())["heldout"])
        viol = set(json.loads((ROOT / "data/diagnostics/"
                               "assembled_caption_certification.json").read_text())["violations"])
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    seen_id = set()
    items = []
    for r in rows:                                   # first-wins dedup (13 known duplicate ids)
        if r["candidate_id"] in seen_id:
            continue
        if args.labels != "both" and r["label"] != args.labels:
            continue
        if keep is not None and (r["candidate_id"] not in keep or r["candidate_id"] in viol):
            continue
        seen_id.add(r["candidate_id"]); items.append(r)
    if args.n:
        items = items[:args.n]

    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{args.key or args.model}.jsonl"
    done = set()
    if fp.exists():
        for l in fp.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["candidate_id"])
    todo = [r for r in items if r["candidate_id"] not in done]
    print(f"{args.model}: {len(items)} {args.labels} items in pool '{args.pool}', "
          f"{len(done)} done, {len(todo)} to go",
          flush=True)

    spec = dict(REGISTRY[args.model])
    if args.adapter:
        spec["adapter"] = args.adapter; spec["adapter_scale"] = args.adapter_scale
        print(f"adapter {args.adapter} @ w={args.adapter_scale}", flush=True)
    gen3 = (make_qwen if spec["type"] == "qwen" else make_hf)(spec, "cuda:0", args.max_new)

    with fp.open("a") as f:
        for r in tqdm(todo, desc=args.model):
            try:
                gvt, gv, gt = gen3(r)
            except Exception as e:                    # one bad image must not kill a long run
                print(f"  SKIP {r['candidate_id']}: {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps({
                "candidate_id": r["candidate_id"], "label": r["label"],
                # assembled items carry source_text_qa instead of source
                "source": r.get("source") or r.get("source_text_qa") or "assembled",
                "question": r["question"], "options": r["options"],
                "correct_index": r["correct_index"], "caption": r["caption_for_filter"],
                "gen_vt": gvt, "gen_v": gv, "gen_t": gt}) + "\n")
            f.flush()
    print("wrote", fp, flush=True)


if __name__ == "__main__":
    main()
