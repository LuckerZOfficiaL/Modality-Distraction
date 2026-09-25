r"""M3ID capability: what does contrastive decoding cost on the general benchmarks?

Every other mitigation row in tab:baseline has a capability number; M3ID's is a dash, because
scripts/benchmark_capability.py runs a single image-conditioned forward and M3ID needs two: the image-conditioned
logits l_c and the image-free logits l_u. This script caches both per row and applies the same
decoding rule scripts/m3id_baseline.py uses, at each model's ALREADY-SELECTED (mu, alpha) -- no re-tuning, so
the capability number is a held-out consequence of a choice made on the distraction surfaces.

    l* = l_c + 1[ max softmax(l_c) < alpha ] * mu * (l_c - l_u)

The image-free condition is the same prompt with the image block dropped, mirroring the T-only
condition of the three-condition verification. Both conditions are restricted to the option-letter
tokens, so per-condition log-partition constants cancel in the argmax.

Writes data/diagnostics/capability/{benchmark}_{tag}_m3idcap_rep.json in scripts/benchmark_capability.py's format, so
the existing readers (make_table_baseline.py, verify_numbers.py) pick it up unchanged. The `acc`
field is the M3ID accuracy; `acc_base` is the mu=0 accuracy in the same channel, which is what the
capability delta must be taken against.

    CUDA_VISIBLE_DEVICES=1 python scripts/m3id_capability.py --model qwen3b --benchmark mmstar
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from moground.steering import get_letter_token_ids, special_token_kwargs

ROOT = Path(__file__).resolve().parent.parent
MAXPX = 1024 * 1024
LETTERS = ["A", "B", "C", "D", "E", "F"]


def _load(name, fn):
    sp = importlib.util.spec_from_file_location(name, Path(__file__).parent / fn)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


c95 = _load("_c95", "cot_baseline.py")        # tag -> (family, model_id, qwen_cls)

# m3id_baseline.json is keyed by the scripts/behavioral_eval.py model key, not the scripts/cot_baseline.py tag.
M3KEY = {"qwen7b": "Qwen_Qwen2.5-VL-7B-Instruct", "internvl": "OpenGVLab_InternVL3-8B-hf",
         "qwen3b": "qwen", "llavaov": "llava-hf_llava-onevision-qwen2-7b-ov-hf",
         "next": "llavanext", "qwen2b": "Qwen_Qwen2-VL-2B-Instruct",
         "llava15": "llava-hf_llava-1.5-7b-hf"}


def prompt_text(row):
    opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, row["options"]))
    return f"{row['question']}\n{opts}\nAnswer with the option's letter only."


def load_qwen(model_id, cls_tag, device):
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
    from qwen_vl_utils import process_vision_info
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen2_vl": Qwen2VLForConditionalGeneration,
           "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[cls_tag]
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    lids = get_letter_token_ids(proc, device, n_letters=6)

    def logits(row, use_image):
        content = ([{"type": "image", "image": row["image_path"], "max_pixels": MAXPX}] if use_image else [])
        content.append({"type": "text", "text": prompt_text(row)})
        msg = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msg)
        if imgs is not None and len(imgs) == 0:
            imgs = None
        inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(device)
        li = int(inp.attention_mask[0].sum().item() - 1)
        k = len(row["options"])
        return model(**inp).logits[0, li][lids[:k]].float().cpu().numpy().astype(np.float64)
    return logits


def load_hf(model_id, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    lids = get_letter_token_ids(proc, device, n_letters=6)
    stk = special_token_kwargs(model_id)

    def logits(row, use_image):
        content = ([{"type": "image"}] if use_image else [])
        content.append({"type": "text", "text": prompt_text(row)})
        msg = [{"role": "user", "content": content}]
        prompt = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        img = [Image.open(row["image_path"]).convert("RGB")] if use_image else None
        inp = proc(text=[prompt], images=img, return_tensors="pt", **stk).to(device)
        li = int(inp["attention_mask"][0].sum().item() - 1)
        k = len(row["options"])
        return model(**inp).logits[0, li][lids[:k]].float().cpu().numpy().astype(np.float64)
    return logits


def m3id(lc, lu, mu, alpha):
    p = np.exp(lc - lc.max()); p /= p.sum()
    return lc + (mu * (lc - lu) if p.max() < alpha else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(M3KEY))
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--key", default=None, help="default {model}_m3idcap_rep")
    ap.add_argument("--skip", type=int, default=400, help="reporting slice starts at row 400")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    key = a.key or f"{a.model}_m3idcap_rep"

    M3 = json.loads((ROOT / "data/diagnostics/m3id_baseline.json").read_text())
    mu, alpha = M3[M3KEY[a.model]]["selected"]
    family, model_id, cls_tag = c95.REGISTRY[a.model]
    rows = [json.loads(l) for l in
            (ROOT / f"data/benchmarks/{a.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    rows = rows[a.skip:][:a.limit] if a.limit else rows[a.skip:]
    print(f"=== m3id-cap {a.model} on {a.benchmark}: mu={mu} alpha={alpha}, {len(rows)} rows ===")

    lg = (load_qwen(model_id, cls_tag, a.device) if family == "qwen" else load_hf(model_id, a.device))
    preds, base_preds, gated = [], [], 0
    with torch.inference_mode():
        for r in tqdm(rows, desc=f"{a.benchmark}/{key}"):
            lc = lg(r, True); lu = lg(r, False)
            p = np.exp(lc - lc.max()); p /= p.sum()
            gated += int(p.max() < alpha)
            preds.append(int(m3id(lc, lu, mu, alpha).argmax()))
            base_preds.append(int(lc.argmax()))
    gold = [int(r["correct_index"]) for r in rows]
    acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
    accb = float(np.mean([p == g for p, g in zip(base_preds, gold)]))
    out = ROOT / "data/diagnostics/capability"; out.mkdir(parents=True, exist_ok=True)
    res = {"benchmark": a.benchmark, "model": a.model, "key": key, "adapter": None,
           "n": len(rows), "acc": acc, "se": float(np.sqrt(acc * (1 - acc) / len(rows))),
           "acc_base": accb, "d_acc_pp": 100 * (acc - accb),
           "mu": mu, "alpha": alpha, "gate_applied_frac": gated / len(rows),
           "skip": a.skip, "preds": preds, "base_preds": base_preds, "correct_index": gold}
    (out / f"{a.benchmark}_{key}.json").write_text(json.dumps(res, indent=2))
    print(f"\n{a.benchmark} [{key}]  base={accb:.4f} -> m3id={acc:.4f} "
          f"({100*(acc-accb):+.2f}pp)  gate fired on {100*gated/len(rows):.0f}% of rows")


if __name__ == "__main__":
    main()
