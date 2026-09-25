"""Plan A / A1: grounding as a design lever via VISUAL-TOKEN BUDGET (the positive arm of the
downscale knob). For the Qwen family the budget is set by max_pixels (06c.build_messages plumbs
it), so sweeping it below->above the default traces one continuous dose-response: more visual
tokens -> higher V-only grounding -> (hypothesis) lower v-distraction, ALONG the cross-model law.
This is a GLOBAL intervention (shifts the whole margin distribution), so it needs no per-input
selectivity -- the wall that killed steering and gating.

Per budget, measure on vision items: V-only grounding and v-distraction (V correct & V+T wrong).
Output: data/diagnostics/grounding_lever_{key}_tokens[_assembled].json  (trajectory over budgets).

Qwen family only for now (clean max_pixels control); InternVL tile-count / LLaVA anyres = TODO.

    CUDA_VISIBLE_DEVICES=0 python scripts/67_grounding_lever_tokenbudget.py --model qwen2.5-vl-7b --pool both
    CUDA_VISIBLE_DEVICES=0 python scripts/67_grounding_lever_tokenbudget.py --model qwen2.5-vl-3b --key qwen --pool both
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s61", ROOT / "61_behavioral_eval.py")
s61 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s61)  # type: ignore
PATCH = 28 * 28  # Qwen visual token = 28x28 px; budget given in TOKENS -> max_pixels = tok * 784


def load_qwen(model_id, cls_name, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from moground.steering import get_letter_token_ids
    _c = importlib.util.spec_from_file_location("_c6", ROOT / "06c_collect_dm_counterfactual_activations.py")
    c6 = importlib.util.module_from_spec(_c); _c.loader.exec_module(c6)
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration, "qwen2_vl": Qwen2VLForConditionalGeneration}[cls_name]
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    lids = get_letter_token_ids(proc, device, n_letters=6)

    @torch.inference_mode()
    def predict(row, kind, max_pixels):  # kind in {"V","VT"}; returns predicted option index
        msgs = c6.build_messages(row, kind, max_pixels)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        if imgs is not None and len(imgs) == 0:
            imgs = None
        inp = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
        li = int(inp.attention_mask[0].sum().item() - 1)
        k = len(row["options"])
        return int(model(**inp).logits[0, li][lids[:k]].argmax().item())
    return model, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="qwen-family REGISTRY key or HF path")
    ap.add_argument("--key", default=None)
    ap.add_argument("--pool", default="both", choices=["dm", "assembled", "both"])
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--budgets", default="256,512,1280,2560,5120", help="visual-token budgets (default ~1280)")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    spec = s61.REGISTRY.get(args.model)
    if spec is None:  # bare HF path -> infer class from the name (Qwen2.5-VL vs Qwen2-VL)
        cls = "qwen2_5_vl" if ("2.5-VL" in args.model or "2_5" in args.model) else "qwen2_vl"
        spec = {"type": "qwen", "model_id": args.model, "cls": cls}
    assert spec["type"] == "qwen", "A1 token-budget lever supports the Qwen family only (max_pixels control)"
    cfg = yaml.safe_load(Path(args.config).read_text()); dr = Path(cfg["paths"]["data_root"])
    key = args.key or re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    budgets = [int(b) for b in args.budgets.split(",")]
    pools = ["dm", "assembled"] if args.pool == "both" else [args.pool]
    print(f"=== A1 token-budget lever {args.model} (key={key}) budgets(tok)={budgets} pools={pools} ===")
    model, predict = load_qwen(spec["model_id"], spec["cls"], args.device)

    for pool in pools:
        rows = [r for r in s61.load_rows(pool, cfg) if r["label"] == "vision"]
        rng = np.random.default_rng(args.seed); rng.shuffle(rows); rows = rows[:args.limit]
        traj = []
        for tok in budgets:
            mp = tok * PATCH
            pv, pvt, ci = [], [], []
            for r in tqdm(rows, desc=f"{key}/{pool} budget={tok}tok", leave=False):
                pv.append(predict(r, "V", mp)); pvt.append(predict(r, "VT", mp)); ci.append(r["correct_index"])
            pv, pvt, ci = map(np.array, (pv, pvt, ci)); vok = pv == ci
            traj.append({"budget_tok": tok, "max_pixels": mp,
                         "grounding": round(float(vok.mean()), 4),
                         "v_distraction": round(float(((pv == ci) & (pvt != ci)).sum() / max(vok.sum(), 1)), 4),
                         "vt_acc": round(float((pvt == ci).mean()), 4), "n": len(rows)})
            print(f"  budget={tok:>5}tok: grounding={traj[-1]['grounding']:.3f}  v-distraction={traj[-1]['v_distraction']:.3f}")
        suffix = "_assembled" if pool == "assembled" else ""
        out = dr / "diagnostics" / f"grounding_lever_{key}_tokens{suffix}.json"
        out.write_text(json.dumps({"model": key, "pool": pool, "knob": "token_budget", "trajectory": traj}, indent=2))
        print(f"-> {out}")

    if not args.keep and spec.get("model_id"):
        del model; torch.cuda.empty_cache(); s61.delete_hf_cache(spec["model_id"])


if __name__ == "__main__":
    main()
