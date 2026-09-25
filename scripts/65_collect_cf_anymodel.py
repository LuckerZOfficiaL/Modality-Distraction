"""Step 65: collect 3-condition (V+T / V-only / T-only) within-item counterfactual activations
for ANY backbone on EITHER pool (D_M or assembled) -- generalizes 06c (qwen) and 52 (llava) to the
full 8-model set so the mechanism analyses (T2/X2/X4/margin: scripts 53/46/45/59) extend cross-model.

Per item, one forward per condition with output_hidden_states; stores the last-input-token residual
at EVERY LM layer + the 4 letter-token logits. Each family uses the SAME message builder its T8/law
run used (qwen=06c.build_messages, hf=generic build, llava=module _build_messages), so the cf-store's
own V-solvable/distracted labels match the behavioral law. Pool items are the FIXED oracle (D_M) /
cosine-filtered (assembled) set -- never re-filtered per model. Download-run-delete (hf/qwen only;
llava checkpoints are left in place since the SAE collection may be using them).

  data/activations/{key}/{out}/activations.npy   (N, 3, n_layers, d)  f16
  data/activations/{key}/{out}/letter_logits.npy (N, 3, 4)            f16
  data/activations/{key}/{out}/index.jsonl
  default out: dm -> dm_multidomain_v1_cf ; assembled -> merged_aokvqa_racehigh_cf

    CUDA_VISIBLE_DEVICES=0 python scripts/65_collect_cf_anymodel.py --model qwen2.5-vl-3b --key qwen --pool assembled
    CUDA_VISIBLE_DEVICES=0 python scripts/65_collect_cf_anymodel.py --model OpenGVLab/InternVL3-8B-hf --pool both
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s61", ROOT / "61_behavioral_eval.py")
s61 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s61)  # type: ignore
COND = s61.COND          # [("VT",T,T), ("V",T,F), ("T",F,T)]
NOPT = 4                 # D_M and assembled are 4-way


def resolve(model):
    spec = s61.REGISTRY.get(model)
    if spec is None:
        p = model.split(":")
        spec = ({"qwen": {"type": "qwen", "model_id": p[1] if len(p) > 1 else "", "cls": p[2] if len(p) > 2 else ""},
                 "gemini": {"type": "gemini", "model_id": p[-1]},
                 "hf": {"type": "hf", "model_id": p[-1]}}.get(p[0], {"type": "hf", "model_id": model}))
    return spec


def setup(spec, device):
    """Return (model, prep_fn, letter_ids, cleanup_model_id). prep_fn(row,name,ui,uc) -> inputs on device."""
    from moground.steering import get_letter_token_ids
    if spec["type"] == "qwen":
        from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                                  Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
        from qwen_vl_utils import process_vision_info
        _c = importlib.util.spec_from_file_location("_c6", ROOT / "06c_collect_dm_counterfactual_activations.py")
        c6 = importlib.util.module_from_spec(_c); _c.loader.exec_module(c6)
        MAXPX = 1024 * 1024
        cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration, "qwen2_vl": Qwen2VLForConditionalGeneration,
               "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[spec["cls"]]
        model = cls.from_pretrained(spec["model_id"], torch_dtype=torch.bfloat16, device_map=device).eval()
        proc = AutoProcessor.from_pretrained(spec["model_id"]); lids = get_letter_token_ids(proc, device, n_letters=6)

        def prep(row, name, ui, uc):
            msgs = c6.build_messages(row, name, MAXPX)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            if imgs is not None and len(imgs) == 0:
                imgs = None
            return proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
        return model, prep, lids, spec["model_id"]

    if spec["type"] == "hf":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], dtype=torch.bfloat16, device_map=device, trust_remote_code=True).eval()
        proc = AutoProcessor.from_pretrained(spec["model_id"], trust_remote_code=True)
        lids = get_letter_token_ids(proc, device, n_letters=6)

        def build(row, ui, uc):
            content = []
            if uc:
                content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
            if ui:
                content.append({"type": "image"})
            opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
            content.append({"type": "text", "text": f"Question: {row['question']}\n{opts}\nReply with one letter (A-D)."})
            return [{"role": "user", "content": content}]

        def prep(row, name, ui, uc):
            prompt = proc.apply_chat_template(build(row, ui, uc), add_generation_prompt=True, tokenize=False)
            img = Image.open(row["image_path"]).convert("RGB") if ui else None
            return proc(text=[prompt], images=[img] if img else None, return_tensors="pt").to(model.device)
        return model, prep, lids, spec["model_id"]

    if spec["type"] == "llava":
        mod = importlib.import_module(f"moground.models_{spec['module']}")
        load_m = getattr(mod, f"load_{spec['module']}"); _bm = mod._build_messages
        model, proc = load_m(device=device)
        lids = torch.tensor([proc.tokenizer.encode(l, add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]],
                            device=device)

        def prep(row, name, ui, uc):
            msgs = _bm(row["question"], row["options"],
                       row["image_path"] if ui else None, row["caption_for_filter"] if uc else None)
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
            image = Image.open(row["image_path"]).convert("RGB") if ui else None
            return proc(text=prompt, images=image, return_tensors="pt").to(model.device)
        return model, prep, lids, None   # llava: do not auto-delete (SAE collection may use it)

    raise ValueError(f"cf collection unsupported for type={spec['type']} (open-weights only)")


def _preflight_dtype(sample: np.ndarray, dtype, n_seen: int) -> None:
    """Fail loudly rather than silently writing saturated activations.

    65,504 ceiling: the store saturated to inf on 9 of 62 layers, which silently produced NaN
    logit-lens trajectories and a non-comparable X2. Large models are the candidates for this and
    a five-minute assertion is cheaper than re-running a GPU-day. Checked on the first few rows.
    """
    finfo = np.finfo(dtype)
    peak = float(np.nanmax(np.abs(sample.astype(np.float64))))
    nonfinite = int((~np.isfinite(sample)).sum())
    print(f"  [preflight] after {n_seen} rows: peak |activation| = {peak:.1f}, "
          f"{dtype.__name__} max = {finfo.max:.1f}, non-finite = {nonfinite}")
    if nonfinite or peak > 0.5 * float(finfo.max):
        raise SystemExit(
            f"\nABORT: activations do not fit {dtype.__name__} safely.\n"
            f"  peak |a| = {peak:.1f} vs dtype max {finfo.max:.1f}; non-finite so far = {nonfinite}\n"
            f"  This model has massive activations. Re-run with --act-dtype float32.\n"
            f"  (numpy has no bfloat16, so float32 is the safe storage type; it doubles the store.)")


@torch.inference_mode()
def collect_pool(model, prep, lids, rows, key, out_name, dr, act_dtype=np.float16):
    out = dr / "activations" / key / out_name; out.mkdir(parents=True, exist_ok=True)
    if (out / "activations.npy").exists() and np.load(out / "activations.npy", mmap_mode="r").shape[0] == len(rows):
        print(f"  [skip] {out} already complete ({len(rows)} items)"); return
    acts = logits = None
    checked = False
    for i, r in enumerate(tqdm(rows, desc=f"{key}/{out_name}")):
        for c, (name, ui, uc) in enumerate(COND):
            inp = prep(r, name, ui, uc)
            o = model(**inp, output_hidden_states=True)
            li = int(inp["attention_mask"][0].sum().item()) - 1   # last non-pad input token (matches 06c/52)
            hs = o.hidden_states; nl = len(hs) - 1
            if acts is None:
                d = hs[0].shape[-1]
                acts = np.zeros((len(rows), 3, nl, d), act_dtype); logits = np.zeros((len(rows), 3, NOPT), np.float16)
            acts[i, c] = torch.stack([hs[L + 1][0, li] for L in range(nl)]).float().cpu().numpy().astype(act_dtype)
            logits[i, c] = o.logits[0, li][lids[:NOPT]].float().cpu().numpy().astype(np.float16)
        if not checked and i >= 7:      # cheap guard, before the GPU-hours are spent
            _preflight_dtype(acts[: i + 1], act_dtype, i + 1); checked = True
    np.save(out / "activations.npy", acts); np.save(out / "letter_logits.npy", logits)
    (out / "index.jsonl").write_text("".join(json.dumps(
        {"candidate_id": r.get("candidate_id"), "label": r["label"],
         "correct_index": int(r["correct_index"]), "source": r.get("source")}) + "\n" for r in rows))
    print(f"-> {out}  (acts {acts.shape})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF path or REGISTRY key (see scripts/61)")
    ap.add_argument("--key", default=None, help="output key (default sanitized --model; use to match law cell, e.g. qwen)")
    ap.add_argument("--pool", default="both", choices=["dm", "assembled", "both"])
    # MoGround-Human is not one of the two registered pools, but its rows have the same schema.
    # Added 2026-09-22 to cache letter logits for LLaVA-NeXT on that pool: script 61's llava
    # adapter is generate-and-parse, so it never wrote an hm3 cf-store for that backbone.
    ap.add_argument("--rows-file", default=None,
                    help="JSONL of rows to use instead of the pool loader (same schema)")
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--out-name", default=None, help="override cf-subdir (default per pool)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--act-dtype", default="float16", choices=["float16", "float32"],
                    help="storage dtype for activations.npy. DEFAULT float16 keeps every existing "
                         "store byte-identical -- do not change it for the paper cohort. Use "
                         "overflows float16's 65,504 ceiling; numpy has no bfloat16, so float32 is "
                         "the safe choice, at 2x the disk.")
    args = ap.parse_args()
    act_dtype = {"float16": np.float16, "float32": np.float32}[args.act_dtype]

    cfg = yaml.safe_load(Path(args.config).read_text()); dr = Path(cfg["paths"]["data_root"])
    key = args.key or re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    pools = ["dm", "assembled"] if args.pool == "both" else [args.pool]
    spec = resolve(args.model)
    print(f"=== cf-collect {args.model} (type={spec['type']}, key={key}) pools={pools} ===")
    model, prep, lids, del_id = setup(spec, args.device)
    default_out = {"dm": "dm_multidomain_v1_cf", "assembled": "merged_aokvqa_racehigh_cf"}
    for pool in pools:
        if args.rows_file:
            rows = [json.loads(l) for l in Path(args.rows_file).read_text().splitlines() if l.strip()]
            print(f"  rows-file: {len(rows)} rows from {args.rows_file}")
        else:
            rows = s61.load_rows(pool, cfg)
        collect_pool(model, prep, lids, rows, key, args.out_name or default_out[pool], dr, act_dtype)
        if args.rows_file:
            break                      # a rows-file is pool-independent; collect it once
    if del_id and not args.keep:
        del model; torch.cuda.empty_cache(); s61.delete_hf_cache(del_id)


if __name__ == "__main__":
    main()
