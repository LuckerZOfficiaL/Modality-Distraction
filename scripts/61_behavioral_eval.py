"""#3 Behavioral multi-model eval: broaden the grounding-strength law / asymmetry to
more backbones (open HF + frontier API) -- BEHAVIORAL ONLY, no SAEs/activations.

Runs the 3-condition (V+T / V-only / T-only) MCQ eval on D_M or the assembled pool for
a configurable model, writing data/eval/t8[_assembled]/{key}.jsonl in the SAME format as
scripts/49 -- so scripts/50 (leaderboard), 54 (grounding law), 58 (robustness) extend
automatically (they glob *.jsonl). Resumable.

Adapters (extend REGISTRY to add models):
  - qwen   : Qwen2.5-VL / Qwen2-VL family (logit-capable: forward + letter-logit argmax)
  - gemini : oracle_clients.GeminiClient (multimodal API; accuracy only)
  (OpenAI/Anthropic: add a client class to oracle_clients.make_client, then a 'gemini'-style entry.)

    CUDA_VISIBLE_DEVICES=0 python scripts/61_behavioral_eval.py --model qwen2.5-vl-7b --pool dm
    python scripts/61_behavioral_eval.py --model gemini-3-flash --pool dm        # API, no GPU
    python scripts/61_behavioral_eval.py --model qwen2.5-vl-7b --pool assembled
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
LETTER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
ALL_COND = [("VT", True, True), ("V", True, False), ("T", False, True)]  # (name, use_image, use_caption)
COND = list(ALL_COND)   # narrowed by --conds; a single-condition probe costs a third of a full pass

REGISTRY = {
    "qwen2.5-vl-3b":  {"type": "qwen",  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "cls": "qwen2_5_vl"},
    "qwen2.5-vl-7b":  {"type": "qwen",  "model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "cls": "qwen2_5_vl"},
    "qwen2-vl-2b":    {"type": "qwen",  "model_id": "Qwen/Qwen2-VL-2B-Instruct",   "cls": "qwen2_vl"},
    # 32B completes a within-family scale ladder 3B -> 7B -> 32B, isolating scale from architecture
    # 32B: registered but DOES NOT FIT LoRA training on one 80GB card (Phase 0 passed checks 1-7,
    # OOM'd on the training step at 79.17/79.25 GiB, single process). Kept for the record; use 14B.
    # Qwen3-VL-30B-A3B (MoE: 30B TOTAL / 3B ACTIVE -- carry that caveat in every size claim).
    # Size point from the TRANSFERRING side for task #25: Qwen2.5-VL 3B/7B both transfer.
    # Vision tower uses fused qkv (bare q/k/v/o LoRA names cannot match it); router is mlp.gate;
    # experts are nn.Parameter -- none of them matchable by the attn target set.
    "qwen3-vl-30b": {"type": "qwen", "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct", "cls": "qwen3_vl_moe"},
    "llavanext":      {"type": "llava", "module": "llavanext"},
    "gemini-3-flash": {"type": "gemini", "model_id": "gemini-3-flash-preview"},
    "gemini-2.5-flash": {"type": "gemini", "model_id": "gemini-2.5-flash"},
    # generic HF (logit-capable; try type 'hf' for any new open VLM):
    "internvl3-8b": {"type": "hf", "model_id": "OpenGVLab/InternVL3-8B-hf"},
    "internvl3-14b": {"type": "hf", "model_id": "OpenGVLab/InternVL3-14B-hf"},   # 2nd large model   # native -hf (InternVL2.5 is remote-code only)
    "llava-onevision-7b": {"type": "hf", "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf"},
    "llava-1.5-7b": {"type": "hf", "model_id": "llava-hf/llava-1.5-7b-hf"},
    # 12B: within-family size point for the size-vs-idiosyncrasy question (the 27B is an
    # assembled-transfer null; does its smaller sibling transfer?). Same template family ->
    # same double-BOS handling via CHAT_TEMPLATE_EMITS_BOS.
    # Mistral-Small-3.1 (4th family: Pixtral vision tower + Mistral LM, Tekken tokenizer; ~24B).
    # 3rd large model / 10th H1 ladder point. Native transformers mistral3 -> hf path.
    "mistral-small-3.1-24b": {"type": "hf", "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"},
}


def load_rows(pool: str, cfg: dict, dm_splits: str = "train,val,test") -> list[dict]:
    if pool == "assembled":
        p = Path(cfg["paths"]["data_root"]) / "dm/merged_aokvqa_racehigh/dm_all.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    dm = Path(cfg["paths"]["dm_dir"]); rows = []
    for sp in [x.strip() for x in dm_splits.split(",") if x.strip()]:
        f = dm / "splits" / f"{sp}.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def _scale_lora(model, spec):
    """Scale a loaded LoRA's effective update by spec['adapter_scale'] (0=base .. >1=over-edit).
    Continuous 'intervention strength' axis for the correctability phase diagram."""
    w = spec.get("adapter_scale", 1.0)
    if w == 1.0:
        return model
    n = 0
    for m in model.modules():
        if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict):
            for k in list(m.scaling.keys()):
                m.scaling[k] *= w
            n += 1
    print(f"scaled {n} LoRA layers by w={w}")
    return model


# ---- qwen-family adapter (logit-capable) ----
def make_qwen(spec, device):
    import torch
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
    from qwen_vl_utils import process_vision_info
    from moground.steering import get_letter_token_ids
    _s = importlib.util.spec_from_file_location("_c6", ROOT / "06c_collect_dm_counterfactual_activations.py")
    _c6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(_c6)
    bm, MAXPX = _c6.build_messages, 1024 * 1024
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration, "qwen2_vl": Qwen2VLForConditionalGeneration,
           "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[spec["cls"]]
    model = cls.from_pretrained(spec["model_id"], torch_dtype=torch.bfloat16, device_map=device).eval()
    if spec.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec["adapter"]).eval()
        model = _scale_lora(model, spec)
    proc = AutoProcessor.from_pretrained(spec["model_id"])
    lids = get_letter_token_ids(proc, device, n_letters=6)

    @torch.inference_mode()
    def predict3(row):
        preds, logs = [], []
        for name, ui, uc in COND:
            msgs = bm(row, name, MAXPX)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            if imgs is not None and len(imgs) == 0:
                imgs = None
            inp = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
            li = int(inp.attention_mask[0].sum().item() - 1)
            k = len(row["options"])
            lr = model(**inp).logits[0, li][lids[:k]]
            preds.append(int(lr.argmax().item())); logs.append([float(x) for x in lr.float().cpu()])
        return preds, logs           # logit-capable -> enables margin (#6)
    return predict3


# ---- generic HF-VLM adapter (logit-capable): any model with a standard chat template ----
def make_hf(spec, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    from moground.steering import get_letter_token_ids, special_token_kwargs
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], dtype=torch.bfloat16, device_map=device, trust_remote_code=True).eval()
    except ValueError as e:
        raise RuntimeError(
            f"'{spec['model_id']}' is not a transformers-native VLM (AutoModelForImageTextToText can't "
            f"load its config -- likely a remote-code-only checkpoint). Use a native '-hf' variant if one "
            f"exists (e.g. InternVL3-*-hf), or add a per-family adapter (mirror make_qwen).\n  original: {e}") from e
    if spec.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec["adapter"]).eval()
        model = _scale_lora(model, spec)
    proc = AutoProcessor.from_pretrained(spec["model_id"], trust_remote_code=True)
    lids = get_letter_token_ids(proc, device, n_letters=6)
    stk = special_token_kwargs(spec["model_id"])   # {} for every pre-existing backbone

    def build(row, ui, uc):
        content = []
        if uc:
            content.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
        if ui:
            content.append({"type": "image"})
        opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
        content.append({"type": "text", "text": f"Question: {row['question']}\n{opts}\nReply with one letter (A-D)."})
        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def predict3(row):
        preds, logs = [], []
        for _, ui, uc in COND:
            prompt = proc.apply_chat_template(build(row, ui, uc), add_generation_prompt=True, tokenize=False)
            img = Image.open(row["image_path"]).convert("RGB") if ui else None
            inp = proc(text=[prompt], images=[img] if img else None, return_tensors="pt", **stk).to(device)
            li = int(inp["attention_mask"][0].sum().item() - 1)
            k = len(row["options"]); lr = model(**inp).logits[0, li][lids[:k]]
            preds.append(int(lr.argmax().item())); logs.append([float(x) for x in lr.float().cpu()])
        return preds, logs
    return predict3


# ---- llava-family adapter (reuse module predict_mcq) ----
def make_llava(spec, device):
    mod = importlib.import_module(f"moground.models_{spec['module']}")
    load = getattr(mod, f"load_{spec['module']}"); predict_mcq = mod.predict_mcq
    model, proc = load(device=device)
    if spec.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec["adapter"]).eval()
        model = _scale_lora(model, spec)

    def predict3(row):
        preds = [predict_mcq(model, proc, row["question"], row["options"],
                             image_path=row["image_path"] if ui else None,
                             caption=row["caption_for_filter"] if uc else None)
                 for _, ui, uc in COND]
        return preds, None           # generate+parse -> accuracy only (no logits)
    return predict3


# ---- gemini adapter (multimodal API; accuracy only) ----
def make_gemini(spec, device):
    from moground.oracle_clients import make_client, ImageInput
    client = make_client("gemini", model=spec["model_id"])
    sysmsg = "Answer the multiple-choice question with exactly one letter (A, B, C, or D)."

    def predict3(row):
        opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
        out = []
        for _, ui, uc in COND:
            user = (f"Caption: {row['caption_for_filter']}\n" if uc else "") + f"Question: {row['question']}\n{opts}"
            imgs = [ImageInput(path=row["image_path"], max_dim=1024)] if ui else []
            txt = client.complete(sysmsg, user, imgs)
            mlet = LETTER.search(txt or "")
            out.append(ord(mlet.group(1).upper()) - ord("A") if mlet else -1)
        return out, None             # closed API -> accuracy only (no logits)
    return predict3


def run_pool(predict3, pool, key, cfg, limit, dm_splits="train,val,test", ids=None):
    rows = load_rows(pool, cfg, dm_splits)
    if ids is not None:
        rows = [r for r in rows if r["candidate_id"] in ids]
    if limit:
        rows = rows[:limit]
    out = Path(cfg["paths"]["data_root"]) / ("eval/t8_assembled" if pool == "assembled" else "eval/t8") / f"{key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["candidate_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    pending = [r for r in rows if r["candidate_id"] not in done]
    print(f"  [{pool}] {len(done)} done, {len(pending)} pending of {len(rows)}")
    if pending:
        with out.open("a") as f:
            for r in tqdm(pending, desc=f"{key}/{pool}"):
                preds, logits = predict3(r)
                rec = {"candidate_id": r["candidate_id"], "source": r.get("source"),
                       "label": r["label"], "correct_index": int(r["correct_index"])}
                # key by condition name, not position: with --conds the tuple is shorter, and
                # positional unpacking would silently mislabel (or crash on) a narrowed run.
                for (nm, _, _), pv in zip(COND, preds):
                    rec[f"pred_{nm.lower()}"] = pv
                if logits is not None:  # logit-capable -> store for margin
                    for (nm, _, _), lg in zip(COND, logits):
                        rec[f"logits_{nm.lower()}"] = lg
                f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"  -> {out}")
    materialize_logit_store(out, key, pool, cfg)


def delete_hf_cache(model_id):
    import shutil
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as cache
    except Exception:
        cache = os.path.join(os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), "hub")
    d = Path(cache) / ("models--" + model_id.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"deleted HF checkpoint cache: {d}")
    else:
        print(f"(no HF cache dir at {d} to delete)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="a HuggingFace model path (e.g. OpenGVLab/InternVL2_5-8B), a REGISTRY key, "
                         "or 'qwen:<id>:<cls>' / 'gemini:<id>'")
    ap.add_argument("--pool", default="both", choices=["dm", "assembled", "both"])
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = all (smoke-test with e.g. 20)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the downloaded HF checkpoint (default: delete it after the run)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to load on top of the base model")
    ap.add_argument("--adapter-scale", type=float, default=1.0,
                    help="scale the LoRA update by this (0=base, 1=as-trained, >1=over-edit); phase-diagram y-axis")
    ap.add_argument("--key", default=None, help="output key (default: sanitized --model)")
    ap.add_argument("--conds", default="",
                    help="comma list of conditions to run (VT,V,T); default all three. A ceiling "
                         "probe only needs T, so this avoids paying for the other two.")
    ap.add_argument("--dm-splits", default="train,val,test",
                    help="comma list of dm splits to evaluate (e.g. 'val,test'); assembled unaffected")
    ap.add_argument("--ids-file", default=None,
                    help="json file with a list (or {'dev':[...]} dict via --ids-key) restricting candidate_ids")
    ap.add_argument("--ids-key", default=None, help="key into --ids-file when it is a dict")
    args = ap.parse_args()
    if args.conds:
        want = {c.strip().upper() for c in args.conds.split(",") if c.strip()}
        globals()["COND"] = [c for c in ALL_COND if c[0] in want]
        assert COND, f"--conds {args.conds} matched none of {[c[0] for c in ALL_COND]}"
        print(f"[conds] running only {[c[0] for c in COND]}")

    spec = REGISTRY.get(args.model)
    if spec is None:  # ad-hoc: bare 'org/model' (->hf) | 'qwen:<id>:<cls>' | 'gemini:<id>' | 'hf:<id>'
        parts = args.model.split(":")
        if parts[0] == "qwen":
            spec = {"type": "qwen", "model_id": parts[1], "cls": parts[2]}
        elif parts[0] == "gemini":
            spec = {"type": "gemini", "model_id": parts[1]}
        elif parts[0] == "hf":
            spec = {"type": "hf", "model_id": parts[1]}
        else:  # bare HuggingFace path
            spec = {"type": "hf", "model_id": args.model}
    spec = {**spec, "adapter": args.adapter, "adapter_scale": args.adapter_scale}
    key = args.key or re.sub(r"[^A-Za-z0-9._-]", "_", args.model)  # filesystem-safe output key
    cfg = yaml.safe_load(Path(args.config).read_text())
    pools = ["dm", "assembled"] if args.pool == "both" else [args.pool]
    print(f"=== {args.model} (type={spec['type']}, key={key}) pools={pools} "
          f"delete_after={'no' if args.keep else 'YES'} ===")
    predict3 = {"qwen": make_qwen, "hf": make_hf, "llava": make_llava,
                "gemini": make_gemini}[spec["type"]](spec, args.device)
    ids = None
    if args.ids_file:
        blob = json.loads(Path(args.ids_file).read_text())
        ids = set(blob[args.ids_key] if args.ids_key else blob)
    for pool in pools:
        run_pool(predict3, pool, key, cfg, args.limit, args.dm_splits, ids)
    if not args.keep and spec["type"] in ("hf", "qwen") and spec.get("model_id"):
        del predict3
        delete_hf_cache(spec["model_id"])


def materialize_logit_store(jsonl_path, key, pool, cfg):
    """If the run produced logits, write a cf-store-format letter_logits.npy + index.jsonl
    (data/activations/{key}/behavioral_{pool}/) so scripts/58,59 (margin)
    run on this model via --model {key} --cf-subdir behavioral_{pool}."""
    import numpy as np
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows or "logits_vt" not in rows[0]:
        return  # accuracy-only model (llava/gemini); nothing to materialize
    k = max(len(r["logits_vt"]) for r in rows)
    ll = np.zeros((len(rows), 3, k), dtype=np.float16)
    for i, r in enumerate(rows):
        for c, fld in enumerate(("logits_vt", "logits_v", "logits_t")):
            v = r.get(fld)          # absent under --conds narrowing; leave those slots zero
            if v is not None:
                ll[i, c, :len(v)] = v
    sd = Path(cfg["paths"]["data_root"]) / "activations" / key / f"behavioral_{pool}"
    sd.mkdir(parents=True, exist_ok=True)
    np.save(sd / "letter_logits.npy", ll)
    (sd / "index.jsonl").write_text("".join(json.dumps(
        {"candidate_id": r["candidate_id"], "label": r["label"],
         "correct_index": int(r["correct_index"]), "source": r.get("source")}) + "\n" for r in rows))
    print(f"-> logit store {sd}  (margin: scripts/58,59 --model {key} --cf-subdir behavioral_{pool})")


if __name__ == "__main__":
    main()
