"""Step 42 (T6): causal test of the robustness asymmetry via reverse patching.

T2 showed (correlationally) that the image perturbs text-item reps MORE than the
caption perturbs vision-item reps, yet only the caption harms accuracy. T6 makes it
CAUSAL: start from the single-modality forward the model answers CORRECTLY
(V-only for vision items, T-only for text items), inject the JOINT state h_VT (which
carries the irrelevant modality's influence) at the last-token residual of layer L,
and measure the flip-to-wrong (CORRUPTION) rate per layer.

  vision items (V-only correct): forward V-only, patch last-tok@L <- h_VT  -> corruption = caption's causal harm to vision
  text   items (T-only correct): forward T-only, patch last-tok@L <- h_VT  -> corruption = image's causal harm to text

Matched null: inject a RANDOM OTHER item's h_VT (same layer, same magnitude scale,
wrong content) -> separates "irrelevant-modality-specific corruption" from "any
perturbation corrupts". Prediction: vision corruption >> text corruption (fragility
asymmetry), and real >> null on vision.

GPU. Reuses 06c build_messages(kind) + SetActivationHook (canonical_eval).

    CUDA_VISIBLE_DEVICES=0 python scripts/42_causal_robustness_patch.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from moground.models import load_qwen_vl
from moground.steering import get_letter_token_ids
from moground.canonical_eval import SetActivationHook

# reuse 06c's exact kind-specific message builder (so patched state matches collection)
_spec = importlib.util.spec_from_file_location("_c6", Path(__file__).parent / "06c_collect_dm_counterfactual_activations.py")
_c6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c6)  # type: ignore
build_messages = _c6.build_messages


@torch.inference_mode()
def patched_pred(model, processor, row, kind, target, layer_idx, letter_ids, max_pixels):
    msgs = build_messages(row, kind, max_pixels)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    if image_inputs is not None and len(image_inputs) == 0:
        image_inputs = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
    hook = SetActivationHook(target); hook.last_idx = last_idx
    layer = model.model.language_model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    try:
        out = model(**inputs)
    finally:
        handle.remove()
    k = len(row["options"])
    return int(out.logits[0, last_idx][letter_ids[:k]].argmax().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual")
    ap.add_argument("--rows-jsonl", default=None,
                    help="row source for questions/options/image_path; default = D_M splits. "
                         "For assembled (non-saturated text) use data/dm/merged_aokvqa_racehigh/dm_all.jsonl")
    ap.add_argument("--layers", default="5,10,16,20,23,26,28,31")
    ap.add_argument("--n-per-label", type=int, default=400)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/diagnostics/causal_robustness/qwen.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    st = dr / "activations" / "qwen" / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")   # (N,3,Lyr,d) [VT,V,T]
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    # join rows (question/options/image_path) — from --rows-jsonl if given, else D_M splits
    rowmap = {}
    if args.rows_jsonl:
        for l in Path(args.rows_jsonl).read_text().splitlines():
            if l.strip():
                r = json.loads(l); rowmap[r["candidate_id"]] = r
    else:
        for sp in ("train", "val", "test"):
            p = dm / "splits" / f"{sp}.jsonl"
            if p.exists():
                for l in p.read_text().splitlines():
                    if l.strip():
                        r = json.loads(l); rowmap[r["candidate_id"]] = r
    ci = np.array([r["correct_index"] for r in idx])
    lab = np.array([r["label"] for r in idx])
    v_ok = ll[:, 1].argmax(1) == ci      # V-only correct
    t_ok = ll[:, 2].argmax(1) == ci      # T-only correct
    rng = np.random.default_rng(args.seed)

    def subsample(mask):
        ix = np.where(mask & np.array([r["candidate_id"] in rowmap for r in idx]))[0]
        rng.shuffle(ix); return ix[:args.n_per_label]

    vis_ix = subsample((lab == "vision") & v_ok)   # vision, V-only correct
    txt_ix = subsample((lab == "text") & t_ok)     # text, T-only correct
    print(f"vision(V-correct) n={len(vis_ix)}  text(T-correct) n={len(txt_ix)}")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device, n_letters=6)
    layers = [int(x) for x in args.layers.split(",")]
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, ix, kind in [("vision", vis_ix, "V"), ("text", txt_ix, "T")]:
        # null source: a shuffled assignment of OTHER items' h_VT within the same group
        perm = ix.copy(); rng.shuffle(perm)
        perm = np.where(perm == ix, np.roll(perm, 1), perm)   # avoid self
        for L in layers:
            corr_real = corr_null = 0
            for j, i in enumerate(tqdm(ix, desc=f"{name} L{L}", leave=False)):
                row = rowmap[idx[i]["candidate_id"]]
                tgt_real = torch.tensor(acts[i, 0, L].astype(np.float32), device=args.device)
                tgt_null = torch.tensor(acts[perm[j], 0, L].astype(np.float32), device=args.device)
                if patched_pred(model, processor, row, kind, tgt_real, L, letter_ids, args.max_pixels) != ci[i]:
                    corr_real += 1
                if patched_pred(model, processor, row, kind, tgt_null, L, letter_ids, args.max_pixels) != ci[i]:
                    corr_null += 1
            rec = {"label": name, "layer": L, "n": int(len(ix)),
                   "corruption_real": corr_real / len(ix), "corruption_null": corr_null / len(ix)}
            results.append(rec)
            print(f"{name:7} L{L:>2}: corruption real {rec['corruption_real']:.3f}  null {rec['corruption_null']:.3f}  "
                  f"(Δ {rec['corruption_real']-rec['corruption_null']:+.3f})")
    out_path.write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
