"""Smoke test: do per-item modality instructions ("use the image"/"use the caption")
yield more disjoint SAE features at L24 than vanilla prompts?

Compares two prompt forms on the same 224 LLaVA-NeXT pass items:
  baseline : {Caption} <image> Question:...
  guided   : {Caption} <image> [INSTRUCTION] Question:...
   where INSTRUCTION =
     vision-grounded items -> "Answer using information from the image."
     text-grounded items   -> "Answer using information from the caption."

Outputs:
  data/features/llavanext/layer24_guided/top_features.json
  prints jaccard V×T mean for both prompt forms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sparsify import Sae
from tqdm import tqdm

from sae_steering.models_llavanext import load_llavanext


def build_prompt(row, processor, *, guided: bool):
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    parts = [{"type": "text", "text": f"Caption: {row['caption_for_filter']}"},
             {"type": "image"}]
    if guided:
        instr = ("Answer using information from the image."
                 if row["label"] == "vision"
                 else "Answer using information from the caption.")
        parts.append({"type": "text", "text": instr})
    parts.append({"type": "text", "text": (
        f"Question: {row['question']}\n{opts}\n\n"
        "Reply with exactly one letter (A, B, C, or D) and nothing else.")})
    msgs = [{"role": "user", "content": parts}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def collect_z(model, processor, sae, rows, layer_idx, *, guided, device, top_n=20):
    captured = {}
    def hook(module, inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["h"] = h.detach()
    handle = model.model.language_model.layers[layer_idx].register_forward_hook(hook)

    n = len(rows)
    z_dense = np.zeros((n, sae.num_latents), dtype=np.float32)
    try:
        for i, r in enumerate(tqdm(rows, desc=f"guided={guided}")):
            prompt = build_prompt(r, processor, guided=guided)
            img = Image.open(r["image_path"]).convert("RGB")
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
            with torch.inference_mode():
                _ = model(**inputs)
            last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
            h_last = captured["h"][:, last_idx, :].to(torch.float32)
            with torch.no_grad():
                enc = sae.encode(h_last)
                acts = enc.top_acts.cpu().numpy()[0]
                idxs = enc.top_indices.cpu().numpy()[0]
            z_dense[i, idxs] = acts
    finally:
        handle.remove()

    labels = np.array([0 if r["label"] == "vision" else 1 for r in rows])
    z_v = z_dense[labels == 0]; z_t = z_dense[labels == 1]
    rho = z_v.mean(0) - z_t.mean(0)
    var = z_dense.var(0)
    alive = int((var > 1e-8).sum())
    alive_idx = np.where(var > 1e-8)[0]
    top_v = alive_idx[np.argsort(-rho[alive_idx])[:top_n]].tolist()
    top_t = alive_idx[np.argsort(rho[alive_idx])[:top_n]].tolist()
    fire_v_sets = [set(np.where(z_dense[:, k] > 0)[0].tolist()) for k in top_v]
    fire_t_sets = [set(np.where(z_dense[:, k] > 0)[0].tolist()) for k in top_t]
    jacc = []
    for a in fire_v_sets:
        for b in fire_t_sets:
            u = a | b
            jacc.append(len(a & b) / len(u) if u else 0.0)
    return {
        "alive": alive,
        "top_v": top_v, "top_t": top_t,
        "top_v_rho": [float(rho[k]) for k in top_v],
        "top_t_rho": [float(rho[k]) for k in top_t],
        "jaccard_mean": float(np.mean(jacc)) if jacc else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=24)
    args = ap.parse_args()

    rows = []
    for sp in ("train", "val"):
        for l in open(f"data/dm/pass_llavanext/{sp}.jsonl"):
            r = json.loads(l); rows.append(r)
    print(f"loaded {len(rows)} pass-set rows")

    sae = Sae.load_from_hub("lmms-lab/llama3-llava-next-8b-hf-sae-131k",
                            hookpoint=f"model.layers.{args.layer}").to(args.device).eval()
    model, processor = load_llavanext(device=args.device)

    print("\n== baseline prompt ==")
    base = collect_z(model, processor, sae, rows, args.layer, guided=False, device=args.device)
    print(f"alive={base['alive']}  jaccard V×T = {base['jaccard_mean']:.3f}")

    print("\n== guided prompt ==")
    gd = collect_z(model, processor, sae, rows, args.layer, guided=True, device=args.device)
    print(f"alive={gd['alive']}  jaccard V×T = {gd['jaccard_mean']:.3f}")

    out = {"baseline": base, "guided": gd}
    Path("data/features/llavanext/layer24_guided").mkdir(parents=True, exist_ok=True)
    Path("data/features/llavanext/layer24_guided/smoke.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote data/features/llavanext/layer24_guided/smoke.json")
    print(f"\ndelta jaccard: {gd['jaccard_mean'] - base['jaccard_mean']:+.3f} "
          f"(lower = more disjoint)")


if __name__ == "__main__":
    main()
