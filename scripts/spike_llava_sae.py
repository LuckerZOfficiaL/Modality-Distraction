"""Spike: load LLaVA-NeXT-LLaMA3-8B + lmms-lab SAE @ L24, run a few D_M items.

Goal: confirm we can hook the L24 residual, push activations through their SAE,
and get sensible top-K activations (sparsity, magnitudes, feature identities).

Outputs (printed):
  - SAE z stats per item (active count, max |z|, top-5 feature ids)
  - Reconstruction MSE on the same activations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from sparsify import Sae
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


VLM_ID = "llava-hf/llama3-llava-next-8b-hf"
SAE_ID = "lmms-lab/llama3-llava-next-8b-hf-sae-131k"
HOOKPOINT = "model.layers.24"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--dm-split", default="data/dm/splits/test.jsonl")
    args = ap.parse_args()

    print(f"loading SAE {SAE_ID}@{HOOKPOINT}")
    sae = Sae.load_from_hub(SAE_ID, hookpoint=HOOKPOINT).to(args.device)
    sae.eval()
    print(f"  d_in={sae.d_in} num_latents={sae.num_latents} k={sae.cfg.k}")

    print(f"loading VLM {VLM_ID}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        VLM_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(VLM_ID)

    # locate the L24 residual stream module
    layer = model.model.language_model.layers[24]
    print(f"  layer module: {type(layer).__name__}")

    captured: dict = {}
    def hook(module, inputs, outputs):
        # transformer block output is a tuple; first is the hidden state
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured["h"] = h.detach()
    handle = layer.register_forward_hook(hook)

    spike_dir = Path(__file__).resolve().parent.parent
    items = [json.loads(l) for l in (spike_dir / args.dm_split).read_text().splitlines() if l.strip()][: args.n]

    for it in items:
        img = Image.open(it["image_path"]).convert("RGB")
        opts = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", it["options"]))
        cap = it.get("caption_for_filter") or it.get("original_caption", "")
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<image>\nCaption: {cap}\nQuestion: {it['question']}\n{opts}\n\n"
            f"Reply with exactly one letter (A, B, C, or D).<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(args.device)

        with torch.no_grad():
            _ = model(**inputs)
        h = captured["h"]
        h_last = h[:, -1, :].to(torch.float32)
        with torch.no_grad():
            enc = sae.encode(h_last)  # returns (top_acts, top_indices)
            top_acts, top_indices = enc.top_acts, enc.top_indices
            x_hat = sae.decode(top_acts, top_indices)
        mse = (h_last - x_hat).pow(2).mean().item()
        nz = top_acts.numel()
        vals, local = top_acts.flatten().topk(5)
        topk_feat_ids = top_indices.flatten()[local].tolist()
        print(f"\n[{it['candidate_id']}] label={it['label']}")
        print(f"  h_last shape={tuple(h_last.shape)}  mean|h|={h_last.abs().mean().item():.3f}")
        print(f"  active={nz}/{sae.num_latents}  max|z|={top_acts.abs().max().item():.3f}  recon MSE={mse:.4f}")
        print(f"  top-5 feature ids={topk_feat_ids}")

    handle.remove()


if __name__ == "__main__":
    main()
