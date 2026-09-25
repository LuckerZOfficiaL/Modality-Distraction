"""LLaVA-NeXT-LLaMA3-8B steering helpers (mirror of steering.py).

Provides:
  build_messages(row)   -> LLaVA chat-template-ready messages
  score_row(...)        -> drop-in for steering.score_row, using LLaVA processor
  SAEHookSparsify(...)  -> SAEHook variant using EleutherAI sparsify.Sae
  load_oracle_passed_llavanext(...) -> uses pass_llavanext/{train,val}.jsonl + splits/test.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from sae_steering.steering import SteeringHook


def build_messages(row: dict) -> list[dict]:
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Caption: {row['caption_for_filter']}"},
            {"type": "image"},
            {"type": "text", "text": (
                f"Question: {row['question']}\n{opts}\n\n"
                "Reply with exactly one letter (A, B, C, or D) and nothing else."
            )},
        ],
    }]


@torch.inference_mode()
def score_row(model, processor, row, hook, layer_idx, letter_token_ids):
    msgs = build_messages(row)
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    img = Image.open(row["image_path"]).convert("RGB")
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
    handle = None
    if hook is not None:
        hook.last_idx = last_idx
        layer = model.model.language_model.layers[layer_idx]
        handle = layer.register_forward_hook(hook)
    try:
        out = model(**inputs)
    finally:
        if handle is not None:
            handle.remove()
            hook.last_idx = None
    logits = out.logits[0, last_idx]
    return int(logits[letter_token_ids].argmax().item())


class SAEHookSparsify(SteeringHook):
    """SAEHook variant using sparsify.Sae's encode/decode interface.

    Multiplicatively scale top-N vision and top-N text features in z-space.
    Because sparsify Sae uses true Top-K (after the perturbation we re-decode
    by injecting modified top_acts at the same indices), we treat the sparse
    representation directly: scan top_indices for matches against vision_idx /
    text_idx and scale top_acts there. Features not in the active top-K on
    this row contribute zero — same as Qwen pipeline.
    """

    def __init__(self, sae, vision_idx: torch.Tensor, text_idx: torch.Tensor, alpha: float):
        super().__init__()
        self.sae = sae
        self.v_idx = vision_idx  # 1D long tensor on device
        self.t_idx = text_idx
        self.alpha = alpha
        self.delta_norms: list[float] = []
        self.zero_count: int = 0

    def edit(self, h_at_last):
        if self.alpha == 0.0:
            return None
        x = h_at_last.to(torch.float32)
        enc = self.sae.encode(x)
        top_acts = enc.top_acts                  # [B, k]
        top_indices = enc.top_indices            # [B, k]
        x_hat = self.sae.decode(top_acts, top_indices)

        # build modified top_acts matching the same top_indices
        mod = top_acts.clone()
        # for each batch row, find positions whose feature id is in v_idx / t_idx
        v_mask = torch.isin(top_indices, self.v_idx)
        t_mask = torch.isin(top_indices, self.t_idx)
        mod = torch.where(v_mask, mod * (1.0 + self.alpha), mod)
        mod = torch.where(t_mask, mod * (1.0 - self.alpha), mod)
        x_hat2 = self.sae.decode(mod, top_indices)
        delta = x_hat2 - x_hat
        dn = float(delta.norm().item())
        self.delta_norms.append(dn)
        if dn == 0.0:
            self.zero_count += 1
        return h_at_last + delta.to(h_at_last.dtype)


def load_oracle_passed_llavanext(data_root: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> list[dict]:
    """Pool used for steering on LLaVA-NeXT.

    For train/val we use the LLaVA-NeXT pass set (pass_llavanext/{train,val}.jsonl)
    so probe/contrastive comparison is apple-to-apple with where the probe was
    trained. test split is never VLM-filtered, taken from data/dm/splits/test.jsonl.
    """
    dm_dir = data_root / "dm"
    rows = []
    for sub in splits:
        if sub == "test":
            path = dm_dir / "splits" / "test.jsonl"
        else:
            path = dm_dir / "pass_llavanext" / f"{sub}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["__split"] = sub
            rows.append(r)
    return rows
