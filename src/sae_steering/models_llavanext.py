"""Loader for LLaVA-NeXT-LLaMA3-8B + MCQ scoring helper.

Mirrors models.py (Qwen) so behavioral filtering and downstream eval can target
either backbone with the same call signature: predict_mcq(model, processor,
question, options, image_path, caption) -> int | None.
"""
from __future__ import annotations

import re
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


MODEL_NAME = "llava-hf/llama3-llava-next-8b-hf"


@lru_cache(maxsize=1)
def load_llavanext(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def _build_messages(question: str, options: list[str], image_path: str | None, caption: str | None) -> list[dict]:
    letters = ["A", "B", "C", "D"]
    opts_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    parts = []
    if caption is not None:
        parts.append({"type": "text", "text": f"Caption: {caption}"})
    if image_path is not None:
        parts.append({"type": "image"})
    parts.append({
        "type": "text",
        "text": (
            f"Question: {question}\n{opts_str}\n\n"
            "Reply with exactly one letter (A, B, C, or D) and nothing else."
        ),
    })
    return [{"role": "user", "content": parts}]


_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_letter(text: str) -> int | None:
    m = _LETTER_RE.search(text)
    if not m:
        return None
    return ord(m.group(1).upper()) - ord("A")


@torch.inference_mode()
def predict_mcq(
    model,
    processor,
    question: str,
    options: list[str],
    image_path: str | None,
    caption: str | None,
    max_new_tokens: int = 8,
) -> int | None:
    messages = _build_messages(question, options, image_path, caption)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB") if image_path else None
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return parse_letter(decoded)
