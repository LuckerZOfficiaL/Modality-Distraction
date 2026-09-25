"""Loader for Qwen2.5-VL-3B-Instruct + helpers for MCQ scoring."""
from __future__ import annotations

import re
from functools import lru_cache

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


@lru_cache(maxsize=1)
def load_qwen_vl(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
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
        parts.append({"type": "image", "image": image_path})
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return parse_letter(decoded)
