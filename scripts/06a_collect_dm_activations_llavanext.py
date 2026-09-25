"""LLaVA-NeXT-LLaMA3-8B variant of 06a: collect last-input-token hidden states
on D_M (train-pass + val-pass + test) across all 32 LM layers.

Outputs:
  data/activations/llavanext/dm/activations.npy   # (N, 32, 4096) float16
  data/activations/llavanext/dm/index.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from sae_steering.activations import ActivationExtractor
from sae_steering.models_llavanext import load_llavanext


def build_prompt(row: dict, processor, include_caption: bool = True) -> tuple[str, Image.Image]:
    letters = ["A", "B", "C", "D"]
    opts_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    parts = []
    if include_caption:
        parts.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
    parts.append({"type": "image"})
    parts.append({"type": "text", "text": (
        f"Question: {row['question']}\n{opts_str}\n\n"
        "Reply with exactly one letter (A, B, C, or D) and nothing else."
    )})
    messages = [{"role": "user", "content": parts}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    img = Image.open(row["image_path"]).convert("RGB")
    return prompt, img


def load_pool(dm_dir: Path) -> list[dict]:
    rows = []
    for split, path in [
        ("train_pass", dm_dir / "pass_llavanext" / "train.jsonl"),
        ("val_pass",   dm_dir / "pass_llavanext" / "val.jsonl"),
        ("test",       dm_dir / "splits" / "test.jsonl"),
    ]:
        if not path.exists():
            continue
        for l in path.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            r["__split"] = split
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    out_dir = Path(cfg["paths"]["data_root"]) / "activations" / "llavanext" / "dm"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pool(dm_dir)
    print(f"D_M activation pool: {len(rows)} rows")

    model, processor = load_llavanext(device=args.device)
    n_layers = len(model.model.language_model.layers)
    d_model = model.config.text_config.hidden_size
    print(f"hidden_size={d_model}, layers={n_layers}")

    layer_indices = list(range(n_layers))
    acts = np.zeros((len(rows), n_layers, d_model), dtype=np.float16)
    index_rows = []

    extractor = ActivationExtractor(model, layer_indices)
    for start in tqdm(range(0, len(rows), args.batch_size), desc="dm activations"):
        batch = rows[start:start + args.batch_size]
        prompts, images = zip(*[build_prompt(r, processor) for r in batch])
        inputs = processor(text=list(prompts), images=list(images),
                           padding=True, return_tensors="pt").to(model.device)
        with extractor, torch.inference_mode():
            _ = model(**inputs)
            last = extractor.last_token_states(inputs.attention_mask)
        acts[start:start + len(batch)] = last.to(torch.float16).cpu().numpy()
        for r in batch:
            index_rows.append({
                "candidate_id": r["candidate_id"],
                "label": r["label"],
                "split": r["__split"],
                "correct_index": r["correct_index"],
            })

    np.save(out_dir / "activations.npy", acts)
    with (out_dir / "index.jsonl").open("w") as f:
        for r in index_rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved -> {out_dir}/activations.npy  shape={acts.shape}")


if __name__ == "__main__":
    main()
