"""Step 6a: collect Qwen2.5-VL hidden states on D_M (train-pass + val-pass + test).

For each row, format the same MCQ prompt the VLM saw at filter time
(image + caption_for_filter + question + lettered options + 'Answer:'),
forward through the model, hook ALL 36 decoder layers, extract the
last-non-pad-input-token hidden state per layer.

Outputs:
  data/activations/dm/activations.npy   # shape (N, n_layers, d_model), float16
  data/activations/dm/index.jsonl       # one row per N: candidate_id, label, split, correct_index

Splits combined:
  - D_M,train-pass  (data/dm/pass/train.jsonl)
  - D_M,val-pass    (data/dm/pass/val.jsonl)
  - D_M,test        (data/dm/splits/test.jsonl)  — never VLM-filtered

Step 7 (probes) reads activations + index and trains per-layer probes
on rows whose split is train-pass or val-pass.

Resume-safe: if outputs exist with the right N, skip; otherwise re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from moground.models import load_qwen_vl
from moground.activations import ActivationExtractor


def build_prompt_messages(row: dict, include_caption: bool = True,
                          max_pixels: int | None = None) -> list[dict]:
    letters = ["A", "B", "C", "D"]
    opts_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    parts = []
    if include_caption:
        parts.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
    img = {"type": "image", "image": row["image_path"]}
    if max_pixels is not None:
        img["max_pixels"] = max_pixels
    parts.append(img)
    parts.append({"type": "text", "text": (
        f"Question: {row['question']}\n{opts_str}\n\n"
        "Reply with exactly one letter (A, B, C, or D) and nothing else."
    )})
    return [{"role": "user", "content": parts}]


def load_pool(dm_dir: Path) -> list[dict]:
    rows = []
    for split, path in [
        ("train_pass", dm_dir / "pass" / "train.jsonl"),
        ("val_pass",   dm_dir / "pass" / "val.jsonl"),
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-caption", action="store_true", help="omit caption from prompt (control)")
    ap.add_argument("--out-suffix", default="", help="appended to output dir name")
    ap.add_argument("--max-pixels", type=int, default=1024*1024,
                    help="cap each image's pixel count (Qwen processor auto-downscales). "
                         "Default ~1M. Lower to ~512*512 if OOM persists.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    out_dir = Path(cfg["paths"]["data_root"]) / "activations" / "qwen" / (f"dm{args.out_suffix}" if args.out_suffix else "dm")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pool(dm_dir)
    print(f"D_M activation pool: {len(rows)} rows")

    n_layers = 36
    layer_indices = list(range(n_layers))

    model, processor = load_qwen_vl(device=args.device)
    d_model = model.config.text_config.hidden_size
    print(f"hidden_size = {d_model}, layers = {n_layers}")

    # Resume: if a prior run wrote partial outputs, load them and skip done rows.
    acts_path = out_dir / "activations.npy"
    idx_path = out_dir / "index.jsonl"
    done_ids: set[str] = set()
    if acts_path.exists() and idx_path.exists():
        prior = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        # row order must align with rows by candidate_id
        prior_by_cid = {r["candidate_id"]: i for i, r in enumerate(prior)}
        if prior and prior_by_cid.keys() == {r["candidate_id"] for r in rows[:len(prior)]} \
                and all(rows[i]["candidate_id"] == prior[i]["candidate_id"] for i in range(len(prior))):
            print(f"resume: {len(prior)} rows already processed (prefix-aligned)")
            acts_prev = np.load(acts_path)
            acts = np.zeros((len(rows), n_layers, d_model), dtype=np.float16)
            acts[:len(prior)] = acts_prev[:len(prior)]
            index_rows = list(prior)
            done_ids = set(prior_by_cid.keys())
        else:
            print("WARN: existing output doesn't align with current pool prefix — starting fresh")
            acts = np.zeros((len(rows), n_layers, d_model), dtype=np.float16)
            index_rows = []
    else:
        acts = np.zeros((len(rows), n_layers, d_model), dtype=np.float16)
        index_rows = []

    def flush():
        np.save(acts_path, acts)
        with idx_path.open("w") as f:
            for r in index_rows:
                f.write(json.dumps(r) + "\n")

    extractor = ActivationExtractor(model, layer_indices)
    pbar = tqdm(total=len(rows), initial=len(done_ids), desc="dm activations")
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        if all(r["candidate_id"] in done_ids for r in batch):
            continue
        messages_batch = [build_prompt_messages(r, include_caption=not args.no_caption,
                                                max_pixels=args.max_pixels) for r in batch]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with extractor, torch.inference_mode():
            _ = model(**inputs)
            last = extractor.last_token_states(inputs.attention_mask)  # [B, n_layers, d]
        acts[start:start + len(batch)] = last.to(torch.float16).cpu().numpy()
        for r in batch:
            index_rows.append({
                "candidate_id": r["candidate_id"],
                "label": r["label"],
                "split": r["__split"],
                "correct_index": r["correct_index"],
            })
            done_ids.add(r["candidate_id"])
        pbar.update(len(batch))
        # checkpoint every ~100 batches
        if (start // args.batch_size + 1) % 100 == 0:
            flush()
    pbar.close()

    flush()
    print(f"saved activations -> {acts_path}  shape={acts.shape}")
    print(f"saved index       -> {idx_path}  ({len(index_rows)} rows)")


if __name__ == "__main__":
    main()
