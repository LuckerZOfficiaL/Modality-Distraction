"""Step 6c: collect Qwen2.5-VL activations under THREE input conditions per item.

For each pool row, run the model three times:
  - VT: image + caption + question + options  (current default)
  - V : image          + question + options   (caption dropped)
  - T :         caption + question + options  (image dropped)

Store last-non-pad-input-token hidden states at every layer for all three
conditions, plus the letter-token logits at the last position (so we can
filter the V/fail pool by P(correct|VT) margin downstream).

Output (per dm-dir):
  data/activations/qwen/<out-name>/activations.npy   (N, 3, n_layers, d_model) float16
  data/activations/qwen/<out-name>/letter_logits.npy (N, 3, 4) float16
  data/activations/qwen/<out-name>/index.jsonl       N rows: candidate_id, label, split, correct_index

Condition order in the leading "3" axis: [VT, V, T].

Pool: train_pass + val_pass + all val (oracle survivors) + distraction. Test
excluded per the 2026-06-04 data policy.

This is the data layer for causal steering: per-item counterfactual directions
h_VT(i) − h_V(i) and h_VT(i) − h_T(i), and confidence-filtered V/fail.

Resume-safe via index.jsonl prefix alignment (same as 06a).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from moground.models import load_qwen_vl
from moground.activations import ActivationExtractor
from moground.steering import get_letter_token_ids


def build_messages(row: dict, kind: str, max_pixels: int | None = None) -> list[dict]:
    """kind ∈ {VT, V, T}: which inputs to include."""
    letters = ["A", "B", "C", "D"]
    opts_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    parts: list[dict] = []
    if kind in ("VT", "T"):
        parts.append({"type": "text", "text": f"Caption: {row['caption_for_filter']}"})
    if kind in ("VT", "V"):
        img = {"type": "image", "image": row["image_path"]}
        if max_pixels is not None:
            img["max_pixels"] = max_pixels
        parts.append(img)
    prefix = os.environ.get("DM_PROMPT_PREFIX", "")
    parts.append({"type": "text", "text": (
        (prefix + "\n\n" if prefix else "")
        + f"Question: {row['question']}\n{opts_str}\n\n"
        "Reply with exactly one letter (A, B, C, or D) and nothing else."
    )})
    return [{"role": "user", "content": parts}]


def load_pool(dm_dir: Path, distraction_path: Path | None) -> list[dict]:
    """train_pass + val_pass + all val (oracle) + distraction. Test excluded."""
    rows: list[dict] = []
    seen: set[str] = set()
    def add(r, split):
        if r["candidate_id"] in seen:
            return
        seen.add(r["candidate_id"])
        r["__split"] = split
        rows.append(r)

    # train_pass + val_pass (Qwen 3-pass survivors)
    for split, path in [
        ("train_pass", dm_dir / "pass" / "train.jsonl"),
        ("val_pass",   dm_dir / "pass" / "val.jsonl"),
    ]:
        if not path.exists():
            continue
        for l in path.read_text().splitlines():
            if l.strip():
                add(json.loads(l), split)

    # all val (so we have counterfactual acts for val items that aren't val_pass — they're
    # still in the unseen steering pool)
    val_path = dm_dir / "splits" / "val.jsonl"
    if val_path.exists():
        for l in val_path.read_text().splitlines():
            if l.strip():
                add(json.loads(l), "val")

    # train rows that aren't train_pass — for steering V/fail pool membership (train V+T-fail).
    train_path = dm_dir / "splits" / "train.jsonl"
    if train_path.exists():
        for l in train_path.read_text().splitlines():
            if l.strip():
                add(json.loads(l), "train")

    # distraction pool rows (their cids usually collide with train/val above; add() skips dups)
    if distraction_path and distraction_path.exists():
        for l in distraction_path.read_text().splitlines():
            if l.strip():
                add(json.loads(l), "distraction")

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl",
                    help="optional; cids that collide with train/val rows are skipped")
    ap.add_argument("--out-name", default="dm_multidomain_v1_counterfactual",
                    help="subdir under data/activations/qwen/")
    ap.add_argument("--rows-jsonl", default=None,
                    help="if set, load rows from this JSONL instead of the default pool. "
                         "Use this to extend the cf-store with new cids (e.g. test split).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="effective B; each row produces 3 forward passes")
    ap.add_argument("--max-pixels", type=int, default=1024*1024)
    ap.add_argument("--checkpoint-every", type=int, default=50, help="batches between flushes")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    out_dir = Path(cfg["paths"]["data_root"]) / "activations" / "qwen" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rows_jsonl is not None:
        rows = [json.loads(l) for l in Path(args.rows_jsonl).read_text().splitlines() if l.strip()]
        for r in rows:
            r.setdefault("__split", "custom")
        print(f"counterfactual activation pool (from {args.rows_jsonl}): {len(rows)} rows")
    else:
        dist_path = Path(args.distraction_pool) if args.distraction_pool else None
        rows = load_pool(dm_dir, dist_path)
        print(f"counterfactual activation pool: {len(rows)} rows "
              f"(train_pass + val_pass + val + train + distraction, test excluded)")
    from collections import Counter
    print(f"  by split: {dict(Counter(r['__split'] for r in rows))}")

    n_layers = 36
    layer_indices = list(range(n_layers))
    n_conditions = 3   # VT, V, T
    cond_names = ["VT", "V", "T"]

    model, processor = load_qwen_vl(device=args.device)
    d_model = model.config.text_config.hidden_size
    letter_ids = get_letter_token_ids(processor, args.device)
    print(f"hidden_size = {d_model}, layers = {n_layers}, conditions = {cond_names}")

    acts_path = out_dir / "activations.npy"
    logits_path = out_dir / "letter_logits.npy"
    idx_path = out_dir / "index.jsonl"

    done_ids: set[str] = set()
    if acts_path.exists() and idx_path.exists():
        prior = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        prior_cids = [r["candidate_id"] for r in prior]
        if prior_cids == [r["candidate_id"] for r in rows[:len(prior_cids)]]:
            print(f"resume: {len(prior)} rows already processed (prefix-aligned)")
            acts_prev = np.load(acts_path)
            logits_prev = np.load(logits_path) if logits_path.exists() else \
                          np.zeros((len(rows), n_conditions, 4), dtype=np.float16)
            acts = np.zeros((len(rows), n_conditions, n_layers, d_model), dtype=np.float16)
            logits = np.zeros((len(rows), n_conditions, 4), dtype=np.float16)
            acts[:len(prior)] = acts_prev[:len(prior)]
            logits[:len(prior)] = logits_prev[:len(prior)]
            index_rows = list(prior)
            done_ids = set(prior_cids)
        else:
            print("WARN: existing output doesn't align — starting fresh")
            acts = np.zeros((len(rows), n_conditions, n_layers, d_model), dtype=np.float16)
            logits = np.zeros((len(rows), n_conditions, 4), dtype=np.float16)
            index_rows = []
    else:
        acts = np.zeros((len(rows), n_conditions, n_layers, d_model), dtype=np.float16)
        logits = np.zeros((len(rows), n_conditions, 4), dtype=np.float16)
        index_rows = []

    def flush():
        np.save(acts_path, acts)
        np.save(logits_path, logits)
        with idx_path.open("w") as f:
            for r in index_rows:
                f.write(json.dumps(r) + "\n")

    extractor = ActivationExtractor(model, layer_indices)
    pbar = tqdm(total=len(rows), initial=len(done_ids), desc="dm counterfactual")
    batch_count = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        if all(r["candidate_id"] in done_ids for r in batch):
            continue
        for ci, cond in enumerate(cond_names):
            msgs_batch = [build_messages(r, cond, args.max_pixels) for r in batch]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in msgs_batch]
            image_inputs, video_inputs = process_vision_info(msgs_batch)
            if image_inputs is not None and len(image_inputs) == 0:
                image_inputs = None
            inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            with extractor, torch.inference_mode():
                out = model(**inputs)
                last = extractor.last_token_states(inputs.attention_mask)  # [B, n_layers, d]
            acts[start:start + len(batch), ci] = last.to(torch.float16).cpu().numpy()
            last_idx = inputs.attention_mask.sum(dim=1) - 1  # [B]
            for bi, li in enumerate(last_idx.tolist()):
                ll = out.logits[bi, li, :][letter_ids].float().cpu().numpy()
                logits[start + bi, ci] = ll.astype(np.float16)
        for bi, r in enumerate(batch):
            index_rows.append({
                "candidate_id": r["candidate_id"],
                "label": r["label"],
                "split": r["__split"],
                "correct_index": r["correct_index"],
            })
            done_ids.add(r["candidate_id"])
        pbar.update(len(batch))
        batch_count += 1
        if batch_count % args.checkpoint_every == 0:
            flush()
    pbar.close()

    flush()
    print(f"saved activations    -> {acts_path}  shape={acts.shape}")
    print(f"saved letter_logits  -> {logits_path}  shape={logits.shape}")
    print(f"saved index          -> {idx_path}  ({len(index_rows)} rows)")

    # quick sanity: per-condition argmax accuracy by label
    preds = logits.argmax(axis=-1)  # (N, 3)
    correct = np.array([r["correct_index"] for r in index_rows])
    labels = np.array([r["label"] for r in index_rows])
    print("\nSanity: argmax accuracy by (condition, label)")
    for ci, cond in enumerate(cond_names):
        for lbl in ("vision", "text"):
            mask = labels == lbl
            acc = (preds[mask, ci] == correct[mask]).mean() if mask.sum() else 0.0
            print(f"  {cond} | {lbl}: {acc:.3f} (n={int(mask.sum())})")


if __name__ == "__main__":
    main()
