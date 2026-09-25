"""Step 6d: collect the h_Q (question + options only) condition.

The pure question-encoding representation: no image, no caption -- just the
question and answer options (same instruction line as the cf-store). This is
the substrate for the question-type confounder analysis (scripts/21): the part
of the label that is decodable from the question text alone.

Aligned 1:1 (by candidate_id, same order) to an existing cf-store index so the
two stores join per item. Single condition, so the output is 2-D in the
layer/condition sense:
  data/activations/qwen/<out-name>/activations.npy    (N, 36, 2048) float16
  data/activations/qwen/<out-name>/letter_logits.npy  (N, 4)        float16
  data/activations/qwen/<out-name>/index.jsonl         N rows (copied from --align-index)

The canonical cf-store is NOT modified. Resume-safe via index prefix alignment.

    CUDA_VISIBLE_DEVICES=0 python scripts/06d_collect_hq_activations.py
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

from moground.activations import ActivationExtractor
from moground.models import load_qwen_vl
from moground.steering import get_letter_token_ids


def build_q_messages(row: dict) -> list[dict]:
    """Question + options only (no image, no caption). Mirrors the cf-store
    instruction line so the question-encoding is collected under the same prompt."""
    letters = ["A", "B", "C", "D", "E", "F"][:len(row["options"])]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    n = len(letters)
    ll = f"{letters[0]} or {letters[1]}" if n == 2 else ", ".join(letters[:-1]) + f", or {letters[-1]}"
    return [{"role": "user", "content": [{"type": "text", "text": (
        f"Question: {row['question']}\n{opts}\n\n"
        f"Reply with exactly one letter ({ll}) and nothing else."
    )}]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--align-index",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual/index.jsonl",
                    help="cf-store index to align cids/order to (so the stores join 1:1)")
    ap.add_argument("--distraction-pool", default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--out-name", default="dm_multidomain_v1_hq")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    out_dir = Path(cfg["paths"]["data_root"]) / "activations" / "qwen" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = [json.loads(l) for l in Path(args.align_index).read_text().splitlines() if l.strip()]
    order = [r["candidate_id"] for r in ref]

    # cid -> full row (need question + options) from splits + distraction pool
    rowmap: dict[str, dict] = {}
    for sp in ("train", "val", "test"):
        p = dm_dir / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); rowmap.setdefault(r["candidate_id"], r)
    dp = Path(args.distraction_pool)
    if dp.exists():
        for l in dp.read_text().splitlines():
            if l.strip():
                r = json.loads(l); rowmap.setdefault(r["candidate_id"], r)

    missing = [c for c in order if c not in rowmap]
    if missing:
        print(f"WARN: {len(missing)}/{len(order)} cids have no full row (skipped); e.g. {missing[:3]}")
    # keep ref rows whose full row exists, preserve cf-store order
    ref_kept = [r for r in ref if r["candidate_id"] in rowmap]
    rows = [rowmap[r["candidate_id"]] for r in ref_kept]
    print(f"aligned pool: {len(rows)}/{len(order)} rows (cf-store order)")

    n_layers = 36
    model, processor = load_qwen_vl(device=args.device)
    d_model = model.config.text_config.hidden_size
    letter_ids = get_letter_token_ids(processor, args.device)  # 4 letters (all D_M are 4-opt)

    acts_path = out_dir / "activations.npy"
    logits_path = out_dir / "letter_logits.npy"
    idx_path = out_dir / "index.jsonl"

    done = 0
    if acts_path.exists() and idx_path.exists():
        prior = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        pc = [r["candidate_id"] for r in prior]
        if pc == [r["candidate_id"] for r in ref_kept[:len(pc)]]:
            done = len(prior)
            acts = np.load(acts_path); logits = np.load(logits_path)
            print(f"resume: {done} rows already done (prefix-aligned)")
        else:
            print("WARN: existing output misaligned — starting fresh")
            acts = np.zeros((len(rows), n_layers, d_model), np.float16); logits = np.zeros((len(rows), 4), np.float16)
    else:
        acts = np.zeros((len(rows), n_layers, d_model), np.float16); logits = np.zeros((len(rows), 4), np.float16)

    def flush():
        np.save(acts_path, acts); np.save(logits_path, logits)
        with idx_path.open("w") as f:
            for r in ref_kept[:done]:
                f.write(json.dumps({"candidate_id": r["candidate_id"], "label": r["label"],
                                    "split": r["split"], "correct_index": r["correct_index"]}) + "\n")

    extractor = ActivationExtractor(model, list(range(n_layers)))
    pbar = tqdm(total=len(rows), initial=done, desc="h_Q")
    bc = 0
    for start in range(done, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        msgs = [build_q_messages(r) for r in batch]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        if image_inputs is not None and len(image_inputs) == 0:
            image_inputs = None
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with extractor, torch.inference_mode():
            out = model(**inputs)
            last = extractor.last_token_states(inputs.attention_mask)  # [B, n_layers, d]
        acts[start:start + len(batch)] = last.to(torch.float16).cpu().numpy()
        last_idx = inputs.attention_mask.sum(dim=1) - 1
        for bi, li in enumerate(last_idx.tolist()):
            logits[start + bi] = out.logits[bi, li, :][letter_ids].float().cpu().numpy().astype(np.float16)
        done = start + len(batch)
        pbar.update(len(batch)); bc += 1
        if bc % args.checkpoint_every == 0:
            flush()
    pbar.close()
    flush()
    print(f"saved {acts_path} {acts.shape}; {logits_path} {logits.shape}; {idx_path} ({done} rows)")

    # sanity: question-only argmax accuracy by label (how often the question alone gets it right)
    preds = logits.argmax(-1); correct = np.array([r["correct_index"] for r in ref_kept])
    labs = np.array([r["label"] for r in ref_kept])
    for lbl in ("vision", "text"):
        m = labs == lbl
        print(f"  h_Q argmax acc | {lbl}: {(preds[m]==correct[m]).mean():.3f} (n={int(m.sum())})")


if __name__ == "__main__":
    main()
