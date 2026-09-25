"""Diagnose the V/fail vs V/pass confidence asymmetry.

For a sample of pool items, record the baseline letter-logit margins:
  - V/fail item: how far is the (wrong) top letter ABOVE the correct letter?
  - V/pass item: how far is the (correct) top letter ABOVE the next-best letter?

Hypothesis: V/fail margins are << V/pass margins. That would explain why any
perturbation (real OR random) recovers V/fail more than it breaks V/pass.

Output: stdout + data/diagnostics/letter_margins.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from moground.models import load_qwen_vl
from moground.steering import (
    baseline_rows_needed, build_messages, get_letter_token_ids,
    load_distraction_pool, load_oracle_passed, select_pool,
)
from qwen_vl_utils import process_vision_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--distraction-pool", default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--baseline-source",
                    default="data/steering/qwen_multidomain_v1/probe_unseen/results.jsonl",
                    help="results.jsonl from a completed run; baseline preds are read here")
    ap.add_argument("--n-per-cell", type=int, default=120,
                    help="samples per (label, base) cell (V/pass, V/fail, T/pass, T/fail)")
    ap.add_argument("--out", default="data/diagnostics/letter_margins.json")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # rebuild the unseen pool exactly as the steering scripts see it
    all_rows = load_oracle_passed(data_root, baseline_rows_needed("unseen"), dm_dir=dm_dir)
    if args.distraction_pool:
        all_rows = all_rows + load_distraction_pool(Path(args.distraction_pool))

    # baseline preds from an existing completed run
    base_preds = {}
    for line in Path(args.baseline_source).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["layer"] == -1:
            base_preds[r["candidate_id"]] = r["pred"]

    pool = select_pool(all_rows, base_preds, "unseen")
    # bucket rows by (label, base) — last write wins for cid collisions (distraction over train/val)
    buckets = {("vision","pass"):[], ("vision","fail"):[], ("text","pass"):[], ("text","fail"):[]}
    seen_cids = set()
    for r in pool:
        cid = r["candidate_id"]
        if cid in seen_cids: continue
        seen_cids.add(cid)
        if cid not in base_preds: continue
        b = "pass" if base_preds[cid] == r["correct_index"] else "fail"
        buckets[(r["label"], b)].append(r)
    for k, v in buckets.items():
        print(f"  {k}: {len(v)} items")

    rng = random.Random(args.seed)
    samples = {}
    for k, v in buckets.items():
        rng.shuffle(v)
        samples[k] = v[:args.n_per_cell]

    print(f"\nloading model on {args.device}")
    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    out = {"meta": {"n_per_cell": args.n_per_cell, "device": args.device,
                    "baseline_source": str(args.baseline_source)}, "rows": []}

    with torch.inference_mode():
        for (label, base), rows in samples.items():
            for i, row in enumerate(rows):
                msgs = build_messages(row)
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msgs)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)
                last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
                logits = model(**inputs).logits[0, last_idx]
                letter_logits = logits[letter_ids].float().cpu()  # [4]
                ci = row["correct_index"]
                pred = int(letter_logits.argmax().item())
                # full letter softmax probs
                probs = torch.softmax(letter_logits, dim=-1).numpy().tolist()
                # margins
                sorted_idx = letter_logits.argsort(descending=True).tolist()
                top, second = sorted_idx[0], sorted_idx[1]
                logit_correct = float(letter_logits[ci].item())
                logit_pred    = float(letter_logits[pred].item())
                margin_pred_over_second = float(letter_logits[top] - letter_logits[second])
                # For V/pass (pred==correct): margin_correct_over_top_wrong
                # For V/fail: margin_top_wrong_over_correct
                if pred == ci:
                    # confident-correct margin
                    margin = float(letter_logits[ci] - letter_logits[second])
                else:
                    # how much the wrong top beats the correct one
                    margin = float(letter_logits[pred] - letter_logits[ci])
                out["rows"].append({
                    "candidate_id": row["candidate_id"],
                    "label": label, "base": base,
                    "pred": pred, "correct_index": ci, "match": pred == ci,
                    "letter_logits": [float(letter_logits[i]) for i in range(4)],
                    "letter_probs": probs,
                    "p_correct": probs[ci],
                    "p_pred": probs[pred],
                    "margin": margin,
                })
                if i % 30 == 0:
                    print(f"  {label}/{base} {i+1}/{len(rows)}  pred={pred} ci={ci} margin={margin:+.2f} p(corr)={probs[ci]:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # quick on-screen summary
    print("\n=== Margin distribution (sorted; positive = predicted letter beats alternative) ===")
    import statistics as st
    for k in [("vision","pass"),("vision","fail"),("text","pass"),("text","fail")]:
        rows = [r for r in out["rows"] if (r["label"], r["base"]) == k]
        if not rows: continue
        margs = [r["margin"] for r in rows]
        probs = [r["p_correct"] for r in rows]
        print(f"  {k[0]:6s}/{k[1]:4s}  n={len(rows):4d}  "
              f"margin median={st.median(margs):+6.2f}  mean={st.mean(margs):+6.2f}  "
              f"p(correct) median={st.median(probs):.3f}  mean={st.mean(probs):.3f}")


if __name__ == "__main__":
    main()
