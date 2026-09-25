"""Step 15 (v2): token-ratio ablation, generalized over D_M sources.

Usage:
    python scripts/15_token_ratio_ablation_v2.py --source sonnet
    python scripts/15_token_ratio_ablation_v2.py --source gemini

sonnet layout: data/dm/sonnet/dm_all.jsonl + data/dm/sonnet/pass_qwen/predictions.jsonl
               (predictions encode vt/v/t as integer predicted indices; compare to correct_index)
gemini layout: data/dm/gemini/survivors.jsonl (pass_vt/pass_v/pass_t inline as booleans)

Output: data/results/token_ratio_ablation_{source}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent


ROOT = ROOT


def load_sonnet():
    dm_all = ROOT / "data/dm/sonnet/dm_all.jsonl"
    preds_path = ROOT / "data/dm/sonnet/pass_qwen/predictions.jsonl"
    preds = {}
    for l in preds_path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        preds[r["candidate_id"]] = r
    rows_in = [json.loads(l) for l in dm_all.read_text().splitlines() if l.strip()]
    out = []
    skipped = 0
    for r in rows_in:
        cid = r["candidate_id"]
        if cid not in preds:
            skipped += 1
            continue
        p = preds[cid]
        out.append({
            "candidate_id": cid,
            "label": r["label"],
            "question": r["question"],
            "options": r["options"],
            "caption_for_filter": r["caption_for_filter"],
            "image_path": r["image_path"],
            "vt_correct": int(p["vt"] == r["correct_index"]),
            "v_correct":  int(p["v"]  == r["correct_index"]),
            "t_correct":  int(p["t"]  == r["correct_index"]),
        })
    return out, skipped


def load_gemini():
    surv = ROOT / "data/dm/gemini/survivors.jsonl"
    out = []
    for l in surv.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out.append({
            "candidate_id": r["candidate_id"],
            "label": r["label"],
            "question": r["question"],
            "options": r["options"],
            "caption_for_filter": r["caption_for_filter"],
            "image_path": r["image_path"],
            "vt_correct": int(bool(r["pass_vt"])),
            "v_correct":  int(bool(r["pass_v"])),
            "t_correct":  int(bool(r["pass_t"])),
        })
    return out, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sonnet", "gemini"], required=True)
    args = ap.parse_args()

    if args.source == "sonnet":
        rows_in, skipped = load_sonnet()
    else:
        rows_in, skipped = load_gemini()

    out_path = ROOT / f"data/results/token_ratio_ablation_{args.source}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    arr = []
    for r in rows_in:
        text = (
            f"Caption: {r['caption_for_filter']}\nQuestion: {r['question']}\n"
            + "\n".join(f"{L}. {o}" for L, o in zip("ABCD", r["options"]))
            + "\n\nReply with exactly one letter (A, B, C, or D) and nothing else."
        )
        n_text = len(proc.tokenizer.encode(text, add_special_tokens=False))
        img = Image.open(r["image_path"]).convert("RGB")
        po = proc.image_processor(images=[img], return_tensors="pt")
        t, h, w = po["image_grid_thw"][0].tolist()
        n_img = (t * h * w) // 4
        arr.append({
            "candidate_id": r["candidate_id"],
            "label": r["label"],
            "n_text": n_text,
            "n_img": n_img,
            "ratio": n_text / n_img,
            "vt_correct": r["vt_correct"],
            "v_correct":  r["v_correct"],
            "t_correct":  r["t_correct"],
        })

    print(f"source={args.source}  rows={len(arr)}  skipped (no pred)={skipped}")
    for label in ("vision", "text"):
        sub = [r for r in arr if r["label"] == label]
        if not sub:
            continue
        ratios = np.array([r["ratio"] for r in sub])
        n_text = np.array([r["n_text"] for r in sub])
        n_img = np.array([r["n_img"] for r in sub])
        print(f"\n=== {label}  N={len(sub)} ===")
        print(f"  n_text:  median={np.median(n_text):.0f}  p10/p90={np.percentile(n_text,10):.0f}/{np.percentile(n_text,90):.0f}")
        print(f"  n_img :  median={np.median(n_img):.0f}  p10/p90={np.percentile(n_img,10):.0f}/{np.percentile(n_img,90):.0f}")
        print(f"  ratio :  median={np.median(ratios):.3f}  p10/p90={np.percentile(ratios,10):.3f}/{np.percentile(ratios,90):.3f}")

    summary = {}
    for label in ("vision", "text"):
        sub = [r for r in arr if r["label"] == label]
        if not sub:
            continue
        sub.sort(key=lambda r: r["ratio"])
        N = len(sub)
        bins = []
        edges = [0, N // 4, N // 2, 3 * N // 4, N]
        for i in range(4):
            chunk = sub[edges[i]:edges[i + 1]]
            if not chunk:
                continue
            ratios = [r["ratio"] for r in chunk]
            bins.append({
                "bin": f"Q{i+1}",
                "n": len(chunk),
                "ratio_lo": min(ratios),
                "ratio_hi": max(ratios),
                "ratio_med": float(np.median(ratios)),
                "vt_acc": float(np.mean([r["vt_correct"] for r in chunk])),
                "v_acc":  float(np.mean([r["v_correct"]  for r in chunk])),
                "t_acc":  float(np.mean([r["t_correct"]  for r in chunk])),
            })
        summary[label] = bins
        print(f"\n=== {label}  Qwen accuracy by ratio quartile ===")
        print(f"  {'bin':<4} {'n':>4} {'ratio_med':>10} {'vt_acc':>8} {'v_acc':>8} {'t_acc':>8}")
        for b in bins:
            print(f"  {b['bin']:<4} {b['n']:>4} {b['ratio_med']:>10.3f} {b['vt_acc']:>8.3f} {b['v_acc']:>8.3f} {b['t_acc']:>8.3f}")

    out_path.write_text(json.dumps({"per_row": arr, "by_quartile": summary}, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
