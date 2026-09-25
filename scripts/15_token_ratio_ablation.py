"""Step 15: token-ratio ablation.

Per row, compute (n_text_tokens, n_image_tokens) under the same prompt format
the VLM saw at filter time. Bin by ratio = n_text / n_image within each
{vision, text} label and report Qwen V+T accuracy per bin.

No GPU needed — just the processor's image_processor + tokenizer.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent


ROOT = ROOT
DM_ALL = ROOT / "data/dm/dm_all.jsonl"
PREDS = ROOT / "data/dm/pass_qwen/predictions.jsonl"
OUT = ROOT / "data/results/token_ratio_ablation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    preds = {}
    for l in PREDS.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        preds[r["candidate_id"]] = r

    rows = []
    skipped = 0
    for l in DM_ALL.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cid = r["candidate_id"]
        if cid not in preds:
            skipped += 1
            continue

        text = (
            f"Caption: {r['caption_for_filter']}\nQuestion: {r['question']}\n"
            + "\n".join(f"{L}. {o}" for L, o in zip("ABCD", r["options"]))
            + "\n\nReply with exactly one letter (A, B, C, or D) and nothing else."
        )
        n_text = len(proc.tokenizer.encode(text, add_special_tokens=False))

        img = Image.open(r["image_path"]).convert("RGB")
        out = proc.image_processor(images=[img], return_tensors="pt")
        t, h, w = out["image_grid_thw"][0].tolist()
        n_img = (t * h * w) // 4  # 2x2 spatial merge

        p = preds[cid]
        rows.append({
            "candidate_id": cid,
            "label": r["label"],
            "n_text": n_text,
            "n_img": n_img,
            "ratio": n_text / n_img,
            "vt_correct": int(p["vt"] == r["correct_index"]),
            "v_correct":  int(p["v"]  == r["correct_index"]),
            "t_correct":  int(p["t"]  == r["correct_index"]),
        })

    print(f"rows={len(rows)}  skipped (no pred)={skipped}")
    arr = rows

    # report overall token-count distribution
    for label in ("vision", "text"):
        sub = [r for r in arr if r["label"] == label]
        ratios = np.array([r["ratio"] for r in sub])
        n_text = np.array([r["n_text"] for r in sub])
        n_img = np.array([r["n_img"] for r in sub])
        print(f"\n=== {label}  N={len(sub)} ===")
        print(f"  n_text:  median={np.median(n_text):.0f}  p10/p90={np.percentile(n_text,10):.0f}/{np.percentile(n_text,90):.0f}")
        print(f"  n_img :  median={np.median(n_img):.0f}  p10/p90={np.percentile(n_img,10):.0f}/{np.percentile(n_img,90):.0f}")
        print(f"  ratio :  median={np.median(ratios):.3f}  p10/p90={np.percentile(ratios,10):.3f}/{np.percentile(ratios,90):.3f}")

    # quartile bin Qwen accuracy by ratio within each label
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

    OUT.write_text(json.dumps({"per_row": arr, "by_quartile": summary}, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
