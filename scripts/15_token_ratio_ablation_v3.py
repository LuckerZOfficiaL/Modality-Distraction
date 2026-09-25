"""Token-numerosity ablation, v3.

Joins oracle-survivor rows (with full content) to per-row Qwen 3-pass
predictions, computes (n_text, n_image, ratio) under the filter-time prompt,
and reports two distractor-direction effects:

  * On VISION rows:  V-only acc  vs  V+T acc      (caption-as-distractor)
  * On TEXT rows:    T-only acc  vs  V+T acc      (image-as-distractor, new)

Plus ρ-quartile binning per label.

Usage:
    .venv/bin/python scripts/15_token_ratio_ablation_v3.py \\
        --survivors data/dm/sonnet/batch_1-3/dm_all.jsonl \\
        --preds     data/dm/sonnet/batch_1-3/pass_qwen/predictions.jsonl \\
        --out       data/results/token_ratio_ablation_sonnet_b1to3.json

    .venv/bin/python scripts/15_token_ratio_ablation_v3.py \\
        --survivors data/dm/gemini/survivors.jsonl \\
        --preds     data/dm/gemini/pass_qwen/predictions.jsonl \\
        --out       data/results/token_ratio_ablation_gemini.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", required=True,
                    help="oracle-survivor jsonl (with question/options/image_path/caption_for_filter/correct_index/label)")
    ap.add_argument("--preds", required=True,
                    help="Qwen 3-pass predictions.jsonl (vt/v/t integer prediction per candidate_id)")
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--processor", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    survivors = load_jsonl(Path(args.survivors))
    preds = {r["candidate_id"]: r for r in load_jsonl(Path(args.preds))}
    print(f"survivors: {len(survivors)} | predictions: {len(preds)}")

    print("loading processor...")
    proc = AutoProcessor.from_pretrained(args.processor)
    print("processor loaded; tokenizing rows...")
    arr = []
    skipped = 0
    for r in tqdm(survivors, desc="rows"):
        cid = r["candidate_id"]
        if cid not in preds:
            skipped += 1
            continue
        p = preds[cid]
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
        ci = r["correct_index"]
        arr.append({
            "candidate_id": cid,
            "label": r["label"],
            "n_text": n_text,
            "n_img": n_img,
            "ratio": n_text / n_img,
            "vt_correct": int(p["vt"] == ci),
            "v_correct":  int(p["v"]  == ci),
            "t_correct":  int(p["t"]  == ci),
        })

    print(f"joined rows: {len(arr)}  (skipped no-pred: {skipped})")

    # overall acc + distractor gaps
    summary = {}
    for label in ("vision", "text"):
        sub = [r for r in arr if r["label"] == label]
        if not sub:
            continue
        vt = np.array([r["vt_correct"] for r in sub])
        v  = np.array([r["v_correct"]  for r in sub])
        t  = np.array([r["t_correct"]  for r in sub])
        on_modality = v if label == "vision" else t  # the modality-only condition
        gap = on_modality.mean() - vt.mean()
        helped = int(((on_modality - vt) > 0).sum())
        hurt   = int(((on_modality - vt) < 0).sum())
        ratios = np.array([r["ratio"] for r in sub])
        n_text = np.array([r["n_text"] for r in sub])
        n_img  = np.array([r["n_img"]  for r in sub])
        print(f"\n=== {label}  N={len(sub)} ===")
        print(f"  V+T acc      : {vt.mean():.3f}")
        print(f"  V-only acc   : {v.mean():.3f}")
        print(f"  T-only acc   : {t.mean():.3f}")
        if label == "vision":
            print(f"  distractor gap (V-only − V+T) : {gap:+.4f}  helped={helped}  hurt={hurt}")
        else:
            print(f"  distractor gap (T-only − V+T) : {gap:+.4f}  helped={helped}  hurt={hurt}")
        print(f"  n_text  med={np.median(n_text):.0f}  p10/p90={np.percentile(n_text,10):.0f}/{np.percentile(n_text,90):.0f}")
        print(f"  n_img   med={np.median(n_img):.0f}  p10/p90={np.percentile(n_img,10):.0f}/{np.percentile(n_img,90):.0f}")
        print(f"  ratio   med={np.median(ratios):.3f}  p10/p90={np.percentile(ratios,10):.3f}/{np.percentile(ratios,90):.3f}")

        # quartile binning by ρ
        sub_sorted = sorted(sub, key=lambda r: r["ratio"])
        N = len(sub_sorted)
        edges = [0, N // 4, N // 2, 3 * N // 4, N]
        bins = []
        for i in range(4):
            chunk = sub_sorted[edges[i]:edges[i + 1]]
            if not chunk:
                continue
            rs = [r["ratio"] for r in chunk]
            bins.append({
                "bin": f"Q{i+1}",
                "n": len(chunk),
                "ratio_lo": min(rs),
                "ratio_hi": max(rs),
                "ratio_med": float(np.median(rs)),
                "vt_acc": float(np.mean([r["vt_correct"] for r in chunk])),
                "v_acc":  float(np.mean([r["v_correct"]  for r in chunk])),
                "t_acc":  float(np.mean([r["t_correct"]  for r in chunk])),
            })
        print(f"  by ρ-quartile: {'bin':<3} {'n':>4} {'ρ_med':>8} {'vt':>6} {'v':>6} {'t':>6}")
        for b in bins:
            print(f"                 {b['bin']:<3} {b['n']:>4} {b['ratio_med']:>8.3f} {b['vt_acc']:>6.3f} {b['v_acc']:>6.3f} {b['t_acc']:>6.3f}")
        summary[label] = {
            "overall": {"vt": float(vt.mean()), "v": float(v.mean()), "t": float(t.mean()),
                        "distractor_gap": float(gap), "helped": helped, "hurt": hurt},
            "by_quartile": bins,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_row": arr, "summary": summary}, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
