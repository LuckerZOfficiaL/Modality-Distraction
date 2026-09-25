"""Test image-as-distractor on text-grounded candidates.

Pool: text-labeled candidates where the oracle's T-only pass is correct
(caption alone suffices). For these, run Qwen 3-pass and compute
T-only − V+T accuracy gap. Symmetric counterpart to the caption-as-distractor
result on vision rows.

This pool intentionally INCLUDES candidates that did not survive the oracle
keep-rule (e.g. oracle V+T wrong, oracle V-only also correct, etc.) — that's
where headroom for V+T < T-only can appear.

Reuses Qwen 3-pass predictions from --existing-preds where available; runs
fresh Qwen inference for the rest. Outputs are written to a new path; nothing
existing is overwritten.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/16_image_distractor_text_rows.py \\
        --candidates  data/dm/gemini/candidates.jsonl \\
        --oracle-t    data/dm/gemini/work/ans_t/results.jsonl \\
        --oracle-vt   data/dm/gemini/work/ans_vt/results.jsonl \\
        --oracle-v    data/dm/gemini/work/ans_v/results.jsonl \\
        --existing-preds data/dm/gemini/pass_qwen/predictions.jsonl \\
        --out-dir     data/results/image_distractor_gemini
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from moground.models import load_qwen_vl, predict_mcq


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_oracle(p: Path) -> dict[str, int]:
    out = {}
    for r in load_jsonl(p):
        if "predicted_index" in r:
            out[r["candidate_id"]] = r["predicted_index"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--oracle-t", required=True)
    ap.add_argument("--oracle-vt", default=None)
    ap.add_argument("--oracle-v", default=None)
    ap.add_argument("--existing-preds", default=None,
                    help="prior Qwen 3-pass predictions to reuse")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "qwen_predictions.jsonl"
    summary_path = out_dir / "summary.json"

    candidates = load_jsonl(Path(args.candidates))
    oracle_t = load_oracle(Path(args.oracle_t))
    oracle_vt = load_oracle(Path(args.oracle_vt)) if args.oracle_vt else {}
    oracle_v  = load_oracle(Path(args.oracle_v))  if args.oracle_v  else {}

    # filter pool: text-labeled, oracle T-only correct
    pool: list[dict] = []
    for c in candidates:
        if c["label"] != "text":
            continue
        cid = c["candidate_id"]
        if cid not in oracle_t:
            continue
        if oracle_t[cid] != c["correct_index"]:
            continue
        pool.append(c)
    print(f"text-labeled candidates: {sum(1 for c in candidates if c['label']=='text')}")
    print(f"  with oracle-T correct: {len(pool)}")

    # reuse existing Qwen preds
    existing: dict[str, dict] = {}
    if args.existing_preds and Path(args.existing_preds).exists():
        for r in load_jsonl(Path(args.existing_preds)):
            if all(k in r for k in ("vt", "v", "t")):
                existing[r["candidate_id"]] = r
    if pred_path.exists():
        for r in load_jsonl(pred_path):
            existing[r["candidate_id"]] = r
    reused = sum(1 for c in pool if c["candidate_id"] in existing)
    pending = [c for c in pool if c["candidate_id"] not in existing]
    print(f"  reusing existing preds: {reused}")
    print(f"  pending Qwen inference: {len(pending)}")

    if pending:
        model, processor = load_qwen_vl(device=args.device)
        with pred_path.open("a") as f:
            for r in tqdm(pending, desc="qwen 3-pass"):
                cid = r["candidate_id"]
                vt = predict_mcq(model, processor, r["question"], r["options"],
                                 image_path=r["image_path"], caption=r["caption_for_filter"])
                v  = predict_mcq(model, processor, r["question"], r["options"],
                                 image_path=r["image_path"], caption=None)
                t  = predict_mcq(model, processor, r["question"], r["options"],
                                 image_path=None, caption=r["caption_for_filter"])
                row = {"candidate_id": cid, "label": "text",
                       "vt": vt, "v": v, "t": t,
                       "correct_index": r["correct_index"]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                existing[cid] = row

    # compute gaps
    n = vt_ok = v_ok = t_ok = 0
    helped = hurt = 0
    image_distractor_n = 0  # T-only correct, V+T wrong
    oracle_imdis_n = 0       # oracle T correct & oracle V+T wrong
    for c in pool:
        cid = c["candidate_id"]
        if cid not in existing: continue
        p = existing[cid]; ci = c["correct_index"]
        n += 1
        a = int(p["vt"] == ci); b = int(p["v"] == ci); d = int(p["t"] == ci)
        vt_ok += a; v_ok += b; t_ok += d
        if d - a > 0: helped += 1
        if d - a < 0: hurt += 1
        if d == 1 and a == 0: image_distractor_n += 1
        if oracle_vt and cid in oracle_vt:
            if oracle_t[cid] == ci and oracle_vt[cid] != ci:
                oracle_imdis_n += 1

    summary = {
        "n": n,
        "qwen_vt_acc": vt_ok / n if n else 0.0,
        "qwen_v_acc":  v_ok  / n if n else 0.0,
        "qwen_t_acc":  t_ok  / n if n else 0.0,
        "distractor_gap_t_minus_vt": (t_ok - vt_ok) / n if n else 0.0,
        "rows_T_correct_VT_wrong": image_distractor_n,
        "rows_VT_correct_T_wrong": sum(1 for c in pool
            if c["candidate_id"] in existing
            and existing[c["candidate_id"]]["vt"] == c["correct_index"]
            and existing[c["candidate_id"]]["t"] != c["correct_index"]),
        "oracle_T_correct_VT_wrong": oracle_imdis_n if oracle_vt else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== text rows, oracle-T correct  N={n} ===")
    print(f"  Qwen V+T acc      : {summary['qwen_vt_acc']:.3f}")
    print(f"  Qwen V-only acc   : {summary['qwen_v_acc']:.3f}")
    print(f"  Qwen T-only acc   : {summary['qwen_t_acc']:.3f}")
    print(f"  T − V+T gap       : {summary['distractor_gap_t_minus_vt']:+.4f}  (helped={helped}, hurt={hurt})")
    print(f"  Qwen T✓ ∧ V+T✗    : {summary['rows_T_correct_VT_wrong']}  (image-as-distractor instances)")
    print(f"  Qwen V+T✓ ∧ T✗    : {summary['rows_VT_correct_T_wrong']}  (image actually helps)")
    if oracle_vt:
        print(f"  oracle T✓ ∧ V+T✗ : {summary['oracle_T_correct_VT_wrong']}  (oracle-level image distraction)")
    print(f"\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
