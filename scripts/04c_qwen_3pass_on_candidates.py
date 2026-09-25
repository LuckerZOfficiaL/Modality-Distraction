"""Step 4c: Qwen2.5-VL-3B 3-pass on the full `candidates.jsonl` (not just dm_all
survivors). Used to identify Qwen-distraction events on candidates that the
oracle keep rule rejected — these become a distraction pool for steering eval.

Mirrors 04_vlm_behavioral_filter.py format and resume semantics, but:
  - Reads <dm-dir>/candidates.jsonl directly (no keep-rule application).
  - Writes <dm-dir>/pass_qwen_candidates/predictions.jsonl in the same
    schema (`candidate_id`, `label`, `vt`, `v`, `t`, `correct_index`).
  - No `__split` field — candidates aren't split yet.

Resume-safe: skips candidate_ids whose row already has all three predictions.
RUN ON GPU.

Usage:
  python scripts/04c_qwen_3pass_on_candidates.py --dm-dir data/dm/gemini/batch1
  python scripts/04c_qwen_3pass_on_candidates.py --dm-dir data/dm/sonnet_hardT/batch6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from sae_steering.models import load_qwen_vl, predict_mcq


def load_existing_preds(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out[r["candidate_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml",
                    help="ignored if --dm-dir is given")
    ap.add_argument("--dm-dir", default=None,
                    help="path to a D_M dir containing candidates.jsonl")
    ap.add_argument("--pass-dir-name", default="pass_qwen_candidates",
                    help="subdir under dm-dir for outputs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.dm_dir is not None:
        dm_dir = Path(args.dm_dir)
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        dm_dir = Path(cfg["paths"]["dm_dir"])
    pass_dir = dm_dir / args.pass_dir_name
    pass_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pass_dir / "predictions.jsonl"

    rows = [
        json.loads(l) for l in (dm_dir / "candidates.jsonl").read_text().splitlines()
        if l.strip()
    ]
    existing = load_existing_preds(pred_path)
    pending = [
        r for r in rows
        if r["candidate_id"] not in existing
        or any(k not in existing[r["candidate_id"]] for k in ("vt", "v", "t"))
    ]
    print(f"total candidates: {len(rows)} | already complete: {len(rows) - len(pending)} | pending: {len(pending)}")
    if not pending:
        return

    model, processor = load_qwen_vl(device=args.device)
    with pred_path.open("a") as out_f:
        for r in tqdm(pending, desc="qwen 3-pass (candidates)"):
            cid = r["candidate_id"]
            vt = predict_mcq(model, processor, r["question"], r["options"],
                             image_path=r["image_path"], caption=r["caption_for_filter"])
            v  = predict_mcq(model, processor, r["question"], r["options"],
                             image_path=r["image_path"], caption=None)
            t  = predict_mcq(model, processor, r["question"], r["options"],
                             image_path=None, caption=r["caption_for_filter"])
            out_f.write(json.dumps({
                "candidate_id": cid,
                "label": r["label"],
                "vt": vt, "v": v, "t": t,
                "correct_index": r["correct_index"],
            }) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
