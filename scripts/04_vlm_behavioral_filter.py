"""Step 4: Target-VLM behavioral filter.

Run Qwen2.5-VL-3B-Instruct on D_M train + val under three conditions:
  V+T   image + caption_for_filter
  V     image only
  T     caption_for_filter only

Apply the same keep-rule as the oracle filter:
  vision-grounded: V+T ✓ ∧ V ✓ ∧ T ✗
  text-grounded:   V+T ✓ ∧ V ✗ ∧ T ✓

Outputs:
  data/dm/pass/train.jsonl       — train rows where Qwen also satisfies the rule
  data/dm/pass/val.jsonl         — val rows where Qwen also satisfies the rule
  data/dm/pass/predictions.jsonl — every row's three Qwen predictions (audit trail)

D_M test is NEVER VLM-filtered.

Resume-safe: rewrites predictions.jsonl by re-loading prior rows and only
predicting candidates that lack any of the three conditions. Predictions are
flushed per row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from sae_steering.models import load_qwen_vl, predict_mcq


def load_split(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_existing_preds(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
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
                    help="path to a D_M dir containing splits/{train,val}.jsonl; "
                         "if set, overrides config['paths']['dm_dir']")
    ap.add_argument("--pass-dir-name", default="pass",
                    help="subdir under dm-dir for outputs (default: pass)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.dm_dir is not None:
        dm_dir = Path(args.dm_dir)
    else:
        cfg = yaml.safe_load(Path(args.config).read_text())
        dm_dir = Path(cfg["paths"]["dm_dir"])
    splits_dir = dm_dir / "splits"
    pass_dir = dm_dir / args.pass_dir_name
    pass_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pass_dir / "predictions.jsonl"

    rows = []
    for split in ("train", "val"):
        for r in load_split(splits_dir / f"{split}.jsonl"):
            r["__split"] = split
            rows.append(r)

    existing = load_existing_preds(pred_path)
    pending = [r for r in rows if r["candidate_id"] not in existing
               or any(k not in existing[r["candidate_id"]] for k in ("vt", "v", "t"))]
    print(f"total train+val rows: {len(rows)} | already complete: {len(rows) - len(pending)} | pending: {len(pending)}")

    if pending:
        model, processor = load_qwen_vl(device=args.device)
        with pred_path.open("a") as out_f:
            for r in tqdm(pending, desc="qwen 3-pass"):
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
                    "split": r["__split"],
                    "vt": vt, "v": v, "t": t,
                    "correct_index": r["correct_index"],
                }) + "\n")
                out_f.flush()
        # reload after writes
        existing = load_existing_preds(pred_path)

    # apply keep-rule per split
    survivors = {"train": [], "val": []}
    diagnostic_counts = {"train": {"vision": [0,0], "text": [0,0]}, "val": {"vision": [0,0], "text": [0,0]}}
    for r in rows:
        cid = r["candidate_id"]
        if cid not in existing:
            continue
        p = existing[cid]
        ci = r["correct_index"]
        vt_ok = p["vt"] == ci
        v_ok  = p["v"]  == ci
        t_ok  = p["t"]  == ci
        split = r["__split"]
        diagnostic_counts[split][r["label"]][1] += 1
        kept = False
        if r["label"] == "vision" and vt_ok and v_ok and not t_ok:
            kept = True
        elif r["label"] == "text" and vt_ok and not v_ok and t_ok:
            kept = True
        if kept:
            diagnostic_counts[split][r["label"]][0] += 1
            survivors[split].append({**{k:v for k,v in r.items() if not k.startswith("__")},
                                      "qwen_pred_vt": p["vt"], "qwen_pred_v": p["v"], "qwen_pred_t": p["t"]})

    for split, rows_ in survivors.items():
        path = pass_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows_:
                f.write(json.dumps(row) + "\n")
        print(f"{split}: {len(rows_)} kept -> {path}")
        for lbl, (ok, n) in diagnostic_counts[split].items():
            print(f"  {lbl}: {ok}/{n}")


if __name__ == "__main__":
    main()
