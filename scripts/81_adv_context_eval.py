"""Misleading-retrieval cell for the RAG-distraction testbed: evaluate models on TEST-split vision
items with the caption replaced by the ADVERSARIAL one (a fluent same-scene caption asserting a wrong
option; scripts/76 --split test). This is the cell the offline Phase-0 sim could not cover: context
that actively misleads rather than merely distracts.

Reuses scripts/61's model adapters + 3-condition protocol on the swapped rows, so outputs are
format-compatible: pred_vt is the model under image+ADVERSARIAL context, pred_v the no-context
reference, pred_t the text-only strength of the adversarial caption. Resumable.

    CUDA_VISIBLE_DEVICES=0 python scripts/81_adv_context_eval.py --model qwen2.5-vl-3b
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s61", ROOT / "61_behavioral_eval.py")
s61 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s61)  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="scripts/61 registry key (e.g. qwen2.5-vl-3b)")
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--key", default=None, help="output key (default: sanitized --model)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm = Path(cfg["paths"]["dm_dir"])
    rows = [json.loads(l) for l in (dm / "splits/test_adv.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["label"] == "vision" and r.get("adv_caption")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} test vision items with adversarial captions")

    spec = s61.REGISTRY.get(args.model) or {"type": "hf", "model_id": args.model}
    key = args.key or s61.re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)
    make = {"qwen": s61.make_qwen, "hf": s61.make_hf, "llava": s61.make_llava}[spec["type"]]
    predict3 = make(spec, args.device)

    out = ROOT.parent / "data/eval/t8_adv"; out.mkdir(parents=True, exist_ok=True)
    op = out / f"{key}.jsonl"
    done = set()
    if op.exists():
        done = {json.loads(l)["candidate_id"] for l in op.read_text().splitlines() if l.strip()}
        print(f"resuming: {len(done)} already done")

    with op.open("a") as f:
        for r in tqdm(rows, desc=f"{key}/adv"):
            if r["candidate_id"] in done:
                continue
            swapped = {**r, "caption_for_filter": r["adv_caption"]}
            preds, logs = predict3(swapped)
            rec = {"candidate_id": r["candidate_id"], "label": r["label"], "source": r["source"],
                   "correct_index": r["correct_index"], "adv_target_index": r.get("adv_target_index"),
                   "pred_vt": preds[0], "pred_v": preds[1], "pred_t": preds[2]}
            if logs is not None:
                rec["logits_vt"], rec["logits_v"], rec["logits_t"] = logs
            f.write(json.dumps(rec) + "\n"); f.flush()
    print(f"-> {op}")


if __name__ == "__main__":
    main()
