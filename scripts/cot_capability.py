#!/usr/bin/env python3
"""Zero-shot chain-of-thought on the general-capability benchmarks.

scripts/benchmark_capability.py scores capability by letter-logit argmax (one forward pass); CoT needs the model to
generate a reasoning chain and then be parsed for 'Answer: X'. This script pairs scripts/benchmark_capability.py's
benchmark loaders and registry with scripts/cot_baseline.py's CoT instruction, generation settings and parser,
so the CoT row of the baseline comparison has the same capability column as every other row.

Parity with scripts/cot_baseline.py is deliberate and load-bearing: identical COT_INSTR, identical greedy
decoding, identical parse_letter (unparseable -> -1, scored wrong), so the CoT numbers on the
benchmarks are produced the same way as the CoT numbers on MoGround/assembled.

    python scripts/cot_capability.py --model qwen3b --benchmark mmstar --limit 400
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c95 = _load("_c95", "cot_baseline.py")     # COT_INSTR, parse_letter, make_qwen/make_hf
c78 = _load("_c78", "benchmark_capability.py")   # REGISTRY + benchmark rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="scripts/cot_baseline.py registry tag, e.g. qwen3b, next")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--key", default=None, help="output label (default {model}_cotcap)")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    key = a.key or f"{a.model}_cotcap"

    rows = [json.loads(l) for l in
            (ROOT / f"data/benchmarks/{a.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    if a.skip:
        rows = rows[a.skip:]
    if a.limit:
        rows = rows[:a.limit]

    family, model_id, cls_tag = c95.REGISTRY[a.model]
    gen = (c95.make_qwen(model_id, cls_tag, a.device) if family == "qwen"
           else c95.make_hf(model_id, a.device))

    out = ROOT / "data/diagnostics/capability" / f"{a.benchmark}_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds, gold, nbad = [], [], 0
    for r in tqdm(rows, desc=f"{a.benchmark}/{key}"):
        n = len(r["options"])
        row = dict(r)
        row["image_path"] = str(ROOT / r["image_path"]) if not str(r["image_path"]).startswith("/") \
            else r["image_path"]
        text = gen(row, True, False, a.max_new_tokens)   # image, no caption (benchmarks have none)
        p = c95.parse_letter(text, n)
        if p < 0 or p >= n:
            nbad += 1
            p = -1
        preds.append(p); gold.append(int(r["correct_index"]))

    ok = [p == g for p, g in zip(preds, gold)]
    acc = sum(ok) / len(ok)
    se = (acc * (1 - acc) / len(ok)) ** 0.5
    out.write_text(json.dumps({"benchmark": a.benchmark, "model": a.model, "key": key,
                               "n": len(ok), "acc": acc, "se": se, "skip": a.skip,
                               "unparseable": nbad, "preds": preds,
                               "correct_index": gold}, indent=2))
    print(f"\n{a.benchmark} [{key}]  acc={acc:.4f} ± {se:.4f}  (n={len(ok)}, unparseable={nbad})")


if __name__ == "__main__":
    main()
