"""MoGround-Human baselines table for one backbone tag: base / prompt / CoT / 4-seed task vector.

Conditional v-distraction, own-set conditioning, first-wins de-dup -- the paper's conventions
(and the schema of data/diagnostics/mistral24b_human_baselines.json, which this reproduces).
Single-digit-event surface: the JSON carries events beside every rate, per the standing rule.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"


def load(fp: Path):
    if not fp.exists():
        return None
    seen = {}
    for line in fp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen.setdefault(r["candidate_id"], r)
    return seen


def vd(rows):
    den = [r for r in rows.values()
           if r["label"] == "vision" and r.get("pred_v") == r["correct_index"]]
    ev = sum(1 for r in den if r["pred_vt"] != r["correct_index"])
    return {"vd": ev / len(den) if den else float("nan"), "events": ev, "n": len(den)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag

    files = {"base": DR / f"eval/t8/hm3_{tag}_w0.0.jsonl",
             "prompt": DR / f"eval/t8/hm3_{tag}_promptbase.jsonl",
             "cot": DR / f"eval/t8/hm3_{tag}_cot.jsonl"}
    for s in range(4):
        files[f"s{s}"] = DR / f"eval/t8/hm3_{tag}_s{s}_w0.5.jsonl"

    cells, missing = {}, []
    for k, fp in files.items():
        rows = load(fp)
        if rows is None:
            missing.append(k)
            continue
        cells[k] = vd(rows)

    seeds = [cells[f"s{s}"] for s in range(4) if f"s{s}" in cells]
    summary = {}
    if len(seeds) == 4 and "base" in cells:
        rates = [c["vd"] for c in seeds]
        summary = {"seed_mean_vd": float(np.mean(rates)), "seed_sd": float(np.std(rates, ddof=1)),
                   "seed_events": [c["events"] for c in seeds],
                   "reduction_pct_vs_base": float(100 * (1 - np.mean(rates) / cells["base"]["vd"]))
                   if cells["base"]["vd"] else float("nan")}

    out = {
        "what": ("MoGround-Human (125-item set): base, 4-seed task vector w=0.5, one-line prompt, "
                 "CoT. Conditional v-distraction, own-set conditioning, first-wins de-dup. "
                 "Single-digit-event surface: read events, not percentages."),
        "files": {k: str(v.relative_to(ROOT)) for k, v in files.items()},
        "cells": cells,
        "four_seed": summary,
        "missing": missing,
    }
    outp = Path(args.out) if args.out else DR / f"diagnostics/{tag}_human_baselines.json"
    outp.write_text(json.dumps(out, indent=1))
    for k in ["base", "prompt", "cot"] + [f"s{s}" for s in range(4)]:
        if k in cells:
            c = cells[k]
            print(f"  {k:7s} vd={c['vd']:.4f}  events={c['events']}/{c['n']}")
    if missing:
        print(f"  MISSING: {missing}")
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
