r"""Reconcile the two base-model evaluation runs of Qwen2-VL-2B.

The paper draws its leaderboard and grounding relation from the June run
(`eval/t8/Qwen_Qwen2-VL-2B-Instruct.jsonl`) and its mitigation baselines from the July run
(`eval/t8/qwen2b_w0.0.jsonl`). The two disagree: V-only accuracy 0.797 vs 0.709, v-distraction
0.047 vs 0.091, on identical items. Three of the seven backbones' July files are byte-copies of
June, and only the Qwen family diverges, which points at an environment change (the repo carries
two Python environments: conda transformers 4.48.3 / torch 2.6, and .venv transformers 5.6.2 /
torch 2.11) rather than a configuration difference in the scripts, which are identical in prompt
builder, pixel budget, dtype and letter-logit decoding.

This script compares a fresh run against both, and reports which the current environment
reproduces, plus what each choice implies for the reported w=0.5 reduction.

    python scripts/110_run_reconciliation.py --new qwen2b_recheck0805
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
JUNE, JULY = "Qwen_Qwen2-VL-2B-Instruct", "qwen2b_w0.0"


def load(key, pool="t8"):
    p = DR / f"eval/{pool}/{key}.jsonl"
    if not p.exists():
        return None
    d = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); d.setdefault(r["candidate_id"], r)
    return d


def rates(d, ids=None):
    """(V-only acc, v-distraction, n_solvable) over vision items."""
    vis = [r for r in d.values() if r["label"] == "vision" and (ids is None or r["candidate_id"] in ids)]
    acc = sum(r["pred_v"] == r["correct_index"] for r in vis) / len(vis)
    s = [r for r in vis if r["pred_v"] == r["correct_index"]]
    return acc, sum(r["pred_vt"] != r["correct_index"] for r in s) / len(s), len(s)


def agree(a, b):
    sh = set(a) & set(b)
    return (sum(a[c]["pred_v"] == b[c]["pred_v"] and a[c]["pred_vt"] == b[c]["pred_vt"] for c in sh)
            / max(len(sh), 1), len(sh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="qwen2b_recheck0805")
    a = ap.parse_args()
    TEST = {json.loads(l)["candidate_id"]
            for l in (DR / "dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}

    runs = {"June (leaderboard)": load(JUNE), "July (mitigation)": load(JULY), "fresh re-run": load(a.new)}
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        print(f"not available yet: {missing}"); return

    print("=== \\dsname{}, vision items ===")
    print(f"{'run':22}{'n':>7}{'V-only acc':>12}{'v-distr (all)':>15}{'v-distr (test)':>16}")
    for k, d in runs.items():
        acc, vd, n = rates(d)
        _, vdt, nt = rates(d, TEST)
        print(f"{k:22}{len(d):>7}{acc:>12.3f}{vd:>15.3f}{vdt:>16.3f}")

    print("\n=== which run does the current environment reproduce? ===")
    for ref in ("June (leaderboard)", "July (mitigation)"):
        ag, n = agree(runs["fresh re-run"], runs[ref])
        print(f"  fresh vs {ref:22} prediction agreement {ag:.3f} on {n} items")

    # what each baseline choice implies for the reported reduction
    ss = []
    for i in range(4):
        d = load(f"qwen2b{'' if i == 0 else f'_s{i}'}_w0.5")
        if d:
            ss.append(rates(d, TEST)[1])
    if ss:
        st = sum(ss) / len(ss)
        print(f"\n=== w=0.5 steered v-distraction on the test split: {st:.3f} (mean of {len(ss)} seeds) ===")
        for k, d in runs.items():
            b = rates(d, TEST)[1]
            print(f"  vs {k:22} base {b:.3f}  ->  reduction {(st-b)/b*100:+.0f}%")
        print("\n  The paper reports the July comparison. A reduction is only meaningful within a\n"
              "  single environment: the steered runs are July-era, so whichever run the current\n"
              "  environment reproduces is the one whose baseline belongs next to them.")


if __name__ == "__main__":
    main()
