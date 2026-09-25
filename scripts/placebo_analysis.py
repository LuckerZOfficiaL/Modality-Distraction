r"""Placebo-caption ladder analysis (task #8): flip rates under true vs placebo captions.

For each backbone, the denominator is the set of vision items it answers correctly from V-only in
its canonical base run; the true-caption flip rate (P0) is that run's own pred_vt. The three
placebo arms come from placebo_{tag}.jsonl (VT-only run over the arm-prefixed pool). All four
rates are computed on the identical item set, so every contrast is paired; exact McNemar
(binomial on the discordant pairs) tests each arm against P0.

    python scripts/placebo_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
ARMS = ["mismatch", "scramble", "neutral"]
TAGS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3b", "Qwen2.5-VL-3B"),
        ("llavaov", "LLaVA-OV-7B"), ("next", "LLaVA-NeXT-8B"), ("qwen2b", "Qwen2-VL-2B"),
        ("llava15", "LLaVA-1.5-7B")]


def load(p):
    seen = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            r = json.loads(line); seen.setdefault(r["candidate_id"], r)
    return seen


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="",
                    help="comma list of 'tag:DisplayName' overriding the paper cohort; "
                         "use together with --out when adding a non-cohort backbone")
    ap.add_argument("--out", default="diagnostics/placebo_ladder.json",
                    help="output json, relative to data/. Override when adding a backbone that is "
                         "not yet part of the paper cohort, so the paper's own input file (the "
                         "default) is never rewritten.")
    args = ap.parse_args()
    tags = TAGS
    if args.tags:
        tags = [(t.split(":", 1)[0], t.split(":", 1)[1] if ":" in t else t.split(":", 1)[0])
                for t in args.tags.split(",") if t.strip()]

    out = {}
    print(f"{'backbone':16}{'n':>6}{'true':>8}" + "".join(f"{a:>10}" for a in ARMS)
          + "   McNemar p (arm vs true)")
    for tag, disp in tags:
        base = load(DR / f"eval/t8/{tag}_w0.0.jsonl")
        plc = load(DR / f"eval/t8/placebo_{tag}.jsonl")
        # denominator: V-only-correct vision items present in every arm
        den = [c for c, r in base.items()
               if r["label"] == "vision" and r["pred_v"] == r["correct_index"]
               and all(f"{a}::{c}" in plc for a in ARMS)]
        flips = {"true": np.array([base[c]["pred_vt"] != base[c]["correct_index"] for c in den])}
        for a in ARMS:
            flips[a] = np.array([plc[f"{a}::{c}"]["pred_vt"] != base[c]["correct_index"]
                                 for c in den])
        rate = {k: float(v.mean()) for k, v in flips.items()}
        pv = {}
        for a in ARMS:
            b01 = int((~flips["true"] & flips[a]).sum())   # placebo flips, true does not
            b10 = int((flips["true"] & ~flips[a]).sum())   # true flips, placebo does not
            pv[a] = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        out[tag] = {"n": len(den), "rates": rate, "mcnemar_p": pv,
                    "discordant": {a: [int((~flips['true'] & flips[a]).sum()),
                                       int((flips['true'] & ~flips[a]).sum())] for a in ARMS}}
        print(f"{disp:16}{len(den):>6}{rate['true']:>8.3f}"
              + "".join(f"{rate[a]:>10.3f}" for a in ARMS)
              + "   " + "  ".join(f"{a[:4]}={pv[a]:.1e}" for a in ARMS))
    # cohort means + the ladder ordering
    mean = {k: float(np.mean([out[t]["rates"][k] for t, _ in tags]))
            for k in ["true"] + ARMS}
    print(f"\n{'cohort mean':16}{'':>6}{mean['true']:>8.3f}"
          + "".join(f"{mean[a]:>10.3f}" for a in ARMS))
    print(f"\nladder (cohort): true {mean['true']:.3f}  >  mismatch {mean['mismatch']:.3f}"
          f"  >  scramble {mean['scramble']:.3f}  >  neutral {mean['neutral']:.3f} ?")
    ratio = {a: mean["true"] / mean[a] if mean[a] else float("inf") for a in ARMS}
    print("true / placebo ratio: " + "  ".join(f"{a}={ratio[a]:.1f}x" for a in ARMS))
    p = DR / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_model": out, "cohort_mean": mean, "ratio": ratio}, indent=2))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
