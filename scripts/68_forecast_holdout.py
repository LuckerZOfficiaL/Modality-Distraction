"""Plan B / B3: does the grounding-strength law FORECAST a model it never saw?

Genuine extrapolation (vs the within-pool LOMO of scripts/58): fit the law on every model
EXCEPT a held-out target, then predict the target's per-cell distraction from ONLY its cheap
single-modality accuracies. The actionable claim: estimate a new VLM's cross-modal robustness on
a domain from single-modality probes alone -- no interference-set construction needed.

Step 1 (GPU, you): eval the novel model, producing the same t8 jsonls as the reported cohort:
    python scripts/61_behavioral_eval.py --model <new-model> --pool both
Step 2 (offline, here): forecast it:
    python scripts/68_forecast_holdout.py --model <new-model>

Kill: per-cell MAE > 0.03-0.05 -> law doesn't extrapolate. Strong: MAE ~ 0.01-0.02.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s54", ROOT / "54_grounding_strength.py")
s54 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s54)  # type: ignore
RT = ROOT.parent


def all_cells():
    pts = []
    # the seven paper backbones only: globbing would fold fine-tuned checkpoints into the fit
    keys = s54.CANONICAL
    for m in keys:
        for ds, path, by in [("dm", RT / f"data/eval/t8/{m}.jsonl", True),
                             ("assembled", RT / f"data/eval/t8_assembled/{m}.jsonl", False)]:
            if path.exists():
                for c in s54.cells_from(path, by):
                    c.update(model=m, dataset=ds); pts.append(c)
    return pts


def mae(cells, a, b):
    e = [abs((a + b * c["ground"]) - c["distr"]) for c in cells]
    return float(np.mean(e)) if e else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="held-out model key (run scripts/61 on it first)")
    args = ap.parse_args()

    pts = all_cells()
    tgt = [c for c in pts if c["model"] == args.model]
    other = [c for c in pts if c["model"] != args.model]
    assert tgt, f"no cells for '{args.model}' -- run scripts/61 --model {args.model} --pool both first"
    n_models = len({c['model'] for c in other})
    go = np.array([c["ground"] for c in other]); do = np.array([c["distr"] for c in other])
    b, a = np.polyfit(go, do, 1)
    r = float(np.corrcoef(go, do)[0, 1])
    print(f"law fit on {n_models} other models ({len(other)} cells): distr = {a:.3f} + {b:.3f}*grounding  (r={r:.3f})")
    print(f"forecasting held-out: {args.model}  ({len(tgt)} cells)\n")
    print(f"{'dataset':10}{'domain':9}{'mod':7}{'ground':>8}{'actual':>8}{'pred':>8}{'err':>8}")
    for c in sorted(tgt, key=lambda c: (c["dataset"], c["modality"], c["domain"])):
        pred = a + b * c["ground"]
        print(f"{c['dataset']:10}{c['domain']:9}{c['modality']:7}{c['ground']:>8.3f}"
              f"{c['distr']:>8.3f}{pred:>8.3f}{pred-c['distr']:>+8.3f}")
    overall = mae(tgt, a, b)
    print(f"\nMAE  all={overall:.3f}  vision={mae([c for c in tgt if c['modality']=='vision'], a, b):.3f}  "
          f"text={mae([c for c in tgt if c['modality']=='text'], a, b):.3f}  "
          f"assembled={mae([c for c in tgt if c['dataset']=='assembled'], a, b):.3f}")
    verdict = ("STRONG (<=0.02): forecasting extrapolates" if overall <= 0.02 else
               "OK (<=0.03)" if overall <= 0.03 else
               "WEAK/KILL (>0.05): law does not extrapolate" if overall > 0.05 else "MARGINAL (0.03-0.05)")
    print(f"VERDICT: MAE {overall:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
