"""Step 29: text-bias failure taxonomy per benchmark (offline).

Characterizes WHICH benchmark failures are text-bias (prior-anchored, correctable by
image-amplification) vs genuine visual errors (image already used, still wrong). Uses
the per-item triple from scripts/27: noimg_pred (answer from text alone), base_pred
(answer with image), correct_index, + the steered preds (scripts/27 sweep) to measure
recoverability. No GPU.

Measures per benchmark:
  1. prior-default rate   = P(base_pred == noimg_pred)             (image-insensitive)
  2. text-bias share      = among failures, frac prior-anchored (base_pred==noimg_pred)
     (complement = image-influenced failures = genuine visual/other errors)
  3. recoverability       = of prior-anchored failures, frac flipped correct by steering
  4. 2x2 contingency      noimg_correct x base_correct (image helps / distracts / etc.)
  ViLP only: gold check base_pred == prior_answer index vs the proxy.

    python scripts/29_text_bias_diagnostic.py --benchmark vilp --layer 33 --alpha 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rows-jsonl", default=None)
    ap.add_argument("--layer", type=int, required=True, help="steer cell to measure recoverability")
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    rows = [json.loads(l) for l in Path(args.rows_jsonl
            or f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rows])
    base = np.load(out_dir / "baseline_pred.npy")
    noimg = np.load(out_dir / "noimg_pred.npy")
    preds = np.load(out_dir / "sweep_preds.npy")
    meta = [json.loads(l) for l in (out_dir / "sweep_pred_meta.jsonl").read_text().splitlines() if l.strip()]
    si = next(i for i, m in enumerate(meta)
              if m["tag"] == "real" and m["layer"] == args.layer and m["alpha"] == args.alpha)
    steered = preds[si]

    base_ok = base == ci
    noimg_ok = noimg == ci
    anchored = base == noimg                      # image did not change the answer
    n = len(rows)
    print(f"=== {args.benchmark}: n={n} ===")
    print(f"base acc (image)      {base_ok.mean():.3f}")
    print(f"prior acc (no image)  {noimg_ok.mean():.3f}")
    print(f"prior-default rate    {anchored.mean():.3f}   (image leaves answer unchanged)")

    fail = ~base_ok
    nf = int(fail.sum())
    anch_fail = fail & anchored                   # prior-anchored failure = text-bias
    infl_fail = fail & ~anchored                  # image-influenced failure = genuine/other
    print(f"\nfailures: {nf} ({fail.mean():.3f} of pool)")
    print(f"  text-bias (prior-anchored)   {anch_fail.sum():4d}  = {anch_fail.sum()/max(nf,1):.3f} of failures")
    print(f"  image-influenced (visual)    {infl_fail.sum():4d}  = {infl_fail.sum()/max(nf,1):.3f} of failures")

    def rec(mask):
        m = mask
        return float((steered[m] == ci[m]).mean()) if m.sum() else float("nan")
    print(f"\nrecoverability by steering (L{args.layer} a={args.alpha}):")
    print(f"  prior-anchored failures flipped correct  {rec(anch_fail):.3f}  (n={int(anch_fail.sum())})")
    print(f"  image-influenced failures flipped correct {rec(infl_fail):.3f}  (n={int(infl_fail.sum())})")

    print(f"\n2x2 contingency (rows=prior/noimg, cols=base):")
    print(f"{'':14}{'base RIGHT':>12}{'base WRONG':>12}")
    for pr, lbl in [(noimg_ok, "prior RIGHT"), (~noimg_ok, "prior WRONG")]:
        print(f"{lbl:14}{int((pr&base_ok).sum()):>12}{int((pr&~base_ok).sum()):>12}")
    helped = (~noimg_ok) & base_ok                # image rescued a wrong prior
    distract = noimg_ok & (~base_ok)              # image broke a right prior
    print(f"  image HELPS (prior wrong->base right): {helped.sum()} ({helped.mean():.3f})")
    print(f"  image DISTRACTS (prior right->base wrong): {distract.sum()} ({distract.mean():.3f})")

    # ViLP gold check: did the model answer the annotated language-prior option?
    if all("prior_answer" in r for r in rows):
        pa = []
        for r in rows:
            try:
                pa.append(r["options"].index(r["prior_answer"]))
            except ValueError:
                pa.append(-1)
        pa = np.array(pa)
        valid = pa >= 0
        gave_prior = (base == pa) & valid
        print(f"\nViLP gold check (annotated prior_answer):")
        print(f"  base==prior_answer overall {gave_prior.mean():.3f}; "
              f"among failures {(gave_prior & fail).sum()/max(nf,1):.3f}")
        agree = (anchored == gave_prior)[valid]
        print(f"  proxy (base==noimg) agrees with gold (base==prior_answer): {agree.mean():.3f}")


if __name__ == "__main__":
    main()
