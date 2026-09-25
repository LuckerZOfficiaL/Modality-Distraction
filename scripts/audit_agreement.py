r"""Agreement between the original MoGround audit and blinded re-audit rater(s).

The original audit (data/release/moground/human_audit.jsonl) judged pass/fail knowing the
intended modality. Blinded raters (data/moground_validation/validations_{rater}.jsonl, from
blinded_audit_notebook.ipynb) instead pick which input determines the answer: vision / text /
both / neither, without seeing the intended label.

A blinded verdict maps to the certificate as: PASS iff the verdict equals the intended modality
exactly ("vision" for a vision-grounded item, "text" for a text-grounded one); "both" and
"neither" are certificate failures, as is naming the wrong modality. Reports per-rater raw
agreement with the original pass/fail, Cohen's kappa per pair, and Fleiss' kappa over all raters
(original included) on the binary pass/fail.

    python scripts/audit_agreement.py [rater2 rater3 ...]   # default: every validations_*.jsonl
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "data/release/moground/human_audit.jsonl"
VDIR = ROOT / "data/moground_validation"


def load_orig():
    out = {}
    for line in ORIG.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["candidate_id"]] = {"label": r["modality"], "pass": r["verdict"] == "yes"}
    return out


def load_rater(name):
    p = VDIR / f"validations_{name}.jsonl"
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); out[r["candidate_id"]] = r["verdict"]
    return out


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def fleiss_kappa(mat):
    """mat: items x 2 category counts (pass, fail), one row per item, raters may vary per item."""
    mat = np.asarray(mat, float)
    n = mat.sum(1)
    keep = n >= 2
    mat, n = mat[keep], n[keep]
    P = ((mat ** 2).sum(1) - n) / (n * (n - 1))
    pbar = P.mean()
    pj = mat.sum(0) / mat.sum()
    pe = (pj ** 2).sum()
    return (pbar - pe) / (1 - pe)


def main():
    orig = load_orig()
    names = sys.argv[1:] or sorted(p.stem.replace("validations_", "")
                                   for p in VDIR.glob("validations_*.jsonl")
                                   if not p.stem.endswith("_notebook"))
    if not names:
        print("no rater files found in", VDIR); return
    raters = {}
    for nm in names:
        v = load_rater(nm)
        raters[nm] = {c: (v[c] == orig[c]["label"]) for c in v if c in orig}
        n = len(raters[nm])
        print(f"{nm}: {n}/250 items judged")
        if n:
            agree = np.mean([raters[nm][c] == orig[c]["pass"] for c in raters[nm]])
            k, po = cohen_kappa([orig[c]["pass"] for c in raters[nm]],
                                [raters[nm][c] for c in raters[nm]])
            passes = np.mean(list(raters[nm].values()))
            print(f"  blinded pass rate {passes:.3f} (original 0.936)"
                  f"   raw agreement with original {agree:.3f}   Cohen kappa {k:.3f}")
    for a, b in combinations(names, 2):
        common = sorted(set(raters[a]) & set(raters[b]))
        if common:
            k, po = cohen_kappa([raters[a][c] for c in common], [raters[b][c] for c in common])
            print(f"{a} vs {b}: n={len(common)}  agreement {po:.3f}  kappa {k:.3f}")
    # Fleiss over original + all raters on the shared items
    ids = sorted(set(orig) & set.union(*[set(r) for r in raters.values()]) if raters else [])
    mat = []
    for c in ids:
        votes = [orig[c]["pass"]] + [raters[nm][c] for nm in names if c in raters[nm]]
        mat.append([sum(votes), len(votes) - sum(votes)])
    if mat:
        print(f"Fleiss kappa (original + {len(names)} rater(s), n={len(mat)}): "
              f"{fleiss_kappa(mat):.3f}")


if __name__ == "__main__":
    main()
