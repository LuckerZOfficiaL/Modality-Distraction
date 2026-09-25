r"""Score the human audit of the assembled pool's irrelevance filter.

Two quantities, from disjoint queues:

  MISS RATE  (kept pool)    -- items the LLM judge KEPT that a human reads as leaking. This is the
                               number the paper would report: how clean the released pool is.
  FALSE-ALARM (dropped pool) -- items the judge DROPPED that a human reads as irrelevant. Costs
                               pool size only, never validity; also the positive control showing
                               the rater detects leaks where they exist.

Exclusions and definitions, fixed before scoring and reported explicitly:
  * "ambiguous" = the item's own answer options are flawed (source-benchmark defect, judged on the
    options alone, independent of the added content). Excluded from denominators and reported
    separately. This is an ITEM-VALIDITY exclusion, not an outcome exclusion.
  * three-way taxonomy of non-irrelevant verdicts:
      LEAK     = the content ASSERTS an answer (asserts_gold, asserts_distractor). This is what
                 the LLM judge screens for and what "answer-irrelevant" denies.
      CONFLICT = the content CONTRADICTS the correct option. A different failure: the item behaves
                 as engineered cross-modal conflict, not as a leak. Counted separately.
      OUT-OF-SET = content pointing to an answer outside the four options; definitional status
                 deferred, reported separately (zero on the kept side).

    python scripts/assembled_audit_analysis.py
"""
from __future__ import annotations

import json
from collections import Counter
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/assembled_validation/validations.jsonl"
LEAK = {"asserts_gold", "asserts_distractor"}      # asserts an answer
CONFLICT = {"contradicts"}                        # contradicts the correct option
OOS = {"out_of_set"}                              # deferred category


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    out = {"n_judged": len(rows)}
    print(f"audited items: {len(rows)}\n")

    for pool, what in (("kept", "MISS RATE"), ("dropped", "FALSE-ALARM RATE")):
        sub = [r for r in rows if r.get("pool") == pool]
        if not sub:
            continue
        amb = [r for r in sub if r["verdict"] == "ambiguous"]
        usable = [r for r in sub if r["verdict"] != "ambiguous"]
        n = len(usable)
        cnt = {k: sum(1 for r in usable if r["verdict"] == "leaks" and r.get("reason") in grp)
               for k, grp in (("leak", LEAK), ("conflict", CONFLICT), ("oos", OOS))}
        irr = sum(1 for r in usable if r["verdict"] == "irrelevant")
        print(f"=== {pool} pool: {what} ===")
        print(f"  judged {len(sub)}   ambiguous (excluded) {len(amb)}   usable {n}")
        print(f"  irrelevant {irr}   leak {cnt['leak']}   conflict {cnt['conflict']}   "
              f"out-of-set {cnt['oos']}")
        assert irr + sum(cnt.values()) == n, "closure failed"
        for k in ("leak", "conflict", "oos"):
            lo, hi = wilson(cnt[k], n)
            print(f"    {k:10} {cnt[k]}/{n} = {100*cnt[k]/n:5.2f}%   "
                  f"Wilson 95% CI [{100*lo:.1f}%, {100*hi:.1f}%]")
        out[pool] = {"judged": len(sub), "ambiguous": len(amb), "usable": n, "irrelevant": irr,
                     **{k: cnt[k] for k in cnt},
                     **{f"ci_{k}": wilson(cnt[k], n) for k in cnt}}
        print()

    # per-modality miss rate on the kept pool (the caption side is what the filter touched)
    print("=== kept pool, per modality ===")
    for mod, label in (("vision", "captions added to vision items (filter's target)"),
                       ("text", "images added to text items (never machine-filtered)")):
        sub = [r for r in rows if r.get("pool") == "kept" and r["modality"] == mod]
        usable = [r for r in sub if r["verdict"] != "ambiguous"]
        nn = len(usable)
        lk = sum(1 for r in usable if r["verdict"] == "leaks" and r.get("reason") in LEAK)
        cf = sum(1 for r in usable if r["verdict"] == "leaks" and r.get("reason") in CONFLICT)
        llo, lhi = wilson(lk, nn); clo, chi = wilson(cf, nn)
        print(f"  {label:52} n={nn}")
        print(f"    leak     {lk}/{nn} = {100*lk/max(nn,1):.1f}%  [{100*llo:.1f}, {100*lhi:.1f}]")
        print(f"    conflict {cf}/{nn} = {100*cf/max(nn,1):.1f}%  [{100*clo:.1f}, {100*chi:.1f}]")
        out.setdefault("kept_by_modality", {})[mod] = {
            "usable": nn, "leak": lk, "conflict": cf,
            "ci_leak": wilson(lk, nn), "ci_conflict": wilson(cf, nn)}

    p = ROOT / "data/diagnostics/assembled_audit.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
