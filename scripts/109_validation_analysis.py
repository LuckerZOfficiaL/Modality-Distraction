r"""Analyse the human validation of single-modality answerability, and re-score without flagged items.

Two questions, kept apart because they are different constructs:
  certificate   does the item's answer lie in the intended modality and not the other?
                failures: `other_answers` (the off modality also answers it) and `intended_fails`
  answer key    is the marked correct option actually correct?  failure: `bad_key`

A wrong key is an annotation defect, not a certification defect, so the headline certificate rate is
computed with those items removed from the denominator and their rate reported separately. Because a
mis-keyed item can still enter a model's conditioning set (the model may share the annotator's
misreading), we also re-score every distraction rate with all flagged items dropped, as a robustness
check on the numbers the paper reports.

    python scripts/109_validation_analysis.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
VAL = DR / "moground_validation/validations.jsonl"
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3b", "Qwen2.5-VL-3B"),
          ("llavaov", "LLaVA-OV-7B"), ("next", "LLaVA-NeXT-8B"), ("qwen2b", "Qwen2-VL-2B"),
          ("llava15", "LLaVA-1.5-7B")]
CERT_FAIL = {"other_answers", "intended_fails", "both"}


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = stats.binomtest(k, n).proportion_ci(0.95, method="wilson")
    return lo, hi


def main():
    V = {}
    for line in VAL.read_text().splitlines():
        if line.strip():
            v = json.loads(line); V[v["candidate_id"]] = v      # last verdict wins
    n = len(V)
    bad = {c for c, v in V.items() if v["verdict"] == "no" and v["reason"] == "bad_key"}
    cfail = {c for c, v in V.items() if v["verdict"] == "no" and v["reason"] in CERT_FAIL}
    unspec = {c for c, v in V.items() if v["verdict"] == "no" and v["reason"] not in CERT_FAIL | {"bad_key"}}
    cert_n = n - len(bad)

    print(f"=== human validation of single-modality answerability ===")
    print(f"items judged: {n}  ({Counter(v['modality'] for v in V.values())})")
    print(f"  by domain: {dict(Counter(v['domain'] for v in V.values()))}")
    if unspec:
        print(f"  NOTE {len(unspec)} 'no' verdicts have no reason recorded; counted as certificate failures")
        cfail |= unspec; cert_n = n - len(bad)

    lo, hi = wilson(cert_n - len(cfail), cert_n)
    print(f"\ncertificate holds : {cert_n-len(cfail)}/{cert_n} = {(cert_n-len(cfail))/cert_n:.3f} "
          f"[{lo:.3f}, {hi:.3f}]   (wrong-key items excluded from the denominator)")
    lo2, hi2 = wilson(len(bad), n)
    print(f"wrong answer key  : {len(bad)}/{n} = {len(bad)/n:.3f} [{lo2:.3f}, {hi2:.3f}]")
    lo3, hi3 = wilson(n - len(bad) - len(cfail), n)
    print(f"both hold         : {n-len(bad)-len(cfail)}/{n} = {(n-len(bad)-len(cfail))/n:.3f} "
          f"[{lo3:.3f}, {hi3:.3f}]")

    print(f"\nby modality (certificate only):")
    tab = {}
    for mod in ("vision", "text"):
        ids = [c for c, v in V.items() if v["modality"] == mod and c not in bad]
        f = sum(c in cfail for c in ids)
        lo, hi = wilson(len(ids) - f, len(ids))
        tab[mod] = (len(ids) - f, len(ids))
        print(f"  {mod:7} {len(ids)-f:3d}/{len(ids):3d} = {(len(ids)-f)/len(ids):.3f} [{lo:.2f}, {hi:.2f}]")
    (vk, vn), (tk, tn) = tab["vision"], tab["text"]
    p = stats.fisher_exact([[vn - vk, vk], [tn - tk, tk]])[1]
    print(f"  vision vs text certificate failure: Fisher p = {p:.3f}")

    print(f"\nby domain (certificate only):")
    for d in ("dci", "vistext", "semart", "roco"):
        ids = [c for c, v in V.items() if v["domain"] == d and c not in bad]
        if not ids:
            continue
        f = sum(c in cfail for c in ids)
        lo, hi = wilson(len(ids) - f, len(ids))
        print(f"  {d:8} {len(ids)-f:3d}/{len(ids):3d} = {(len(ids)-f)/len(ids):.3f} [{lo:.2f}, {hi:.2f}]")

    # ---- robustness: re-score distraction with every flagged item dropped ---------------------
    drop = bad | cfail
    print(f"\n=== distraction re-scored without the {len(drop)} flagged items ===")
    print(f"  {'backbone':20}{'v-distr':>9}{'excl.':>9}{'delta':>8}   {'t-distr':>9}{'excl.':>9}{'delta':>8}")
    dv, dt = [], []
    for tag, disp in MODELS:
        seen = {}
        for line in (DR / f"eval/t8/{tag}_w0.0.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line); seen.setdefault(r["candidate_id"], r)
        def rate(rows, mod, cond):
            s = [r for r in rows if r["label"] == mod and r[cond] == r["correct_index"]]
            return (sum(r["pred_vt"] != r["correct_index"] for r in s) / len(s)) if s else float("nan")
        allr = list(seen.values())
        keep = [r for r in allr if r["candidate_id"] not in drop]
        v0, v1 = rate(allr, "vision", "pred_v"), rate(keep, "vision", "pred_v")
        t0, t1 = rate(allr, "text", "pred_t"), rate(keep, "text", "pred_t")
        dv.append(v1 - v0); dt.append(t1 - t0)
        print(f"  {disp:20}{v0:>9.4f}{v1:>9.4f}{v1-v0:>+8.4f}   {t0:>9.4f}{t1:>9.4f}{t1-t0:>+8.4f}")
    print(f"  {'max |delta|':20}{max(abs(x) for x in dv):>27.4f}{max(abs(x) for x in dt):>35.4f}")

    out = DR / "diagnostics/human_validation.json"
    out.write_text(json.dumps(dict(
        n=n, n_cert_denom=cert_n, n_cert_fail=len(cfail), n_bad_key=len(bad),
        cert_rate=(cert_n - len(cfail)) / cert_n, bad_key_rate=len(bad) / n,
        cert_ci=list(wilson(cert_n - len(cfail), cert_n)),
        bad_key_ci=list(wilson(len(bad), n)),
        vision=list(tab["vision"]), text=list(tab["text"]), fisher_p=float(p),
        max_abs_delta_v=float(max(abs(x) for x in dv)),
        max_abs_delta_t=float(max(abs(x) for x in dt))), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
