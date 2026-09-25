"""P1-partial: independent cross-oracle re-certification of the 240-item audit sample.

Gemini answers each sample item under the same three conditions the original (Sonnet-track)
oracle used (V+T / V-only / T-only), and we compare its 3-condition signature against the
item's certified grounding label:
  vision-grounded expects  VT correct, V correct, T wrong
  text-grounded  expects  VT correct, V wrong,  T correct

Resumable (per condition file). Outputs under data/human_audit/gemini_rejudge/.

    python scripts/92_gemini_rejudge.py            # run answering (needs Gemini API key)
    python scripts/92_gemini_rejudge.py --report   # score agreement from saved answers
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("opa", ROOT / "scripts/oracle_pipeline_api.py")
opa = importlib.util.module_from_spec(_s); _s.loader.exec_module(opa)  # type: ignore

OUT = ROOT / "data/human_audit/gemini_rejudge"


def report():
    items = [json.loads(l) for l in (ROOT / "data/human_audit/sample.jsonl").read_text().splitlines() if l.strip()]
    preds = {m: opa.load_predictions(OUT / f"ans_{m}.jsonl") for m in ("vt", "v", "t")}
    n_ok = agree = sig_agree = 0
    per_label = {"vision": [0, 0], "text": [0, 0]}
    errors = 0
    for it in items:
        cid = it["candidate_id"]; ci = int(it["correct_index"])
        if any(cid not in preds[m] for m in ("vt", "v", "t")):
            errors += 1; continue
        n_ok += 1
        ok = {m: int(preds[m][cid]) == ci for m in ("vt", "v", "t")}
        want = {"vision": (True, True, False), "text": (True, False, True)}[it["label"]]
        match = (ok["vt"], ok["v"], ok["t"]) == want
        sig_agree += match
        per_label[it["label"]][0] += match; per_label[it["label"]][1] += 1
        # weaker criterion: grounded-modality behavior only (single-modality answerability)
        agree += (ok["v"] and not ok["t"]) if it["label"] == "vision" else (ok["t"] and not ok["v"])
    print(f"items with all 3 Gemini answers: {n_ok}/{len(items)} (missing/errors {errors})")
    print(f"full 3-condition signature agreement: {sig_agree}/{n_ok} = {sig_agree/max(n_ok,1):.1%}")
    print(f"single-modality-grounding agreement (V xor T): {agree}/{n_ok} = {agree/max(n_ok,1):.1%}")
    for lab, (a, n) in per_label.items():
        print(f"  {lab}: signature agreement {a}/{n} = {a/max(n,1):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    if args.report:
        report(); return
    items = [json.loads(l) for l in (ROOT / "data/human_audit/sample.jsonl").read_text().splitlines() if l.strip()]
    client = opa.make_client("gemini", model=args.model, key_path=args.key_path)
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("t", "v", "vt"):
        opa.run_answer_pass(client, items, OUT / f"ans_{mode}.jsonl", mode, max_workers=args.max_workers)
    report()


if __name__ == "__main__":
    main()
