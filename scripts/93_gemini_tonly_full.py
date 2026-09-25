"""W2: second-oracle T-only answering of ALL D_M vision items (text-only calls, no images).

Purpose: quantify per-item exclusivity leakage (a vision-grounded item a second oracle can solve
from caption+question alone) so the leaderboard / domain-fragility claims can be re-tabulated
excluding leaky items. Resumable.

    python scripts/93_gemini_tonly_full.py            # run
    python scripts/93_gemini_tonly_full.py --report   # leak rates per domain
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("opa", ROOT / "scripts/oracle_pipeline_api.py")
opa = importlib.util.module_from_spec(_s); _s.loader.exec_module(opa)  # type: ignore

OUT = ROOT / "data/diagnostics/gemini_tonly_full"


def items():
    rows = [json.loads(l) for l in (ROOT / "data/dm/multidomain_v1/dm_all.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["label"] == "vision"]


def report():
    from collections import Counter
    preds = opa.load_predictions(OUT / "ans_t.jsonl")
    solve, n = Counter(), Counter()
    leaky = []
    for r in items():
        cid = r["candidate_id"]
        if cid not in preds:
            continue
        n[r["source"]] += 1
        if int(preds[cid]) == int(r["correct_index"]):
            solve[r["source"]] += 1; leaky.append(cid)
    tot_n = sum(n.values()); tot_s = sum(solve.values())
    print(f"coverage {tot_n}/{len(items())}   T-solve overall {tot_s}/{tot_n} = {tot_s/max(tot_n,1):.1%} (chance 25%)")
    for d in sorted(n):
        print(f"  {d}: {solve[d]}/{n[d]} = {solve[d]/n[d]:.1%}")
    (OUT / "leaky_ids.json").write_text(json.dumps(sorted(leaky), indent=1))
    print(f"-> {OUT/'leaky_ids.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    if args.report:
        report(); return
    client = opa.make_client("gemini", model=args.model, key_path=args.key_path)
    OUT.mkdir(parents=True, exist_ok=True)
    opa.run_answer_pass(client, items(), OUT / "ans_t.jsonl", "t", max_workers=args.max_workers)
    report()


if __name__ == "__main__":
    main()
