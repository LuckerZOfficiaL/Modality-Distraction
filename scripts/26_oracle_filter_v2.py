"""Step 26: oracle behavioral filter on the v2 candidates (object-matched + decoy).

Runs the SAME oracle 3-pass keep rule the original D_M used, on BOTH row types in
candidates_v2.jsonl (scripts/25): the T-counterparts (label=text) and the
decoy-augmented V rows (label=vision). Certifies correctness that the construction
alone cannot guarantee:
  text   (T-counterpart): VT correct  ∧  V-only WRONG    ∧  T-only correct
                          -> answer is in the caption; swapped object not visible.
  vision (V + decoy):     VT correct  ∧  V-only correct  ∧  T-only WRONG
                          -> answer still in the image; decoy did NOT leak it to text.

Then keeps only PAIRS (same source_v_cid) where BOTH the V and the T survive, so the
matched set stays balanced and paired — the official v2 set, parity with
official_matched_set.jsonl (v1).

Reuses run_answer_pass / load_predictions / apply_keep_rule from oracle_pipeline_api.
Gemini Batch API (50% off); images at max_dim=1024 for the VT and V passes only
(the T pass is caption-only -> no image budget).

    python scripts/26_oracle_filter_v2.py --key-path .credentials/google_gladia
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import yaml

from moground.oracle_clients import make_client

_spec = importlib.util.spec_from_file_location("_opa", Path(__file__).parent / "oracle_pipeline_api.py")
_opa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_opa)  # type: ignore
run_answer_pass = _opa.run_answer_pass
load_predictions = _opa.load_predictions
apply_keep_rule = _opa.apply_keep_rule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--v2-dir", default="data/dm/multidomain_v1/tcounterparts_v2/full")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--max-image-dim", type=int, default=1024)
    ap.add_argument("--batch-poll-seconds", type=int, default=60)
    args = ap.parse_args()

    yaml.safe_load(Path(args.config).read_text())  # validate config is loadable
    v2 = Path(args.v2_dir)
    cands = [json.loads(l) for l in (v2 / "candidates_v2.jsonl").read_text().splitlines() if l.strip()]
    n_v = sum(1 for c in cands if c["label"] == "vision")
    n_t = sum(1 for c in cands if c["label"] == "text")
    print(f"candidates_v2: {len(cands)} rows (V-decoy={n_v}, T={n_t})")

    client = make_client("gemini", model=args.model, key_path=args.key_path)
    work = v2 / "work" / "filter"
    for mode in ("vt", "v", "t"):
        run_answer_pass(client, cands, work / f"ans_{mode}" / "results.jsonl", mode,
                        max_workers=8, max_image_dim=args.max_image_dim,
                        use_batch=True, batch_poll_seconds=args.batch_poll_seconds)

    vt = load_predictions(work / "ans_vt" / "results.jsonl")
    v = load_predictions(work / "ans_v" / "results.jsonl")
    t = load_predictions(work / "ans_t" / "results.jsonl")

    surv_path = v2 / "official_v2_survivors.jsonl"
    n, sv, st, skipped = apply_keep_rule(cands, vt, v, t, surv_path, skip_vt=False)
    print(f"\noracle survivors: {n} (vision={sv}, text={st}; skipped {skipped} missing preds)")

    # keep only matched pairs where BOTH the V and T row survived
    by_cid = defaultdict(dict)
    for r in (json.loads(l) for l in surv_path.read_text().splitlines() if l.strip()):
        by_cid[r["source_v_cid"]][r["label"]] = r
    matched = []
    for cid, pair in by_cid.items():
        if "vision" in pair and "text" in pair:
            matched.append(pair["vision"]); matched.append(pair["text"])
    out = v2 / "official_matched_set_v2.jsonl"
    with out.open("w") as f:
        for r in matched:
            f.write(json.dumps(r) + "\n")
    n_pairs = len(matched) // 2
    print(f"matched pairs (both V and T survive): {n_pairs}  -> {out}  ({len(matched)} rows)")

    def acc(pred):
        d = [c for c in cands if c["candidate_id"] in pred]
        return sum(1 for c in d if pred[c["candidate_id"]] == c["correct_index"]) / max(len(d), 1)
    print(f"oracle acc on all candidates — vt={acc(vt):.3f}  v(image-only)={acc(v):.3f}  t(caption-only)={acc(t):.3f}")


if __name__ == "__main__":
    main()
