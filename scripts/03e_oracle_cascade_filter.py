"""Step 3e (Sonnet path): apply the oracle cascade after subagent ans_t + ans_vt.

Reads `candidates.jsonl` + `work/ans_t/chunk_*.result.jsonl` + `work/ans_vt/...`.
Applies the same per-label oracle cascade as oracle_pipeline_api.py's default
mode, and writes `qwen_candidates.jsonl` in the dm_dir. The Qwen V-only
majority pass (scripts/04_qwen_v_only_majority.py) runs on that file next.

Use this when you generated with Sonnet subagents and want the new cascade.
For legacy 3-pass behaviour (all 3 modalities judged by Sonnet), continue
using 03d_oracle_filter_and_split.py.

Usage:
  python scripts/03e_oracle_cascade_filter.py --config configs/pilot_hardT_dci_batch6.yaml
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
from pathlib import Path

import yaml


# Reuse the same cascade function as the Gemini pipeline so both backends are
# byte-for-byte identical on the keep rule.
_spec = _ilu.spec_from_file_location(
    "_pp", Path(__file__).parent / "oracle_pipeline_api.py")
_pp = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pp)  # type: ignore
apply_oracle_cascade = _pp.apply_oracle_cascade


def load_predictions_from_chunks(work_dir: Path) -> dict[str, int]:
    """Collect predicted_index by candidate_id from any chunk_*.result.jsonl."""
    preds: dict[str, int] = {}
    for f in sorted(work_dir.glob("chunk_*.result.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "predicted_index" in r:
                preds[r["candidate_id"]] = int(r["predicted_index"])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--skip-vt-filter", action="store_true",
                    help="Skip the oracle V+T pass (same semantics as the Gemini "
                         "pipeline flag). Run only ans_t subagents in that case.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])

    candidates = [
        json.loads(l)
        for l in (dm_dir / "candidates.jsonl").read_text().splitlines() if l.strip()
    ]
    t_preds  = load_predictions_from_chunks(dm_dir / "work" / "ans_t")
    vt_preds = {} if args.skip_vt_filter else load_predictions_from_chunks(dm_dir / "work" / "ans_vt")

    out_path = dm_dir / "qwen_candidates.jsonl"
    n_q, n_v, n_t, n_skip = apply_oracle_cascade(
        candidates, t_preds, vt_preds, out_path, skip_vt=args.skip_vt_filter)
    print(f"candidates: {len(candidates)} | skipped (missing preds): {n_skip}")
    print(f"oracle-cascade survivors: {n_q} ({n_v} vision, {n_t} text)")
    print(f"-> {out_path}")
    print(f"\nNEXT (GPU): python scripts/04_qwen_v_only_majority.py \\")
    print(f"              --input {out_path} \\")
    print(f"              --output {dm_dir / 'qwen_v_only.jsonl'}")


if __name__ == "__main__":
    main()
