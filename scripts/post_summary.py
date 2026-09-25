"""Re-run compute_summary post-hoc for steering runs that crashed at the
summary step but completed all forward passes.

Usage:
  python scripts/post_summary.py --config configs/pilot_merged_aokvqa_racehigh.yaml \
      --root data/steering/qwen_merged_aokvqa_racehigh --pool test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from moground.steering import compute_summary, print_summary, load_oracle_passed, baseline_rows_needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True, help="parent dir containing variant subdirs")
    ap.add_argument("--pool", default="test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root = Path(cfg["paths"]["data_root"]); dm_dir = Path(cfg["paths"]["dm_dir"])
    all_rows = load_oracle_passed(data_root, baseline_rows_needed(args.pool), dm_dir=dm_dir)
    pool = [r for r in all_rows if r.get("__split") == args.pool]
    print(f"loaded pool: {len(pool)} ({args.pool} split of {dm_dir})")

    for vdir in sorted(Path(args.root).iterdir()):
        rj = vdir / "results.jsonl"
        if not rj.is_file(): continue
        rows = [json.loads(l) for l in rj.read_text().splitlines() if l.strip()]
        layers = sorted({r["layer"] for r in rows if r["layer"] != -1})
        alphas = sorted({r["alpha"] for r in rows if r["layer"] != -1})
        print(f"\n=== {vdir.name} ===  layers={layers}  alphas={alphas}")
        try:
            summary = compute_summary(rj, pool, layers, alphas)
        except Exception as e:
            print(f"  ! summary failed: {e}")
            continue
        json.dump(summary, open(vdir / "summary.json", "w"), indent=2)
        print(f"  wrote {vdir/'summary.json'}")


if __name__ == "__main__":
    main()
