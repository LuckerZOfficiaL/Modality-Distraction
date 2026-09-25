"""H2-style margin decomposition on the assembled pool for one backbone tag.

Distinguishes the two known assembled-null mechanisms (and healthy repair):
    the clean 97.7% (frac improved 0.466, delta_nm_clean -0.0196), logit scale halved.
  * Mistral-24B  = SUB-THRESHOLD REPAIR: helps broadly (frac 0.618), harms nothing, but the
    distracted-item repair (+0.14) sits below every transferring backbone's (+0.15..+0.30).
  * transferring cohort: delta_nm_all +0.011..+0.031, frac 0.54..0.72, distracted +0.15..+0.30.

Conventions (identical to data/diagnostics/mistral24b_h2_margin.json, which this script
reproduces exactly -- its correctness proof):
  * denominator: assembled HELD minus certified VIOLations, label==vision, base pred_v correct
    (own-set BASE V-correct), first-wins de-dup;
  * nm = (logit_correct - max_other) / (max - min) over the VT letter logits -- scale-invariant,
    because the 27B's adapter HALVES its logit scale and raw margins are meaningless there;
  * per seed: delta_nm averaged over items; frac_improved = share of items with nm_seed > nm_base;
    "distracted" = base pred_vt wrong on the denominator; "clean" = the rest;
  * 4-seed mean of each per-seed statistic.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"


def load(fp: Path) -> dict:
    seen = {}
    for line in fp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen.setdefault(r["candidate_id"], r)
    return seen


def nm(row) -> float:
    lg = np.array(row["logits_vt"], dtype=float)
    ci = row["correct_index"]
    other = np.delete(lg, ci)
    spread = lg.max() - lg.min()
    return float((lg[ci] - other.max()) / spread) if spread > 0 else 0.0


def spread_median(rows, ids) -> float:
    return float(np.median([max(rows[c]["logits_vt"]) - min(rows[c]["logits_vt"]) for c in ids]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None, help="default data/diagnostics/<tag>_h2_margin.json")
    args = ap.parse_args()
    tag = args.tag

    held = set(json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json")
                          .read_text())["heldout"])
    viol = set(json.loads((DR / "diagnostics/assembled_caption_certification.json")
                          .read_text())["violations"])

    base = load(DR / f"eval/t8_assembled/{tag}_w0.0.jsonl")
    den = [c for c, r in base.items()
           if c in held and c not in viol and r["label"] == "vision"
           and r.get("pred_v") == r["correct_index"]]
    distracted = [c for c in den if base[c]["pred_vt"] != base[c]["correct_index"]]
    clean = [c for c in den if c not in set(distracted)]
    base_nm = {c: nm(base[c]) for c in den}

    per_seed = {}
    for s in range(4):
        key = f"{tag}_w0.5" if s == 0 else f"{tag}_s{s}_w0.5"
        fp = DR / f"eval/t8_assembled/{key}.jsonl"
        if not fp.exists():
            print(f"missing seed cell {fp} -- refusing to write a partial decomposition")
            return 1
        rows = load(fp)
        missing = [c for c in den if c not in rows]
        assert not missing, f"s{s}: {len(missing)} denominator items missing from {fp}"
        d = np.array([nm(rows[c]) - base_nm[c] for c in den])
        per_seed[f"s{s}"] = {
            "delta_nm_all": float(d.mean()),
            "frac_improved": float((d > 0).mean()),
            "delta_nm_distracted": float(np.mean([nm(rows[c]) - base_nm[c] for c in distracted]))
            if distracted else float("nan"),
            "delta_nm_clean": float(np.mean([nm(rows[c]) - base_nm[c] for c in clean])),
            "median_logit_spread": spread_median(rows, den),
        }

    keys = ["delta_nm_all", "frac_improved", "delta_nm_distracted", "delta_nm_clean",
            "median_logit_spread"]
    out = {
        "what": (f"H2-style margin decomposition on assembled (HELD, non-VIOL, base own-set "
                 f"V-correct). nm=(logit_correct - max_other)/(max-min) on VT logits; deltas are "
                 f"seed - base, per item, averaged over items then over 4 seeds. Reference "
                 f"distracted +0.3828, clean -0.0196, logit scale halved 29.0->13.8; "
                 f"Mistral-24B (sub-threshold): +0.0047 / 0.618 / +0.1413 / +0.0016, scale "
                 f"unchanged; transferring cohort: +0.011..+0.031 / 0.54..0.72 / +0.15..+0.30."),
        "n": {"V_correct": len(den), "base_distracted": len(distracted), "clean": len(clean)},
        "per_seed": per_seed,
        "base_median_logit_spread": spread_median(base, den),
        "four_seed_mean": {k: float(np.mean([per_seed[f"s{s}"][k] for s in range(4)]))
                           for k in keys},
    }
    outp = Path(args.out) if args.out else DR / f"diagnostics/{tag}_h2_margin.json"
    outp.write_text(json.dumps(out, indent=1))
    m = out["four_seed_mean"]
    print(f"{tag}: n={len(den)} (distracted {len(distracted)}) | delta_nm_all {m['delta_nm_all']:+.4f} "
          f"| frac_improved {m['frac_improved']:.3f} | distracted {m['delta_nm_distracted']:+.4f} "
          f"| clean {m['delta_nm_clean']:+.4f} | spread {out['base_median_logit_spread']:.1f} -> "
          f"{m['median_logit_spread']:.1f}  -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
