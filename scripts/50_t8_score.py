"""T8 scoring: turn the per-model 3-condition predictions (scripts/49) into the
cross-model modality-robustness leaderboard.

Per model x domain reports, on items the model solves from the grounding modality:
  vision: V-only acc, V+T acc, v-distraction = P(V+T wrong | V-only correct)
  text  : T-only acc, V+T acc, t-distraction = P(V+T wrong | T-only correct)
plus the natural(dci)-vs-non-natural distraction split and the v/t asymmetry.

    python scripts/50_t8_score.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DOMAINS = ["dci", "vistext", "semart", "roco"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval/t8")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / "*.jsonl")))
    if not files:
        print(f"no prediction files in {args.dir} (run scripts/49 first)"); return

    for fp in files:
        model = Path(fp).stem
        rows = [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]
        ci = np.array([r["correct_index"] for r in rows])
        vt = np.array([r["pred_vt"] for r in rows]); v = np.array([r["pred_v"] for r in rows])
        t = np.array([r["pred_t"] for r in rows])
        lab = np.array([r["label"] for r in rows]); src = np.array([r.get("source") for r in rows])
        print(f"\n================ {model}  (n={len(rows)}) ================")
        print(f"{'domain':9}| {'nV':>4} {'V-only':>7} {'V+T':>6} {'v-distr':>8} | {'nT':>4} {'T-only':>7} {'V+T':>6} {'t-distr':>8}")
        agg = {}
        for d in DOMAINS + ["ALL"]:
            dm = np.ones(len(rows), bool) if d == "ALL" else (src == d)
            isV = dm & (lab == "vision"); isT = dm & (lab == "text")
            if d != "ALL" and isV.sum() == 0 and isT.sum() == 0:
                continue  # domain not present (e.g. assembled has no D_M source tags)
            vsolv = isV & (v == ci); tsolv = isT & (t == ci)
            vd = (vsolv & (vt != ci)).sum() / max(vsolv.sum(), 1)
            td = (tsolv & (vt != ci)).sum() / max(tsolv.sum(), 1)
            vt_v = (vt[isV] == ci[isV]).mean() if isV.sum() else float("nan")
            vt_t = (vt[isT] == ci[isT]).mean() if isT.sum() else float("nan")
            vonly = (v[isV] == ci[isV]).mean() if isV.sum() else float("nan")
            tonly = (t[isT] == ci[isT]).mean() if isT.sum() else float("nan")
            agg[d] = {"vd": vd, "td": td}
            print(f"{d:9}| {int(isV.sum()):>4} {vonly:>7.3f} {vt_v:>6.3f} {vd:>8.3f} "
                  f"| {int(isT.sum()):>4} {tonly:>7.3f} {vt_t:>6.3f} {td:>8.3f}")
        if all(d in agg for d in ["dci", "vistext", "semart", "roco"]):
            nat = agg["dci"]["vd"]; nonnat = np.mean([agg[d]["vd"] for d in ["vistext", "semart", "roco"]])
            print(f"  v-distraction: natural(dci) {nat:.3f}  vs  non-natural(mean) {nonnat:.3f}  "
                  f"(ratio {nonnat/max(nat,1e-9):.2f}x)")
        print(f"  ASYMMETRY (ALL): v-distraction {agg['ALL']['vd']:.3f}  vs  t-distraction {agg['ALL']['td']:.3f}")


if __name__ == "__main__":
    main()
