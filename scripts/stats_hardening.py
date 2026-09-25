"""P4: statistical hardening of the paper's headline correlations.

1) Grounding-strength law r: bootstrap CIs (cell-level binomial noise; cluster bootstrap over
   models; cluster bootstrap over model FAMILIES for the LLaVA-family confound), plus the
   deconfounded variant EXCLUDING the 32 D_M text cells that sit at (ground~1, distr~0) by
   construction (certified-saturated corner).
2) Asymmetry-tracks-gap r (n=8 models): item-level bootstrap over the assembled per-item
   predictions -> distribution of the 8 (gap, asymmetry) points -> CI on r; plus per-model
   bootstrap CIs on the asymmetry itself (Table 3 noise bands).

    python scripts/stats_hardening.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
RNG = np.random.default_rng(0)
B = 2000

# canonical (current-environment) base-run keys, grouped by LLM family for the clustered bootstrap
FAMILY = {"qwen3b_w0.0": "qwen", "qwen7b_w0.0": "qwen", "qwen2b_w0.0": "qwen",
          "llava15_w0.0": "llava", "next_w0.0": "llava",
          "llavaov_w0.0": "llava", "internvl_w0.0": "internvl"}


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


OUT = {}


def law():
    pts = [p for p in json.load(open(DR / "diagnostics/grounding_strength.json"))
           if p["model"] in FAMILY]  # FAMILY excludes models dropped from the paper
    g = np.array([p["ground"] for p in pts]); d = np.array([p["distr"] for p in pts])
    n = np.array([p["n"] for p in pts]); mo = np.array([p["model"] for p in pts])
    dm_text = np.array([(p["dataset"] == "dm") and (p["modality"] == "text") for p in pts])
    vis = np.array([p["modality"] == "vision" for p in pts])
    fam = np.array([FAMILY[m] for m in mo])

    def r_of(mask):
        return float(np.corrcoef(g[mask], d[mask])[0, 1])

    subsets = {f"all {len(pts)} cells": np.ones(len(pts), bool),
               f"excl D_M text ({int((~dm_text).sum())}, deconfounded)": ~dm_text,
               f"vision cells only ({int(vis.sum())})": vis}
    print("=== grounding-strength law: r with bootstrap CIs ===")
    for name, mask in subsets.items():
        # (a) cell-noise bootstrap: resample each cell's rates binomially
        rs_cell = []
        for _ in range(B):
            gb = RNG.binomial(n[mask], np.clip(g[mask], 0, 1)) / n[mask]
            ns = np.maximum((n[mask] * g[mask]).astype(int), 1)   # solvable count approximates distr denominator
            db = RNG.binomial(ns, np.clip(d[mask], 0, 1)) / ns
            rs_cell.append(np.corrcoef(gb, db)[0, 1])
        # (b) cluster bootstrap over models
        models = sorted(set(mo[mask])); rs_mod = []
        for _ in range(B):
            pick = RNG.choice(models, len(models), replace=True)
            idx = np.concatenate([np.where(mask & (mo == m))[0] for m in pick])
            if len(set(g[idx])) > 2:
                rs_mod.append(np.corrcoef(g[idx], d[idx])[0, 1])
        # (c) cluster bootstrap over families
        fams = sorted(set(fam[mask])); rs_fam = []
        for _ in range(B):
            pick = RNG.choice(fams, len(fams), replace=True)
            idx = np.concatenate([np.where(mask & (fam == f))[0] for f in pick])
            if len(set(g[idx])) > 2:
                rs_fam.append(np.corrcoef(g[idx], d[idx])[0, 1])
        print(f"{name:36} r={r_of(mask):+.3f}   cell-CI [{ci(rs_cell)[0]:+.3f},{ci(rs_cell)[1]:+.3f}]   "
              f"model-cluster CI [{ci(rs_mod)[0]:+.3f},{ci(rs_mod)[1]:+.3f}]   "
              f"family-cluster CI [{ci(rs_fam)[0]:+.3f},{ci(rs_fam)[1]:+.3f}]")
        OUT[name] = {"r": r_of(mask), "cell_ci": ci(rs_cell), "model_ci": ci(rs_mod), "family_ci": ci(rs_fam)}


def asymmetry():
    files = {Path(f).stem: f for f in glob.glob(str(DR / "eval/t8_assembled/*.jsonl"))
             if Path(f).stem in FAMILY}
    viol = set(json.load(open(DR / "diagnostics/assembled_caption_certification.json"))["violations"])
    data = {}
    for m, f in files.items():
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        data[m] = {r["candidate_id"]: r for r in rows if r["candidate_id"] not in viol}
    common = set.intersection(*[set(d) for d in data.values()])
    print(f"\n=== asymmetry~gap on assembled (common n={len(common)}) ===")
    ids = sorted(common)

    def stats(rows):
        ci_ = np.array([r["correct_index"] for r in rows]); lab = np.array([r["label"] for r in rows])
        vt = np.array([r["pred_vt"] for r in rows]); v = np.array([r["pred_v"] for r in rows])
        t = np.array([r["pred_t"] for r in rows])
        isV, isT = lab == "vision", lab == "text"
        vg = (v[isV] == ci_[isV]).mean(); tg = (t[isT] == ci_[isT]).mean()
        vs = isV & (v == ci_); ts = isT & (t == ci_)
        vd = (vs & (vt != ci_)).sum() / max(vs.sum(), 1); td = (ts & (vt != ci_)).sum() / max(ts.sum(), 1)
        return tg - vg, vd - td            # gap, asymmetry

    base = {m: stats([data[m][i] for i in ids]) for m in data}
    gaps = np.array([base[m][0] for m in sorted(data)]); asy = np.array([base[m][1] for m in sorted(data)])
    r0 = float(np.corrcoef(gaps, asy)[0, 1])
    # item bootstrap -> r distribution + per-model asymmetry CIs
    rs, per_model = [], {m: [] for m in data}
    for _ in range(B):
        bids = [ids[k] for k in RNG.integers(0, len(ids), len(ids))]
        gb, ab = [], []
        for m in sorted(data):
            gp, ay = stats([data[m][i] for i in bids])
            gb.append(gp); ab.append(ay); per_model[m].append(ay)
        rs.append(np.corrcoef(gb, ab)[0, 1])
    lo, hi = ci(rs)
    print(f"r(asymmetry, gap) = {r0:+.3f}   item-bootstrap CI [{lo:+.3f},{hi:+.3f}]   P(r>0)={np.mean(np.array(rs)>0):.3f}")
    OUT["asym_gap"] = {"r": r0, "ci": [lo, hi], "p_pos": float(np.mean(np.array(rs) > 0))}
    OUT["per_model_asym"] = {}
    print("\nper-model asymmetry (v-distr - t-distr) with 95% CI:")
    for m in sorted(data, key=lambda x: base[x][0]):
        l, h = ci(per_model[m])
        sig = "" if l <= 0 <= h else "  *excludes 0*"
        OUT["per_model_asym"][m] = {"gap": base[m][0], "asym": base[m][1], "ci": [l, h],
                                    "excludes_zero": not (l <= 0 <= h)}
        print(f"  {m:44} gap={base[m][0]:+.3f}  asym={base[m][1]:+.3f}  CI [{l:+.3f},{h:+.3f}]{sig}")


if __name__ == "__main__":
    law()
    asymmetry()
    out = DR / "diagnostics" / "stats_hardening.json"
    out.write_text(json.dumps(OUT, indent=2))
    print(f"\n-> {out}")
