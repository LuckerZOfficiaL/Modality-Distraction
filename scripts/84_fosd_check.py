"""Empirical check of Proposition 1(ii)'s ordering assumption (FOSD).

The proposition assumes: if cell c' has higher grounding than cell c, then c''s solvable-item
margin distribution (m | m > 0) first-order stochastically dominates c's. We check this
WITHIN each backbone (margins share a logit scale only within a model), across its cells:
the 4 D_M domains + the assembled A-OKVQA vision cell where a cf-store exists.

Margins are computed exactly as in scripts/59 (V-condition logits: gold minus best other;
solvable iff m > 0; grounding g = P(m > 0)).

For every within-backbone cell pair ordered by grounding (higher-g = "hi"), we report:
  AUC        P(m_hi > m_lo) on solvable margins (Mann-Whitney); >0.5 = stochastically larger
  KS-viol    max_x [F_hi(x) - F_lo(x)]_+  (one-sided KS violation; 0 = exact dominance)
  dec-order  fraction of deciles q10..q90 with q_hi >= q_lo

Pairs with near-equal grounding (|dg| < 0.02) are reported separately (ordering is noise there).

    python scripts/84_fosd_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
# Backbones whose cf-store sits in data/activations/ but which are NOT part of the paper cohort.
# This function enumerates models by scanning that directory, so anything added there silently
# joins the cohort unless listed here (adding a backbone moves the ordered-pair count).
# MAINTENANCE HAZARD -- this has now broken the verifier twice. compute() enumerates backbones by
# SCANNING data/activations/, so ANY new cf-store silently joins the paper cohort and shifts these

# frac-stochastically-larger 0.88->0.84). Every non-cohort backbone added to data/activations/ MUST
# be listed here in the same commit that collects its store.
EXCLUDED = {# dropped from the paper
            # not in the cohort
            "OpenGVLab_InternVL3-14B-hf",     # 2nd large model (scale control), not in the cohort
            "mistralai_Mistral-Small-3.1-24B-Instruct-2503",  # 3rd large model (4th family), not in the cohort
            # size-vs-idiosyncrasy campaign (task #25) -- registered BEFORE any cf-store exists,
            # so these can never silently join the cohort scan (the 585/587 lesson):
            # family size point
            "Qwen_Qwen3-VL-30B-A3B-Instruct"}  # Qwen family size point (MoE: 30B total / 3B active)
DOMS = ["dci", "vistext", "semart", "roco"]
DM_SUB, ASM_SUB = "dm_multidomain_v1_cf", "merged_aokvqa_racehigh_cf"
MIN_N = 30      # min solvable items for a cell to enter
GAP = 0.02      # |dg| below this -> "near-tie" bucket


def split_sources():  # candidate_id -> source fallback (some cf-store indexes lack "source"), as in scripts/59
    dmdir = Path(yaml.safe_load((ROOT / "configs/pilot_multidomain_v1.yaml").read_text())["paths"]["dm_dir"])
    src = {}
    for sp in ("train", "val", "test"):
        p = dmdir / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); src[r["candidate_id"]] = r.get("source")
    return src


SRC = None


VIOL = set(json.load(open(DR / "diagnostics/assembled_caption_certification.json"))["violations"]) \
    if (DR / "diagnostics/assembled_caption_certification.json").exists() else set()


def load_margins(model, sub):
    global SRC
    st = DR / "activations" / model / sub
    if not (st / "letter_logits.npy").exists():
        return None
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    out = []  # (domain, margin)
    for i, r in enumerate(idx):
        if r["label"] != "vision" or (sub == ASM_SUB and r["candidate_id"] in VIOL):
            continue
        g = r["correct_index"]
        lv = ll[i, 1].astype(np.float64)
        m = lv[g] - np.delete(lv, g).max()
        if sub == DM_SUB:
            dom = r.get("source")
            if dom is None:
                if SRC is None:
                    SRC = split_sources()
                dom = SRC.get(r["candidate_id"])
            if dom not in DOMS:
                continue
        else:
            dom = "aokvqa"
        out.append((dom, float(m)))
    return out


def cdf_violation(hi, lo):
    grid = np.unique(np.concatenate([hi, lo]))
    F_hi = np.searchsorted(np.sort(hi), grid, side="right") / len(hi)
    F_lo = np.searchsorted(np.sort(lo), grid, side="right") / len(lo)
    return float(np.maximum(F_hi - F_lo, 0).max())


def auc(hi, lo):
    # P(m_hi > m_lo) + 0.5 P(=) via rank-sum
    comb = np.concatenate([hi, lo])
    r = comb.argsort().argsort().astype(np.float64) + 1  # ranks 1..N (ties ~ arbitrary, margins are floats)
    return float((r[: len(hi)].sum() - len(hi) * (len(hi) + 1) / 2) / (len(hi) * len(lo)))


def compute(verbose=False):
    """Return (ordered, near) pair records: (model, hi_dom, lo_dom, dg, AUC, KSviol, decile_frac)."""
    models = sorted(p.name for p in (DR / "activations").iterdir()
                    if (p / DM_SUB / "letter_logits.npy").exists()
                    and p.name not in EXCLUDED)
    if verbose:
        print(f"backbones with D_M cf-store: {len(models)}")
    ordered, near = [], []
    for mo in models:
        rows = load_margins(mo, DM_SUB) or []
        asm = load_margins(mo, ASM_SUB)
        if asm:
            rows += asm
        doms = sorted({d for d, _ in rows})
        cells = {}
        for d in doms:
            m = np.array([x for dd, x in rows if dd == d])
            pos = m[m > 0]
            if len(pos) >= MIN_N:
                cells[d] = (float((m > 0).mean()), pos)
        ds = sorted(cells, key=lambda d: cells[d][0])
        if verbose:
            print(f"\n== {mo} ==  cells: " + ", ".join(f"{d}(g={cells[d][0]:.3f},n+={len(cells[d][1])})" for d in ds))
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                lo_d, hi_d = ds[a], ds[b]
                dg = cells[hi_d][0] - cells[lo_d][0]
                hi, lo = cells[hi_d][1], cells[lo_d][1]
                A, V = auc(hi, lo), cdf_violation(hi, lo)
                q = np.linspace(0.1, 0.9, 9)
                dec = float(np.mean(np.quantile(hi, q) >= np.quantile(lo, q)))
                rec = (mo, hi_d, lo_d, dg, A, V, dec)
                (near if dg < GAP else ordered).append(rec)
                if verbose:
                    tag = "near-tie" if dg < GAP else ("OK" if A > 0.5 else "VIOLATION")
                    print(f"   {hi_d:>8} > {lo_d:<8} dg={dg:+.3f}  AUC={A:.3f}  KSviol={V:.3f}  dec={dec:.2f}  [{tag}]")
    return ordered, near


def main() -> None:
    ordered, near = compute(verbose=True)
    for name, R in (("ORDERED pairs (|dg|>=0.02)", ordered), ("near-tie pairs (|dg|<0.02)", near)):
        if not R:
            continue
        A = np.array([r[4] for r in R]); V = np.array([r[5] for r in R]); D = np.array([r[6] for r in R])
        print(f"\n==== {name}: n={len(R)} ====")
        print(f"  AUC>0.5 (stochastically larger): {(A > 0.5).mean():.2%}   mean AUC={A.mean():.3f}")
        print(f"  KS violation: median={np.median(V):.3f}  90th pct={np.quantile(V, .9):.3f}")
        print(f"  decile ordering: mean={D.mean():.2%}  all-9-ordered={np.mean(D == 1):.2%}")
        bad = [r for r in R if r[4] <= 0.5]
        if bad:
            print("  violations: " + "; ".join(f"{m}:{h}>{l}(dg={dg:+.3f},AUC={a:.3f})" for m, h, l, dg, a, v, d in bad))


if __name__ == "__main__":
    main()
