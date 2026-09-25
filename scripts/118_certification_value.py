r"""What certification buys the MEASUREMENT: certified MoGround vs the same pipeline without the gate.

Compares the 7 frozen backbones on two pools that differ only in whether the three-condition gate
was applied: MoGround (certified, keys {tag}_w0.0) and the uncertified counterfactual built by
scripts/117 from the gate's rejects (keys unc_{tag}_w0.0), stratified to the same domain x label
composition. Four readouts:

  E1  the grounding-strength relation (the paper's central regularity) recomputed on each pool
  E2  item informativeness: what fraction of items can enter the conditional at all, and how well
      items discriminate between backbones (classical point-biserial item analysis; with 7
      "persons" a 2PL is not identifiable, so we use the CTT statistic IRT reduces to here)
  E3  leaderboard stability: does the backbone ranking survive removing the gate
  E4  composition: what an uncertified "vision-grounded" item actually is, per backbone
  E5  discriminant validity: does the measured rate track grounding, or difficulty/leakage instead

    python scripts/118_certification_value.py            # both pools, all available backbones
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
RNG = np.random.default_rng(0)
DOMS = ["dci", "vistext", "semart", "roco"]
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3b", "Qwen2.5-VL-3B"),
          ("llavaov", "LLaVA-OV-7B"), ("next", "LLaVA-NeXT-8B"), ("qwen2b", "Qwen2-VL-2B"),
          ("llava15", "LLaVA-1.5-7B")]
# family, for the clustered bootstrap the paper uses on the relation
FAMILY = {"qwen7b": "qwen", "qwen3b": "qwen", "qwen2b": "qwen", "internvl": "internvl",
          "llavaov": "llava", "next": "llava", "llava15": "llava"}


def load(key):
    p = DR / f"eval/t8/{key}.jsonl"
    if not p.exists():
        return None
    seen = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); seen.setdefault(r["candidate_id"], r)
    return list(seen.values())


def arrays(rs):
    return (np.array([r["correct_index"] for r in rs]), np.array([r["label"] for r in rs]),
            np.array([r["pred_v"] for r in rs]), np.array([r["pred_t"] for r in rs]),
            np.array([r["pred_vt"] for r in rs]), np.array([r.get("source") for r in rs]))


def cells(rs, cap=None, rng=None):
    """Per domain x modality: grounding strength and the matching distraction rate.

    `cap` maps (domain, modality) -> item count; cells are subsampled to it. Used for the
    n-matched control: a correlation computed on fewer items per cell is attenuated by sampling
    noise alone, so the certified pool has to be thinned to the uncertified pool's cell sizes
    before the two correlations can be compared."""
    ci, lab, v, t, vt, src = arrays(rs)
    out = []
    for dom in DOMS:
        for mod, pred in (("vision", v), ("text", t)):
            m = (src == dom) & (lab == mod)
            if cap and (dom, mod) in cap:
                idx = np.flatnonzero(m)
                k = min(cap[(dom, mod)], len(idx))
                m = np.zeros_like(m)
                m[(rng or RNG).choice(idx, k, replace=False)] = True
            if m.sum() < 20:
                continue
            solv = m & (pred == ci)
            out.append({"domain": dom, "modality": mod, "n": int(m.sum()),
                        "ground": float((pred[m] == ci[m]).mean()),
                        "distr": float((solv & (vt != ci)).sum() / max(solv.sum(), 1)),
                        "n_solv": int(solv.sum())})
    return out


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def spearman(x, y):
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def clustered_ci(pts, B=5000):
    """Family-clustered bootstrap on the correlation, matching the paper's protocol."""
    fams = sorted({p["family"] for p in pts})
    by = {f: [p for p in pts if p["family"] == f] for f in fams}
    rs = []
    for _ in range(B):
        draw = [q for f in RNG.choice(fams, len(fams), replace=True) for q in by[f]]
        g = np.array([q["ground"] for q in draw]); d = np.array([q["distr"] for q in draw])
        if len(set(g)) > 2:
            rs.append(pearson(g, d))
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))) if rs else (np.nan, np.nan)


def _nopts(path):
    """candidate_id -> number of answer options, from the pool file (eval rows omit `options`)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["candidate_id"]] = len(r["options"])
    return out


NOPTS = {"certified": _nopts(DR / "dm/multidomain_v1/dm_all.jsonl"),
         "uncertified": _nopts(DR / "dm/uncertified_v1/splits/test.jsonl")}


def collect(prefix):
    """{tag: rows} for one pool; prefix '' = certified MoGround, 'unc_' = uncertified."""
    out = {}
    for tag, _ in MODELS:
        rs = load(f"{prefix}{tag}_w0.0")
        if rs is not None:
            out[tag] = rs
    return out


# ----------------------------------------------------------------------------- E1
def e1(pools):
    # `pools` is used for the n-matched control below as well as the per-pool fits
    print("\n" + "=" * 78)
    print("E1  grounding-strength relation, recomputed per pool (paper: r = -0.90 over 70 cells)")
    print("=" * 78)
    res = {}
    for name, P in pools.items():
        pts = []
        for tag, rs in P.items():
            for c in cells(rs):
                c.update(model=tag, family=FAMILY[tag]); pts.append(c)
        if len(pts) < 5:
            print(f"  {name}: only {len(pts)} cells, skipping"); continue
        g = np.array([p["ground"] for p in pts]); d = np.array([p["distr"] for p in pts])
        lo, hi = clustered_ci(pts)
        res[name] = {"n_cells": len(pts), "pearson": pearson(g, d), "spearman": spearman(g, d),
                     "ci": [lo, hi], "cells": pts,
                     "ground_range": [float(g.min()), float(g.max())], "ground_sd": float(g.std())}
        print(f"  {name:14} n={len(pts):>3} cells   Pearson r={pearson(g,d):+.3f}"
              f"   Spearman={spearman(g,d):+.3f}   family-clustered 95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {'':14} grounding spread: [{g.min():.3f}, {g.max():.3f}]  SD={g.std():.3f}")
    if len(res) == 2:
        # A weaker correlation can be an artifact of a narrower predictor range (classical
        # restriction-of-range attenuation), so also compare on the overlapping grounding window.
        A = res["certified"]["cells"]; B = res["uncertified"]["cells"]
        lo_g = max(min(p["ground"] for p in A), min(p["ground"] for p in B))
        hi_g = min(max(p["ground"] for p in A), max(p["ground"] for p in B))
        print(f"\n  range-matched control, grounding in [{lo_g:.3f}, {hi_g:.3f}]:")
        for nm, pts_ in (("certified", A), ("uncertified", B)):
            sub = [p for p in pts_ if lo_g <= p["ground"] <= hi_g]
            if len(sub) < 5:
                print(f"     {nm:12} n={len(sub)} (too few)"); continue
            gg = np.array([p["ground"] for p in sub]); dd = np.array([p["distr"] for p in sub])
            res[nm]["pearson_rangematched"] = pearson(gg, dd)
            res[nm]["n_cells_rangematched"] = len(sub)
            print(f"     {nm:12} n={len(sub):>3}   Pearson r={pearson(gg,dd):+.3f}")
    if len(res) == 2 and "certified" in pools and "uncertified" in pools:
        # n-matched control: thin certified cells to the uncertified cell sizes, B times
        cap = {}
        for tag, rs in pools["uncertified"].items():
            _, lab, _, _, _, src = arrays(rs)
            for dom in DOMS:
                for mod in ("vision", "text"):
                    n = int(((src == dom) & (lab == mod)).sum())
                    cap[(dom, mod)] = max(cap.get((dom, mod), 0), n)
        rng = np.random.default_rng(1)
        rs_boot = []
        for _ in range(200):
            pts = []
            for tag, rows in pools["certified"].items():
                for c in cells(rows, cap=cap, rng=rng):
                    c.update(model=tag, family=FAMILY[tag]); pts.append(c)
            g = np.array([p["ground"] for p in pts]); d = np.array([p["distr"] for p in pts])
            rs_boot.append(pearson(g, d))
        m_, l_, h_ = float(np.mean(rs_boot)), *np.percentile(rs_boot, [2.5, 97.5])
        res["certified"]["pearson_nmatched"] = m_
        res["certified"]["pearson_nmatched_ci"] = [float(l_), float(h_)]
        print(f"\n  n-matched control (certified thinned to the uncertified cell sizes, 200 draws):")
        print(f"     certified   Pearson r = {m_:+.3f}  [{l_:+.3f}, {h_:+.3f}]")
        print(f"     uncertified Pearson r = {res['uncertified']['pearson']:+.3f}"
              f"   -> {'OUTSIDE' if not (l_ <= res['uncertified']['pearson'] <= h_) else 'INSIDE'}"
              f" the n-matched certified interval")
    if len(res) == 2:
        a, b = res["certified"], res["uncertified"]
        print(f"\n  -> relation strength |r|: certified {abs(a['pearson']):.3f}"
              f" vs uncertified {abs(b['pearson']):.3f}"
              f"   ({100*(1-abs(b['pearson'])/abs(a['pearson'])):+.0f}% change in |r|)")
        print(f"  -> uncertified CI {'EXCLUDES' if (b['ci'][1] < 0) else 'INCLUDES'} 0"
              f"; certified CI {'EXCLUDES' if (a['ci'][1] < 0) else 'INCLUDES'} 0")
    return res


# ----------------------------------------------------------------------------- E2
def e2(pools):
    print("\n" + "=" * 78)
    print("E2  item informativeness: can the item enter the conditional, and does it discriminate?")
    print("=" * 78)
    res = {}
    for name, P in pools.items():
        entered, disc, zerovar = [], [], []
        # response matrix over the items every backbone was scored on
        common = sorted(set.intersection(*[{r["candidate_id"] for r in rs} for rs in P.values()]))
        idx = {c: i for i, c in enumerate(common)}
        M = np.zeros((len(P), len(common)))
        for k, (tag, rs) in enumerate(sorted(P.items())):
            byid = {r["candidate_id"]: r for r in rs}
            for c in common:
                r = byid[c]
                M[k, idx[c]] = int(r["pred_vt"] == r["correct_index"])
            ci, lab, v, t, vt, _ = arrays(rs)
            pred = np.where(lab == "vision", v, t)
            entered.append(float((pred == ci).mean()))          # fraction usable as a conditional
        ability = M.mean(1)
        for j in range(M.shape[1]):
            col = M[:, j]
            if col.std() == 0:
                zerovar.append(1); disc.append(0.0)
            else:
                zerovar.append(0); disc.append(abs(pearson(col, ability)))
        res[name] = {"entered_mean": float(np.mean(entered)), "n_items": len(common),
                     "discrimination_mean": float(np.mean(disc)),
                     "zero_variance_frac": float(np.mean(zerovar))}
        print(f"  {name:14} items={len(common):>5}"
              f"   enters conditional: {100*np.mean(entered):5.1f}%"
              f"   mean |discrimination|: {np.mean(disc):.3f}"
              f"   all-backbones-identical: {100*np.mean(zerovar):5.1f}%")
    return res


# ----------------------------------------------------------------------------- E3
def e3(pools):
    print("\n" + "=" * 78)
    print("E3  leaderboard stability: does the backbone ranking survive removing the gate?")
    print("=" * 78)
    rates = {}
    for name, P in pools.items():
        r = {}
        for tag, rs in P.items():
            ci, lab, v, t, vt, _ = arrays(rs)
            m = (lab == "vision") & (v == ci)
            r[tag] = float((m & (vt != ci)).sum() / max(m.sum(), 1))
        rates[name] = r
    if len(rates) < 2:
        return {}
    tags = sorted(set(rates["certified"]) & set(rates["uncertified"]))
    a = np.array([rates["certified"][t] for t in tags])
    b = np.array([rates["uncertified"][t] for t in tags])
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    disp = dict(MODELS)
    print(f"  {'backbone':17}{'certified':>11}{'rank':>6}{'uncertified':>13}{'rank':>6}")
    for i, t in enumerate(tags):
        print(f"  {disp[t]:17}{a[i]:>11.4f}{ra[i]+1:>6}{b[i]:>13.4f}{rb[i]+1:>6}")
    rho = spearman(a, b)
    conc = sum(np.sign(a[i]-a[j]) == np.sign(b[i]-b[j])
               for i in range(len(tags)) for j in range(i+1, len(tags)))
    npairs = len(tags)*(len(tags)-1)//2
    print(f"\n  -> Spearman rank correlation of the two leaderboards: {rho:+.3f}")
    print(f"  -> concordant pairs: {conc}/{npairs}  (Kendall tau = {2*conc/npairs - 1:+.3f})")
    return {"rates": rates, "spearman": rho, "kendall_tau": 2*conc/npairs - 1}


# ----------------------------------------------------------------------------- E4
def e4(pools):
    print("\n" + "=" * 78)
    print("E4  what an 'intended-vision' item actually is (per backbone, mean over the 7)")
    print("=" * 78)
    res = {}
    for name, P in pools.items():
        acc = {"leak_T_answerable": [], "not_V_answerable": [], "clean_and_robust": [],
               "clean_and_flipped": []}
        chance = []
        for tag, rs in P.items():
            ci, lab, v, t, vt, _ = arrays(rs)
            m = lab == "vision"
            if not m.sum():
                continue
            Vok, Tok, VTok = (v[m] == ci[m]), (t[m] == ci[m]), (vt[m] == ci[m])
            acc["leak_T_answerable"].append(float(Tok.mean()))
            acc["not_V_answerable"].append(float((~Vok).mean()))
            acc["clean_and_robust"].append(float((Vok & ~Tok & VTok).mean()))
            acc["clean_and_flipped"].append(float((Vok & ~Tok & ~VTok).mean()))
            nopt = NOPTS[name]                # eval rows don't carry options; read the pool file
            chance += [1 / nopt[r["candidate_id"]] for r in rs
                       if r["label"] == "vision" and r["candidate_id"] in nopt]
        res[name] = {k: float(np.mean(v)) for k, v in acc.items()}
        res[name]["chance"] = float(np.mean(chance))
        # A certified vision item is unanswerable from its caption BY CONSTRUCTION, so a backbone
        # scoring it text-only should sit at guessing. Anything above chance is real leakage.
        res[name]["leak_above_chance_pp"] = 100 * (res[name]["leak_T_answerable"] - res[name]["chance"])
        print(f"  {name}:  (chance = {100*res[name]['chance']:.1f}%)")
        print(f"     answerable from the CAPTION alone (leak)      {100*res[name]['leak_T_answerable']:5.1f}%"
              f"   [{res[name]['leak_above_chance_pp']:+.1f}pp vs chance]")
        print(f"     NOT answerable from the image alone           {100*res[name]['not_V_answerable']:5.1f}%")
        print(f"     clean vision item, survives the caption       {100*res[name]['clean_and_robust']:5.1f}%")
        print(f"     clean vision item, genuinely distracted       {100*res[name]['clean_and_flipped']:5.1f}%")
    # Per domain: E0 (scripts/119) measures off-modality answerability exactly, but only where the
    # oracle's per-condition answers were kept (SemArt, ROCO). Agreement there licenses reading the
    # backbone-measured leakage on DCI and VisText, where no oracle record survives.
    print("\n  caption leakage above chance, by domain (backbone-measured, mean over backbones):")
    print(f"     {'domain':10}{'certified':>12}{'uncertified':>14}")
    perdom = {}
    for dom in DOMS:
        row = {}
        for name, P in pools.items():
            vals = []
            for tag, rs in P.items():
                ci, lab, v, t, vt, src = arrays(rs)
                m = (src == dom) & (lab == "vision")
                if m.sum() < 20:
                    continue
                ch = np.mean([1 / NOPTS[name][r["candidate_id"]] for r in rs
                              if r.get("source") == dom and r["label"] == "vision"
                              and r["candidate_id"] in NOPTS[name]])
                vals.append(float((t[m] == ci[m]).mean()) - float(ch))
            if vals:
                row[name] = 100 * float(np.mean(vals))
        if row:
            perdom[dom] = row
            print(f"     {dom:10}{row.get('certified', float('nan')):>+11.1f}pp"
                  f"{row.get('uncertified', float('nan')):>+13.1f}pp")
    for name in res:
        res[name]["per_domain_leak_above_chance_pp"] = {d: v.get(name) for d, v in perdom.items()}
    if len(res) == 2:
        a, b = res["certified"], res["uncertified"]
        print(f"\n  -> caption leakage above chance: certified {a['leak_above_chance_pp']:+.1f}pp"
              f"  vs uncertified {b['leak_above_chance_pp']:+.1f}pp")
        print(f"  -> items that cannot enter a vision conditional at all: "
              f"certified {100*a['not_V_answerable']:.1f}%  vs uncertified {100*b['not_V_answerable']:.1f}%")
    return res


# ----------------------------------------------------------------------------- E5
def e5(pools):
    """Discriminant validity: does the measured rate track what it claims to, or a confound?

    v-distraction is meant to track grounding strength in the target modality. Two rival things it
    could track instead on an ungated pool: raw item difficulty, and off-modality answerability
    (a 'vision' item that is really answerable from its caption). On certified items the gate
    forces off-modality accuracy to guessing, so that channel is closed by construction; the point
    of this check is whether it re-opens once the gate is removed."""
    print("\n" + "=" * 78)
    print("E5  discriminant validity: what does the measured rate actually track?")
    print("=" * 78)
    res = {}
    for name, P in pools.items():
        rows = []
        for tag, rs in P.items():
            ci, lab, v, t, vt, src = arrays(rs)
            for dom in DOMS:
                for mod, pred, off in (("vision", v, t), ("text", t, v)):
                    m = (src == dom) & (lab == mod)
                    if m.sum() < 20:
                        continue
                    solv = m & (pred == ci)
                    if solv.sum() < 10:
                        continue
                    rows.append({
                        "distr": float((solv & (vt != ci)).sum() / solv.sum()),
                        "ground": float((pred[m] == ci[m]).mean()),
                        "difficulty": float(1 - (vt[m] == ci[m]).mean()),
                        "offmod": float((off[m] == ci[m]).mean()),
                    })
        if len(rows) < 6:
            print(f"  {name}: {len(rows)} cells, too few"); continue
        d = np.array([r["distr"] for r in rows])
        out = {k: pearson(np.array([r[k] for r in rows]), d)
               for k in ("ground", "difficulty", "offmod")}
        out["n_cells"] = len(rows)
        out["offmod_mean"] = float(np.mean([r["offmod"] for r in rows]))
        res[name] = out
        print(f"  {name:14} n={len(rows):>3} cells")
        print(f"     corr(distraction, grounding strength)      {out['ground']:+.3f}   <- the intended construct")
        print(f"     corr(distraction, raw item difficulty)     {out['difficulty']:+.3f}"
              f"   (context only: near-collinear with grounding by construction)")
        print(f"     corr(distraction, off-modality accuracy)   {out['offmod']:+.3f}"
              f"   <- the discriminant test (mean off-modality acc {out['offmod_mean']:.3f})")
    if len(res) == 2:
        a, b = res["certified"], res["uncertified"]
        # Difficulty is 1 - VT accuracy and grounding is single-modality accuracy, so the two are
        # near-mirror images; only off-modality answerability is a genuinely rival explanation,
        # and the gate is what forces it to guessing.
        print(f"\n  -> off-modality contamination: certified r={a['offmod']:+.3f} at mean acc "
              f"{a['offmod_mean']:.3f}   vs uncertified r={b['offmod']:+.3f} at mean acc "
              f"{b['offmod_mean']:.3f}")
    return res


def main():
    pools = {}
    for name, prefix in (("certified", ""), ("uncertified", "unc_")):
        P = collect(prefix)
        if P:
            pools[name] = P
        print(f"{name:14} backbones loaded: {len(P)}/7  {sorted(P)}")
    if "uncertified" not in pools:
        print("\n[partial] no uncertified evals yet -- reporting the certified side only, so the "
              "certified numbers can be checked against the paper before the comparison lands")
    out = {"E1": e1(pools), "E2": e2(pools), "E3": e3(pools), "E4": e4(pools), "E5": e5(pools)}
    for k in out.get("E1", {}):
        out["E1"][k].pop("cells", None)          # keep the summary file small
    p = DR / "diagnostics/certification_value.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
