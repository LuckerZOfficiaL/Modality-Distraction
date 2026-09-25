"""Stress-test assumption (A1) of Proposition 1: is the distractor perturbation margin-independent?

Proposition 1 assumes the perturbation delta = margin_VT - margin_grounded is distributed
independently of the pre-existing margin m and of the cell c.  scripts/59 checks corr(delta, m)
pooled; this script does the thing the assumption actually needs:

  (1) the delta DISTRIBUTION stratified by margin quintile x backbone x domain x modality,
      with a KS test between the lowest and highest margin quintile inside each backbone;
  (2) a SENSITIVITY analysis: re-run the margin-model cell prediction twice, once under A1
      (one global/per-backbone F_delta) and once with a margin-quantile-CONDITIONAL F_delta
      that lets A1 be violated exactly as much as the data says it is.  If the cross-cell
      correlation and MAE barely move, the violation is immaterial to the claim; if they move,
      that is the honest size of the problem.

Margins follow scripts/59 exactly: condition 0 = VT, 1 = V, 2 = T; margin = gold - best other;
an item is solvable in its grounded modality iff that margin > 0; it flips iff margin_VT < 0.

Offline, CPU only.

    python scripts/100_perturbation_invariance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
DOMS = ["dci", "vistext", "semart", "roco"]
DM_SUB, ASM_SUB = "dm_multidomain_v1_cf", "merged_aokvqa_racehigh_cf"
NQ = 5          # margin quantile strata
MIN_N = 40      # min items for a stratum/cell to be reported


def load_pool(sub, pool):
    """-> list of dicts: model, domain, modality, m, delta, flip."""
    dmdir = Path(yaml.safe_load((ROOT / "configs/pilot_multidomain_v1.yaml").read_text())["paths"]["dm_dir"])
    src = {}
    for sp in ("train", "val", "test"):
        p = dmdir / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); src[r["candidate_id"]] = r.get("source")

    viol = set()
    if pool == "assembled":
        viol = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])

    out = []
    for mdir in sorted((DR / "activations").iterdir()):
        m = mdir.name
        if m in EXCLUDED or not (mdir / sub / "letter_logits.npy").exists():
            continue
        ll = np.load(mdir / sub / "letter_logits.npy").astype(np.float64)
        idx = [json.loads(l) for l in (mdir / sub / "index.jsonl").read_text().splitlines() if l.strip()]
        for i, r in enumerate(idx):
            if r["candidate_id"] in viol:
                continue
            lab = r["label"]
            dom = "aokvqa" if pool == "assembled" and lab == "vision" else \
                  "race" if pool == "assembled" else (r.get("source") or src.get(r["candidate_id"]))
            if pool == "dm" and dom not in DOMS:
                continue
            cond = 1 if lab == "vision" else 2          # grounded-modality condition
            g = int(r["correct_index"])
            lg, lvt = ll[i, cond], ll[i, 0]
            mg = lg[g] - np.delete(lg, g).max()
            mvt = lvt[g] - np.delete(lvt, g).max()
            if mg <= 0:                                  # not solvable from the grounded modality
                continue
            out.append(dict(model=m, domain=dom, modality=lab, m=float(mg),
                            delta=float(mvt - mg), flip=int(mvt < 0)))
    return out


def qbins(x, nq):
    """Quantile bin index 0..nq-1 (ties broken by rank so bins are balanced)."""
    r = stats.rankdata(x, method="ordinal")
    return np.minimum((r - 1) * nq // len(x), nq - 1)


def report(rows, pool, res):
    mdl = np.array([r["model"] for r in rows]); dom = np.array([r["domain"] for r in rows])
    mod = np.array([r["modality"] for r in rows]); m = np.array([r["m"] for r in rows])
    d = np.array([r["delta"] for r in rows]); fl = np.array([r["flip"] for r in rows])
    print(f"\n{'='*100}\nPOOL = {pool}   n={len(rows)}  models={len(set(mdl))}  "
          f"cells={len(set(zip(mdl, dom, mod)))}\n{'='*100}")

    # ---------- (1) delta stratified by margin quintile, within backbone x modality ----------
    print(f"\n(1) delta distribution by margin quintile (within backbone x modality)\n")
    print(f"    {'backbone/mod':28}{'n':>6}{'corr(d,m)':>11}{'Q1 mean d':>11}{'Q5 mean d':>11}"
          f"{'Q1 sd':>8}{'Q5 sd':>8}{'KS(Q1,Q5)':>11}{'p':>9}")
    strat = {}
    for mm in sorted(set(mdl)):
        for mo in ("vision", "text"):
            k = (mdl == mm) & (mod == mo)
            if k.sum() < MIN_N * NQ // 2:
                continue
            mk, dk = m[k], d[k]
            q = qbins(mk, NQ)
            strat[(mm, mo)] = (mk, dk, q)
            r = np.corrcoef(dk, mk)[0, 1]
            d1, d5 = dk[q == 0], dk[q == NQ - 1]
            ks, p = stats.ks_2samp(d1, d5)
            res.setdefault("strata", []).append(dict(pool=pool, model=mm, modality=mo, n=int(k.sum()),
                                                     corr=float(r), ks=float(ks), p=float(p),
                                                     q1_mean=float(d1.mean()), q5_mean=float(d5.mean())))
            print(f"    {mm+'/'+mo:28}{k.sum():>6}{r:>+11.3f}{d1.mean():>+11.2f}{d5.mean():>+11.2f}"
                  f"{d1.std():>8.2f}{d5.std():>8.2f}{ks:>11.3f}{p:>9.2g}")

    # per-domain view (vision only; the D_M text side is saturated)
    if pool == "dm":
        print(f"\n    per-domain corr(delta, m), vision items:")
        print(f"    {'domain':12}" + "".join(f"{mm[:10]:>12}" for mm in sorted(set(mdl))))
        for dd in DOMS:
            line = f"    {dd:12}"
            for mm in sorted(set(mdl)):
                k = (mdl == mm) & (dom == dd) & (mod == "vision")
                line += f"{np.corrcoef(d[k], m[k])[0,1]:>+12.3f}" if k.sum() >= MIN_N else f"{'-':>12}"
            print(line)

    # ---------- (2) sensitivity: A1 vs margin-conditional F_delta ----------
    cells = sorted(set(zip(mdl, dom, mod)))
    obs, pa1, pcond = [], [], []
    for (mm, dd, mo) in cells:
        k = (mdl == mm) & (dom == dd) & (mod == mo)
        if k.sum() < MIN_N or (mm, mo) not in strat:
            continue
        mk, dk, q = strat[(mm, mo)]
        obs.append(fl[k].mean())
        # A1: one F_delta for the whole backbone x modality
        s = np.sort(dk)
        pa1.append(np.mean(np.searchsorted(s, -m[k], side="right") / len(s)))
        # margin-conditional: each item drawn from its OWN margin-quintile's F_delta
        # (assign this cell's items to strata by the backbone-level quantile cutpoints)
        cuts = [mk[q == j].max() for j in range(NQ - 1)]
        qi = np.searchsorted(np.array(cuts), m[k], side="right")
        pr = []
        for j in range(NQ):
            sel = qi == j
            if sel.sum():
                sj = np.sort(dk[q == j])
                pr.append(np.searchsorted(sj, -m[k][sel], side="right") / len(sj))
        pcond.append(float(np.concatenate(pr).mean()) if pr else np.nan)
    obs, pa1, pcond = map(np.array, (obs, pa1, pcond))

    def sm(p):
        good = ~np.isnan(p)
        return (np.corrcoef(p[good], obs[good])[0, 1], np.abs(p[good] - obs[good]).mean())

    ra, ma = sm(pa1); rc, mc = sm(pcond)
    print(f"\n(2) sensitivity of the margin model to A1 ({len(obs)} cells with n>={MIN_N})")
    print(f"    {'variant':38}{'corr(pred,obs)':>16}{'MAE':>9}")
    print(f"    {'A1 (margin-independent F_delta)':38}{ra:>+16.3f}{ma:>9.4f}")
    print(f"    {'margin-quantile-conditional F_delta':38}{rc:>+16.3f}{mc:>9.4f}")
    print(f"    -> conclusion: relaxing A1 moves corr by {abs(rc-ra):.3f} and MAE by {abs(mc-ma):.4f}")
    res.setdefault("sensitivity", []).append(dict(pool=pool, n_cells=int(len(obs)),
                                                  corr_a1=float(ra), mae_a1=float(ma),
                                                  corr_cond=float(rc), mae_cond=float(mc)))
    return res


def main():
    res = {}
    report(load_pool(DM_SUB, "dm"), "dm", res)
    report(load_pool(ASM_SUB, "assembled"), "assembled", res)
    out = DR / "diagnostics/perturbation_invariance.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
