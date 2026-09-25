"""Covariate controls for the grounding-strength relation.

The relation is a correlation over cells (backbone x domain x modality) between single-modality
grounding strength and distraction rate.  A reviewer can reasonably ask whether it is carried by
something that co-varies with grounding rather than by grounding: cells differ in question length,
in how much context the distractor adds, in option count, in source dataset, and in how strong the
OFF-modality is (a stronger distractor could flip items without the grounded side being weak).

This script does three things:
  (1) covariate BALANCE: how each covariate varies across the grounding range;
  (2) PARTIAL correlation r(distraction, grounding | covariates), by residualising both on the
      covariate set with OLS and correlating the residuals -- reported for nested covariate sets so
      the cost of each control is visible;
  (3) a WITHIN-STRATUM check: the correlation recomputed inside each source dataset and inside each
      modality, where the covariate that a stratum holds fixed cannot be doing the work.

Offline, CPU only.

    python scripts/covariate_controls.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
DOMS = ["dci", "vistext", "semart", "roco"]
DM_SUB, ASM_SUB = "dm_multidomain_v1_cf", "merged_aokvqa_racehigh_cf"
MIN_N = 30


def item_meta():
    """candidate_id -> covariates, for both pools."""
    meta = {}
    dmdir = Path(yaml.safe_load((ROOT / "configs/pilot_multidomain_v1.yaml").read_text())["paths"]["dm_dir"])
    files = [dmdir / "splits" / f"{sp}.jsonl" for sp in ("train", "val", "test")]
    files.append(DR / "dm/merged_aokvqa_racehigh/dm_all.jsonl")
    for f in files:
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            opts = r["options"]
            if isinstance(opts, str):
                opts = eval(opts)                        # assembled store keeps options as a repr
            meta[r["candidate_id"]] = dict(
                qlen=len(str(r["question"]).split()),
                clen=len(str(r.get("caption_for_filter") or "").split()),
                nopt=len(opts),
                optlen=float(np.mean([len(str(o).split()) for o in opts])),
                source=r.get("source") or r.get("source_text_qa") or "aokvqa",
            )
    return meta


def cells(sub, pool, meta):
    """One row per (model, domain, modality) cell with grounding, distraction and covariates."""
    viol = set()
    if pool == "assembled":
        viol = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
    rows = []
    for mdir in sorted((DR / "activations").iterdir()):
        m = mdir.name
        if m in EXCLUDED or not (mdir / sub / "letter_logits.npy").exists():
            continue
        ll = np.load(mdir / sub / "letter_logits.npy").astype(np.float64)
        idx = [json.loads(l) for l in (mdir / sub / "index.jsonl").read_text().splitlines() if l.strip()]
        buckets = {}
        for i, r in enumerate(idx):
            cid = r["candidate_id"]
            if cid in viol or cid not in meta:
                continue
            lab = r["label"]
            dom = meta[cid]["source"] if pool == "dm" else ("aokvqa" if lab == "vision" else "race_high")
            if pool == "dm" and dom not in DOMS:
                continue
            g = int(r["correct_index"])
            gc, oc = (1, 2) if lab == "vision" else (2, 1)   # grounded / off-modality condition
            ok_g = int(ll[i, gc].argmax() == g)
            ok_o = int(ll[i, oc].argmax() == g)
            ok_vt = int(ll[i, 0].argmax() == g)
            buckets.setdefault((dom, lab), []).append((ok_g, ok_o, ok_vt, meta[cid]))
        for (dom, lab), v in buckets.items():
            if len(v) < MIN_N:
                continue
            ok_g = np.array([x[0] for x in v]); ok_o = np.array([x[1] for x in v])
            ok_vt = np.array([x[2] for x in v])
            solv = ok_g == 1
            if solv.sum() < MIN_N:
                continue
            rows.append(dict(model=m, domain=dom, modality=lab, n=len(v), n_solv=int(solv.sum()),
                             grounding=float(ok_g.mean()),
                             distraction=float((ok_vt[solv] == 0).mean()),
                             off_acc=float(ok_o.mean()),
                             qlen=float(np.mean([x[3]["qlen"] for x in v])),
                             clen=float(np.mean([x[3]["clen"] for x in v])),
                             nopt=float(np.mean([x[3]["nopt"] for x in v])),
                             optlen=float(np.mean([x[3]["optlen"] for x in v])),
                             pool=pool))
    return rows


def partial_r(y, x, Z):
    """Pearson r between y and x after residualising both on Z (with intercept)."""
    Z = np.column_stack([np.ones(len(y))] + ([Z] if Z is None else list(Z.T))) if Z is not None \
        else np.ones((len(y), 1))
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def main():
    meta = item_meta()
    rows = cells(DM_SUB, "dm", meta) + cells(ASM_SUB, "assembled", meta)
    print(f"cells: {len(rows)}  (dm {sum(r['pool']=='dm' for r in rows)}, "
          f"assembled {sum(r['pool']=='assembled' for r in rows)})")

    g = np.array([r["grounding"] for r in rows]); d = np.array([r["distraction"] for r in rows])
    cov = {k: np.array([r[k] for r in rows]) for k in ("qlen", "clen", "nopt", "optlen", "off_acc")}
    dom = np.array([r["domain"] for r in rows]); mod = np.array([r["modality"] for r in rows])

    # ---------- (1) covariate balance across the grounding range ----------
    print(f"\n(1) covariate balance: cell means by grounding tercile")
    cut = np.quantile(g, [1 / 3, 2 / 3]); ter = np.searchsorted(cut, g)
    print(f"    {'tercile':16}{'n':>5}{'grounding':>11}{'distr':>9}" +
          "".join(f"{k:>10}" for k in cov))
    for t, nm in enumerate(("low g", "mid g", "high g")):
        k = ter == t
        print(f"    {nm:16}{k.sum():>5}{g[k].mean():>11.3f}{d[k].mean():>9.3f}" +
              "".join(f"{cov[c][k].mean():>10.2f}" for c in cov))

    # ---------- (2) partial correlations, nested covariate sets ----------
    print(f"\n(2) r(distraction, grounding) with nested controls ({len(rows)} cells)")
    src_d = np.column_stack([(dom == s).astype(float) for s in sorted(set(dom))[:-1]])
    sets = [("none", None),
            ("+ question length", np.column_stack([cov["qlen"]])),
            ("+ context length", np.column_stack([cov["qlen"], cov["clen"]])),
            ("+ option count & length", np.column_stack([cov["qlen"], cov["clen"], cov["nopt"], cov["optlen"]])),
            ("+ off-modality accuracy", np.column_stack([cov["qlen"], cov["clen"], cov["nopt"],
                                                         cov["optlen"], cov["off_acc"]])),
            ("+ source-dataset dummies", np.column_stack([cov["qlen"], cov["clen"], cov["nopt"],
                                                          cov["optlen"], cov["off_acc"], src_d]))]
    out = {"nested": []}
    for nm, Z in sets:
        r = partial_r(d, g, Z)
        print(f"    {nm:30}{r:>+9.3f}")
        out["nested"].append(dict(controls=nm, r=r))

    # ---------- (3) within-stratum ----------
    print(f"\n(3) correlation recomputed within strata (control by construction)")
    for nm, sel in ([(f"domain={s}", dom == s) for s in sorted(set(dom))] +
                    [(f"modality={s}", mod == s) for s in sorted(set(mod))] +
                    [("pool=assembled (ceiling-free)", np.array([r["pool"] == "assembled" for r in rows]))]):
        if sel.sum() >= 6 and g[sel].std() > 1e-6:
            r = float(np.corrcoef(d[sel], g[sel])[0, 1])
            print(f"    {nm:30}{sel.sum():>5} cells{r:>+9.3f}")
            out.setdefault("strata", []).append(dict(stratum=nm, n_cells=int(sel.sum()), r=r))

    (DR / "diagnostics/covariate_controls.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {DR / 'diagnostics/covariate_controls.json'}")


if __name__ == "__main__":
    main()
