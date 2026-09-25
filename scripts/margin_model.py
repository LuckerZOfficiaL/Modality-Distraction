"""#6 Margin model of distraction (mechanism behind the grounding-strength law).

Per V-solvable vision item: V-only answer margin m = logit_gold - max_other (>0).
Adding the caption shifts it: margin_VT = m + delta, and the item flips iff margin_VT < 0,
i.e. delta < -m. If the caption perturbation delta is ~margin-independent, then
  P(flip | m) = P(delta < -m) = F_delta(-m),
and a cell's distraction rate is E_m[F_delta(-m)] over the cell's margins -- so grounding
strength (which governs the margin distribution) determines distraction with NO free
per-cell parameter. We test: (i) is delta ~independent of m? (ii) does a single global
(or per-backbone) F_delta + each cell's margins REPRODUCE the observed cell distraction?

Offline. Establishes the law as a margin mechanism, not a bare correlation.

    python scripts/margin_model.py                              # D_M, all models with a cf-store
    python scripts/margin_model.py --pool assembled --cf-subdir merged_aokvqa_racehigh_cf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
DOMS = ["dci", "vistext", "semart", "roco"]


def discover_models(cf_subdir):  # any backbone with this cf-store present
    return sorted(p.name for p in (DR / "activations").iterdir()
                  if (p / cf_subdir / "letter_logits.npy").exists())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="dm", choices=["dm", "assembled"])
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_cf")
    ap.add_argument("--models", default=None, help="comma list of keys; default = all with a cf-store")
    ap.add_argument("--exclude-file", default=None,
                    help="json with a 'violations' list of candidate_ids to drop (certified pool)")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else discover_models(args.cf_subdir)
    CF = {m: args.cf_subdir for m in models}
    excl = set(json.load(open(args.exclude_file))["violations"]) if args.exclude_file else set()
    print(f"pool={args.pool} cf-subdir={args.cf_subdir} models={models} excluded={len(excl)}")

    dmdir = Path(yaml.safe_load((ROOT / "configs/pilot_multidomain_v1.yaml").read_text())["paths"]["dm_dir"])
    src = {}
    for sp in ("train", "val", "test"):
        p = dmdir / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); src[r["candidate_id"]] = r.get("source")

    def domain_of(r):  # D_M: per-source-domain cells; assembled: one cell per model
        if args.pool == "assembled":
            return "aokvqa"
        return r.get("source") or src.get(r["candidate_id"])

    items = []  # (model, domain, m, delta, flip)
    for m, sub in CF.items():
        ll = np.load(DR / "activations" / m / sub / "letter_logits.npy")
        idx = [json.loads(l) for l in (DR / "activations" / m / sub / "index.jsonl").read_text().splitlines() if l.strip()]
        ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
        v_ok = ll[:, 1].argmax(1) == ci
        for i, r in enumerate(idx):
            dom = domain_of(r)
            keep = ((args.pool == "assembled") or (dom in DOMS)) and r["candidate_id"] not in excl
            if lab[i] == "vision" and v_ok[i] and keep:
                lv = ll[i, 1].astype(np.float64); lvt = ll[i, 0].astype(np.float64); g = ci[i]
                mv = lv[g] - np.delete(lv, g).max()
                mvt = lvt[g] - np.delete(lvt, g).max()
                items.append((m, dom, float(mv), float(mvt - mv), int(mvt < 0)))

    model = np.array([x[0] for x in items]); dom = np.array([x[1] for x in items])
    mv = np.array([x[2] for x in items]); delta = np.array([x[3] for x in items]); flip = np.array([x[4] for x in items])
    print(f"V-solvable vision items: {len(items)}  flip rate={flip.mean():.3f}")

    # (i) is delta ~ independent of the V-only margin m?
    print(f"\n(i) caption perturbation delta vs margin m:  corr(delta, m) = {np.corrcoef(delta, mv)[0,1]:+.3f}  "
          f"(near 0 => margin-independent perturbation)")
    print(f"    delta: mean={delta.mean():+.2f} sd={delta.std():.2f}  (negative mean => caption tends to erode the vision margin)")

    # (ii) predict each cell's distraction from a GLOBAL F_delta + the cell's margins (no per-cell param)
    def Fdelta_global(x):  # P(delta < x) empirical
        return np.searchsorted(np.sort(delta), x, side="right") / len(delta)
    # per-backbone F_delta too (delta scale may differ by model)
    sorted_delta_by = {mm: np.sort(delta[model == mm]) for mm in CF}
    def Fdelta_bk(mm, x):
        s = sorted_delta_by[mm]; return np.searchsorted(s, x, side="right") / len(s)

    cells = sorted(set(zip(model, dom)))
    obs, pred_g, pred_b = [], [], []
    for (mm, dd) in cells:
        msk = (model == mm) & (dom == dd)
        obs.append(flip[msk].mean())
        pred_g.append(np.mean([Fdelta_global(-x) for x in mv[msk]]))
        pred_b.append(np.mean([Fdelta_bk(mm, -x) for x in mv[msk]]))
    obs, pred_g, pred_b = map(np.array, (obs, pred_g, pred_b))
    def r2(p):
        return 1 - ((p - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    print(f"\n(ii) margin model reproduces cell distraction ({len(cells)} vision cells), NO per-cell param:")
    print(f"     global  F_delta: R^2={r2(pred_g):+.3f}  corr={np.corrcoef(pred_g,obs)[0,1]:+.3f}  MAE={np.abs(pred_g-obs).mean():.3f}")
    print(f"     per-bkbone F_delta: R^2={r2(pred_b):+.3f}  corr={np.corrcoef(pred_b,obs)[0,1]:+.3f}  MAE={np.abs(pred_b-obs).mean():.3f}")
    print(f"\n     {'cell':22}{'obs':>7}{'pred(bk)':>10}")
    for (mm, dd), o, pb in sorted(zip(cells, obs, pred_b), key=lambda t: -t[1]):
        print(f"     {mm+'/'+dd:22}{o:>7.3f}{pb:>10.3f}")


if __name__ == "__main__":
    main()
