"""#1 Is the grounding-strength law substantive or near-tautological?

Part A (cell level, 30 cells from scripts/54): does distraction ~ grounding generalize
OUT-OF-SAMPLE? leave-one-cell-out and leave-one-MODEL-out prediction (held-out R^2/MAE).

Part B (item level, from cf-stores): the difficulty control. Distraction of a V-solvable
item is plausibly just "low V-only margin flips." We test whether the item's own V-only
margin predicts flipping (the MECHANISM), and whether CELL grounding strength adds
predictive power BEYOND the item's margin (the anti-tautology test). If grounding only
matters via margin, the law reduces to a margin effect; if grounding survives the margin
control, it is substantive.

Offline. cf-stores discovered under data/activations/*/{cf-subdir} (uniform name dm_multidomain_v1_cf).

    python scripts/58_grounding_law_robustness.py                                          # D_M
    python scripts/58_grounding_law_robustness.py --pool assembled --cf-subdir merged_aokvqa_racehigh_cf
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


def discover_models(cf_subdir):
    return sorted(p.name for p in (DR / "activations").iterdir()
                  if (p / cf_subdir / "letter_logits.npy").exists())


def partial_corr(y, x, z):
    """corr(y, x | z): correlation of residuals after regressing out z (1-D)."""
    def resid(a):
        b = np.polyfit(z, a, 1); return a - (b[0] * z + b[1])
    ry, rx = resid(y), resid(x)
    return float(np.corrcoef(ry, rx)[0, 1])


def part_A():
    pts = json.loads((DR / "diagnostics/grounding_strength.json").read_text())
    g = np.array([p["ground"] for p in pts]); d = np.array([p["distr"] for p in pts])
    model = np.array([p["model"] for p in pts])
    print(f"=== Part A: out-of-sample generalization of distraction~grounding ({len(pts)} cells) ===")
    print(f"  in-sample Pearson r = {np.corrcoef(g, d)[0,1]:+.3f}")
    # leave-one-cell-out
    preds = np.zeros(len(pts))
    for i in range(len(pts)):
        m = np.arange(len(pts)) != i
        b = np.polyfit(g[m], d[m], 1); preds[i] = b[0] * g[i] + b[1]
    sse = ((preds - d) ** 2).sum(); sst = ((d - d.mean()) ** 2).sum()
    print(f"  LOCO: held-out R^2 = {1 - sse/sst:+.3f}  MAE = {np.abs(preds-d).mean():.3f}  corr(pred,actual)={np.corrcoef(preds,d)[0,1]:+.3f}")
    # leave-one-MODEL-out
    for M in sorted(set(model)):
        tr = model != M; te = model == M
        b = np.polyfit(g[tr], d[tr], 1); p = b[0] * g[te] + b[1]
        mae = np.abs(p - d[te]).mean()
        print(f"  LOMO predict {M:14}: MAE={mae:.3f}  (held-out cells n={int(te.sum())})")


def load_items(CF, pool):
    """per V-solvable vision item: margin_v, distracted, model, domain."""
    splits = {}
    dmdir = Path(yaml.safe_load((ROOT / "configs/pilot_multidomain_v1.yaml").read_text())["paths"]["dm_dir"])
    for sp in ("train", "val", "test"):
        p = dmdir / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); splits[r["candidate_id"]] = r.get("source")
    rows = []
    for m, sub in CF.items():
        st = DR / "activations" / m / sub
        ll = np.load(st / "letter_logits.npy")
        idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
        ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
        v_ok = ll[:, 1].argmax(1) == ci; vt_wrong = ll[:, 0].argmax(1) != ci
        for i, r in enumerate(idx):
            dom = "aokvqa" if pool == "assembled" else (r.get("source") or splits.get(r["candidate_id"]))
            if lab[i] == "vision" and v_ok[i] and (pool == "assembled" or dom in DOMS):
                lv = ll[i, 1].astype(np.float64); gold = ci[i]
                others = np.delete(lv, gold)
                rows.append({"model": m, "domain": dom,
                             "margin": float(lv[gold] - others.max()),
                             "distracted": int(vt_wrong[i])})
    return rows


def part_B(CF, pool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    rows = load_items(CF, pool)
    margin = np.array([r["margin"] for r in rows]); dist = np.array([r["distracted"] for r in rows])
    model = np.array([r["model"] for r in rows]); dom = np.array([r["domain"] for r in rows])
    print(f"\n=== Part B: difficulty (margin) control ({len(rows)} V-solvable vision items) ===")
    # B1: does item V-only margin predict flipping? (the mechanism)
    for m in sorted(set(model)) + ["ALL"]:
        msk = np.ones(len(rows), bool) if m == "ALL" else (model == m)
        if dist[msk].sum() < 10: continue
        X = StandardScaler().fit_transform(margin[msk].reshape(-1, 1))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        auc = cross_val_score(clf, X, dist[msk], cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
        clf.fit(X, dist[msk])
        print(f"  margin->flip {m:14}: AUROC={auc:.3f}  coef={clf.coef_[0,0]:+.2f} (neg = smaller margin flips more)")
    # B2: cell-level partial correlation -- does grounding add beyond mean margin?
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["domain"]), []).append(r)
    cg, cmean_margin, cdistr = [], [], []
    for k, rs in cells.items():
        cg.append(1.0)  # placeholder, grounding computed below
        cmean_margin.append(np.mean([x["margin"] for x in rs]))
        cdistr.append(np.mean([x["distracted"] for x in rs]))
    # grounding strength per cell = V-only accuracy; recompute from full vision pool (not just V-solvable)
    gs = {(p["model"], p["domain"]): p["ground"]
          for p in json.loads((DR / "diagnostics/grounding_strength.json").read_text())
          if p["modality"] == "vision" and p["dataset"] == ("assembled" if pool == "assembled" else "dm")}
    keys = list(cells.keys())
    g = np.array([gs[k] for k in keys]); mm = np.array(cmean_margin); dd = np.array(cdistr)
    print(f"\n  cell-level (n={len(keys)} vision cells):")
    print(f"    corr(distraction, grounding)           = {np.corrcoef(dd,g)[0,1]:+.3f}")
    print(f"    corr(distraction, mean-margin)         = {np.corrcoef(dd,mm)[0,1]:+.3f}")
    print(f"    corr(grounding, mean-margin)           = {np.corrcoef(g,mm)[0,1]:+.3f}")
    print(f"    PARTIAL corr(distraction, grounding | mean-margin)   = {partial_corr(dd,g,mm):+.3f}")
    print(f"    PARTIAL corr(distraction, mean-margin | grounding)   = {partial_corr(dd,mm,g):+.3f}")
    # B3: pooled item logistic distracted ~ margin + grounding_cell (do both survive?)
    gcell = np.array([gs[(r["model"], r["domain"])] for r in rows])
    X = StandardScaler().fit_transform(np.column_stack([margin, gcell]))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, dist)
    print(f"\n  pooled item logistic distracted ~ z(margin)+z(grounding_cell): "
          f"coef margin={clf.coef_[0,0]:+.2f}, grounding={clf.coef_[0,1]:+.2f} "
          f"(both large => grounding adds beyond item margin)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="dm", choices=["dm", "assembled"])
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_cf")
    ap.add_argument("--models", default=None, help="comma list of keys; default = all with a cf-store")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else discover_models(args.cf_subdir)
    CF = {m: args.cf_subdir for m in models}
    print(f"pool={args.pool} cf-subdir={args.cf_subdir} models={models}\n")
    part_A()
    part_B(CF, args.pool)
