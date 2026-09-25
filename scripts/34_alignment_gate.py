"""Step 34: alignment-based selectivity gate — does geometry separate FIX from BREAK?

scripts/33 showed confidence (base_margin) and SAE-atom activations barely separate
"steering will fix" from "steering will break" (AUC 0.58 / 0.66). The hypothesis here
(user): the right selectivity signal is GEOMETRIC, not confidence — each input has a
different steerability, readable from alignment.

Same decisive fix/break set as scripts/33 (de-confounded D_M V-items, VT->V-condition
flip; y=1 FIX, 0 BREAK). Candidate per-item gate signals, per layer:
  cos(h_V, h_VT)              user proposal (a): is image-only aligned with joint?
  ||h_V - h_VT|| = ||delta||  how much the caption moved the state
  cos(h_VT, v_global)         user proposal (b): align input with the global causal dir
  cos(delta_i, v_global)      is the per-item counterfactual aligned with the global dir?
  proj(h_VT, v_global)        signed projection
where v_global = unit(mean_{V items}[h_V - h_VT]) (global causal steering direction).

Report single-feature AUC for each, vs base_margin baseline, and a combined geometric
model. AUC notably > 0.66 => a real selectivity signal => per-item selective steering viable.

    python scripts/34_alignment_gate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")


def auc1(x, y):
    try:
        a = roc_auc_score(y, x); return max(a, 1 - a)   # |direction|
    except ValueError:
        return float("nan")


def auc_multi(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", default="data/activations/qwen/dm_matched_deconf_v2_all")
    ap.add_argument("--layers", default="13,20,28,31")
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy")          # (N,3,n_layers,d) [VT,V,T]
    idx = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    ll = np.load(act_dir / "letter_logits.npy")
    ci = np.array([r["correct_index"] for r in idx])
    is_V = np.array([r["label"] == "vision" for r in idx])
    vt_ok = ll[:, 0].argmax(1) == ci
    v_ok = ll[:, 1].argmax(1) == ci
    p = torch.softmax(torch.from_numpy(ll[:, 0].astype(np.float32)), -1).numpy()
    ps = np.sort(p, 1); base_margin = ps[:, -1] - ps[:, -2]

    decisive = is_V & (vt_ok != v_ok)
    y = v_ok[decisive].astype(int)
    print(f"V items={int(is_V.sum())}  decisive={int(decisive.sum())}  FIX={int(y.sum())}  BREAK={int((1-y).sum())}")
    print(f"\nbaseline AUC(base_margin) = {auc1(base_margin[decisive], y):.3f}\n")
    print(f"{'L':>3} {'cos(hV,hVT)':>12} {'||delta||':>10} {'cos(hVT,vg)':>12} "
          f"{'cos(d,vg)':>10} {'proj(hVT,vg)':>13} {'geom-multi':>11}")

    def cos(a, b):
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

    for L in [int(x) for x in args.layers.split(",")]:
        hVT = acts[:, 0, L].astype(np.float32)
        hV = acts[:, 1, L].astype(np.float32)
        delta = hV - hVT
        v_global = delta[is_V].mean(0); v_global /= (np.linalg.norm(v_global) + 1e-8)

        f_cos_vvt = cos(hV, hVT)[decisive]
        f_dnorm = np.linalg.norm(delta, axis=1)[decisive]
        f_cos_vtg = (hVT @ v_global)[decisive] / (np.linalg.norm(hVT, axis=1)[decisive] + 1e-8)
        f_cos_dg = (delta @ v_global)[decisive] / (np.linalg.norm(delta, axis=1)[decisive] + 1e-8)
        f_proj = (hVT @ v_global)[decisive]
        feats = np.column_stack([f_cos_vvt, f_dnorm, f_cos_vtg, f_cos_dg, f_proj])
        print(f"{L:>3} {auc1(f_cos_vvt,y):>12.3f} {auc1(f_dnorm,y):>10.3f} {auc1(f_cos_vtg,y):>12.3f} "
              f"{auc1(f_cos_dg,y):>10.3f} {auc1(f_proj,y):>13.3f} {auc_multi(feats,y):>11.3f}")


if __name__ == "__main__":
    main()
