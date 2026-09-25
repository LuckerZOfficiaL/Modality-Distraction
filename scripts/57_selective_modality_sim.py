"""Direction A feasibility (offline, no GPU): does DETECTION-GATED selective modality
use net-improve accuracy where steering could not?

Policy: a cheap gate (logistic on h_V, the image-only residual -- available pre-caption)
predicts per-item distraction-risk; if flagged, answer from V-only (drop the caption),
else from V+T. We measure net accuracy on the MIXED pool vs always-V+T, against the
oracle-gate ceiling. Gate trained with out-of-fold CV (no leakage). Threshold swept;
deployment would set it by conformal/val calibration (noted, not done here).

Bounded by design: the max recoverable is the v-distraction mass, so absolute gains are
modest on controlled pools and largest where distraction is high (non-natural domains,
weaker backbones, RAG). Reports overall + per-domain.

    python scripts/57_selective_modality_sim.py --model qwen --cf-subdir dm_multidomain_v1_counterfactual_full --layer 28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual_full")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--rows-jsonl", default=None, help="for assembled: dm_all.jsonl (source tags)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    st = dr / "activations" / args.model / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    pred_vt = ll[:, 0].argmax(1); pred_v = ll[:, 1].argmax(1)
    # source tags
    smap = {}
    src_iter = ([json.loads(l) for l in Path(args.rows_jsonl).read_text().splitlines() if l.strip()]
                if args.rows_jsonl else
                [json.loads(l) for sp in ("train", "val", "test")
                 for l in (dm / "splits" / f"{sp}.jsonl").read_text().splitlines()
                 if (dm / "splits" / f"{sp}.jsonl").exists() and l.strip()])
    for r in src_iter:
        smap[r["candidate_id"]] = r.get("source") or ("aokvqa" if r.get("label") == "vision" else "race")
    src = np.array([smap.get(r["candidate_id"], "?") for r in idx])

    # gate target: v-distracted = vision & V-only correct & V+T wrong (where dropping caption helps)
    y = ((lab == "vision") & (pred_v == ci) & (pred_vt != ci)).astype(int)
    X = acts[:, 1, args.layer].astype(np.float32)  # h_V
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]

    base = (pred_vt == ci).mean()
    # oracle gate: route the TRUE v-distracted to V-only
    oracle = np.where(y == 1, pred_v == ci, pred_vt == ci).mean()
    print(f"=== Direction A sim: {args.model} / {args.cf_subdir} / h_V@L{args.layer} ===")
    print(f"n={len(idx)}  v-distracted(gate target)={int(y.sum())}  baseline(always V+T)={base:.4f}  oracle-gate ceiling={oracle:.4f}")
    print(f"{'thresh':>8}{'fire%':>8}{'routed_acc':>12}{'gain':>9}{'txt_broken':>11}")
    best = (base, 0.5, 0.0)
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        gate = p > tau
        routed = np.where(gate, pred_v == ci, pred_vt == ci)
        acc = routed.mean()
        txt_broken = int(((lab == "text") & gate & (pred_vt == ci) & (pred_v != ci)).sum())
        print(f"{tau:>8.2f}{gate.mean()*100:>7.1f}%{acc:>12.4f}{acc-base:>+9.4f}{txt_broken:>11}")
        if acc > best[0]:
            best = (acc, tau, acc - base)
    print(f"best: routed_acc={best[0]:.4f} at tau={best[1]:.2f} (gain {best[2]:+.4f}); "
          f"oracle headroom={oracle-base:+.4f}")
    sp = Path(cfg["paths"]["data_root"]) / "diagnostics" / "selective_modality_sim.json"
    d = json.loads(sp.read_text()) if sp.exists() else {}
    d[args.model] = {"baseline": round(float(base), 4), "oracle": round(float(oracle), 4),
                     "best_gain": round(float(best[2]), 4), "headroom": round(float(oracle - base), 4)}
    sp.write_text(json.dumps(d, indent=2))

    # per-domain net gain at best tau
    gate = p > best[1]; routed = np.where(gate, pred_v == ci, pred_vt == ci); basev = (pred_vt == ci)
    print(f"\nper-domain net gain @tau={best[1]:.2f}:")
    for d in sorted(set(src)):
        m = src == d
        if m.sum() >= 20:
            print(f"  {d:9} n={int(m.sum()):>4}  base {basev[m].mean():.3f} -> routed {routed[m].mean():.3f} ({routed[m].mean()-basev[m].mean():+.3f})")


if __name__ == "__main__":
    main()
