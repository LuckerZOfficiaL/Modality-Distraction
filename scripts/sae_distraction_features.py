"""X3: do the released SAE features expose DISTRACTION SUSCEPTIBILITY (non-modality)?

The SAE-utility demo for the resource paper: use the diversified VL-SAE for something
other than the (scooped) modality axis. Among vision items the model solves V-only,
split distracted (V+T wrong) vs robust (V+T correct), encode h_VT through the SAE, and
ask: (a) are there individual features that fire differentially on distracted items
(per-feature AUROC)? (b) does a probe on the SAE code separate distracted from robust
above a label-permutation null? Top features are saved for later auto-interp.

Offline: cf-store h_VT + diversified SAE checkpoints. n_distracted is small (~89) — we
report AUROC + a permutation null and are honest about power.

    python scripts/sae_distraction_features.py --layers 20,28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from moground.sae import TopKSAE, SAEConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layers", default="20,28")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--n-perms", type=int, default=5)
    ap.add_argument("--out", default="data/diagnostics/sae_distraction")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"])
    st = dr / "activations" / args.model / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")  # (N,3,Lyr,d) [VT,V,T]
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    v_ok = ll[:, 1].argmax(1) == ci
    vt_ok = ll[:, 0].argmax(1) == ci
    sel = (lab == "vision") & v_ok                    # vision, image-alone-solvable
    y = (~vt_ok[sel]).astype(int)                     # 1 = distracted (V+T wrong)
    sel_ix = np.where(sel)[0]
    print(f"[{args.model}] vision V-solvable n={len(sel_ix)}  distracted={int(y.sum())}  robust={int((y==0).sum())}")
    out = Path(args.out) / (args.model if args.model != "qwen" else "")  # qwen keeps the legacy base dir
    out.mkdir(parents=True, exist_ok=True)

    for L in [int(x) for x in args.layers.split(",")]:
        ck = torch.load(dr / "sae" / args.model / args.variant / f"layer{L}" / "sae.pt",
                        map_location=args.device, weights_only=False)
        sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
        sae.load_state_dict(ck["state_dict"]); sae.to(args.device).eval()
        with torch.no_grad():
            x = torch.tensor(acts[sel_ix, 0, L].astype(np.float32), device=args.device)
            z = sae(x)["z"].cpu().numpy()             # (n, d_sae)
        alive = np.where((z > 0).any(0))[0]
        za = z[:, alive]
        fire = (za > 0)
        fr_d = fire[y == 1].mean(0); fr_r = fire[y == 0].mean(0)   # firing rate per alive feat
        # per-feature AUROC (activation magnitude vs distracted label)
        auc = np.array([roc_auc_score(y, za[:, j]) if za[:, j].std() > 0 else 0.5
                        for j in range(za.shape[1])])
        order = np.argsort(-np.abs(auc - 0.5))
        print(f"\n=== L{L}: {len(alive)} alive features; top {args.top} distraction discriminators ===")
        print(f"{'feat':>8}{'AUROC':>8}{'fr_distr':>10}{'fr_robust':>11}{'Δfr':>8}")
        top = []
        for j in order[:args.top]:
            print(f"{alive[j]:>8}{auc[j]:>8.3f}{fr_d[j]:>10.3f}{fr_r[j]:>11.3f}{fr_d[j]-fr_r[j]:>+8.3f}")
            top.append({"feature": int(alive[j]), "auroc": float(auc[j]),
                        "fr_distracted": float(fr_d[j]), "fr_robust": float(fr_r[j])})
        # probe: does the SAE code separate distracted vs robust above a permutation null?
        clf = LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced", solver="liblinear")
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        real = cross_val_score(clf, za, y, cv=skf, scoring="roc_auc", n_jobs=-1).mean()
        rng = np.random.default_rng(0)
        null = np.mean([cross_val_score(clf, za, rng.permutation(y), cv=skf,
                                        scoring="roc_auc", n_jobs=-1).mean() for _ in range(args.n_perms)])
        print(f"L{L} probe on SAE code: CV-AUROC real {real:.3f}  vs  perm-null {null:.3f}  (Δ {real-null:+.3f})")
        (out / f"layer{L}_top.json").write_text(json.dumps(
            {"layer": L, "n": len(sel_ix), "n_distracted": int(y.sum()),
             "probe_auroc_real": float(real), "probe_auroc_null": float(null),
             "top_features": top}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
