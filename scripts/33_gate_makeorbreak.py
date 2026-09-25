"""Step 33: C' make-or-break — do SAE features separate steering FIX from BREAK
better than confidence?

The deployable label-free correction gate failed because the confidence signal
(base_margin) is SYMMETRIC between a fixable-wrong item and a breakable-right item.
C' bets that SAE feature activations carry an ASYMMETRIC "text-prior mode" signal
that confidence doesn't. We test it on the de-confounded store, using the
VT->V-condition flip as the cleanest maximal image-steer:

  on V-grounded items:
    baseline correct = argmax(letter_logits[VT]) == correct
    steered  correct = argmax(letter_logits[V])  == correct   (image-only = max image-steer)
    decisive = baseline_correct != steered_correct
    label y  = steered_correct  (1 = FIX, 0 = BREAK) on the decisive subset

Compare 5-fold CV AUC of (a) base_margin [confidence baseline], (b) de-confounded
SAE atom activations z(h_VT) [the bet], (c) both, for predicting FIX vs BREAK.
SAE >> margin  => C' alive.

    python scripts/33_gate_makeorbreak.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

from moground.sae import TopKSAE, SAEConfig


def _atoms(feat_dir: Path, L: int, topn: int):
    d = json.loads((feat_dir / f"layer{L}" / "top_features.json").read_text())
    s = []
    for key in ("top_vision_causal", "top_vision_disc"):
        s += [int(k) for k in d.get(key, [])][:topn]
    s += [int(x["k"]) for x in d.get("top_text", [])][:topn]
    return sorted(set(s))


def _auc(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--act-dir", default="data/activations/qwen/dm_matched_deconf_v2_all")
    ap.add_argument("--feat-dir", default="data/features/qwen/diversified_v1_deconf")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layers", default="13,20,28,31")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy")
    idx = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    ll = np.load(act_dir / "letter_logits.npy")     # (N,3,4) [VT,V,T]
    ci = np.array([r["correct_index"] for r in idx])
    is_V = np.array([r["label"] == "vision" for r in idx])

    vt_ok = (ll[:, 0].argmax(1) == ci)
    v_ok = (ll[:, 1].argmax(1) == ci)
    # base_margin = top1-top2 softmax prob on the VT letter logits
    p = torch.softmax(torch.from_numpy(ll[:, 0].astype(np.float32)), -1).numpy()
    ps = np.sort(p, 1)
    base_margin = ps[:, -1] - ps[:, -2]

    decisive = is_V & (vt_ok != v_ok)
    y = v_ok[decisive].astype(int)    # 1 = FIX (V-steer correct, VT wrong), 0 = BREAK
    print(f"V items={int(is_V.sum())}  decisive(VT!=V)={int(decisive.sum())}  "
          f"FIX={int(y.sum())}  BREAK={int((1-y).sum())}")
    if y.sum() < 10 or (1 - y).sum() < 10:
        print("too few in a class — abort"); return

    auc_margin = _auc(base_margin[decisive].reshape(-1, 1), y)
    print(f"\nbaseline gate  AUC(base_margin)               = {auc_margin:.3f}")
    print(f"{'L':>3} {'AUC(SAE atoms)':>15} {'AUC(SAE+margin)':>16} {'n_atoms':>8}")
    for L in [int(x) for x in args.layers.split(",")]:
        ckpt = data_root / "sae" / "qwen" / args.variant / f"layer{L}" / "sae.pt"
        if not ckpt.exists():
            print(f"{L:>3}  SAE missing — skip"); continue
        ck = torch.load(ckpt, map_location=args.device, weights_only=False)
        sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
        sae.load_state_dict(ck["state_dict"]); sae.to(args.device).to(torch.float32).eval()
        atoms = _atoms(Path(args.feat_dir), L, args.top_n)
        X = torch.from_numpy(acts[decisive, 0, L].astype(np.float32)).to(args.device)
        with torch.no_grad():
            z = sae(X)["z"].cpu().numpy()[:, atoms]
        auc_sae = _auc(z, y)
        auc_both = _auc(np.column_stack([z, base_margin[decisive]]), y)
        print(f"{L:>3} {auc_sae:>15.3f} {auc_both:>16.3f} {len(atoms):>8}")


if __name__ == "__main__":
    main()
