"""Step 24: de-confounded V/T probe on the matched T-counterpart set.

Tests whether h_VT carries modality-USE signal beyond question surface form, using
the matched T-counterparts (same question template as the source V item, answer
injected into the caption). Reports, under ONE protocol (standardized LR, 5-fold CV):

  1. BoW floors  : question+options, caption, caption+question (surface-text predictability).
  2. h_VT per layer (V/T), NEW (deconf matched) vs OLD (natural D_M).
  3. Condition decomposition: VT (img+cap+q) vs V (img+q, no caption) vs T (cap+q, no image),
     showing the discriminative axis lives in the caption channel, not the image.

Conclusion (June 19): de-confounding the question (BoW 0.96 -> 0.63) only exposes the
caption-content confound (BoW 0.70); h_VT ~ h_T >> h_V, i.e. the axis is an input-
difference (caption-asserts-answer) detector, not routing. See project_log
Section "De-Confounding T Questions".

    python scripts/24_deconf_probe.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
import warnings; warnings.filterwarnings("ignore")


def _clf():
    return make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=200, C=0.5, class_weight="balanced"))


def cv_acc(X, y):
    return (cross_val_predict(_clf(), X, y, cv=5, n_jobs=5) == y).mean()


def bow_acc(texts, y):
    Xb = CountVectorizer(lowercase=True).fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return (cross_val_predict(clf, Xb, y, cv=5) == y).mean()


def _text(r):
    return (r.get("caption_for_filter", ""), r["question"] + " " + " ".join(map(str, r["options"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--new-store", default="data/activations/qwen/dm_matched_deconf")
    ap.add_argument("--old-store", default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--matched-rows", default="data/dm/multidomain_v1/tcounterparts/official_matched_set.jsonl")
    ap.add_argument("--layers", default="0,5,10,13,15,18,20,23,25,28,31,33,35")
    args = ap.parse_args()

    LAYERS = [int(x) for x in args.layers.split(",")]
    dm = Path(yaml.safe_load(Path(args.config).read_text())["paths"]["dm_dir"])

    # ---- NEW: de-confounded matched set ----
    nst = Path(args.new_store)
    nacts = np.load(nst / "activations.npy", mmap_mode="r")            # (N,3,36,d) [VT,V,T]
    nidx = [json.loads(l) for l in (nst / "index.jsonl").read_text().splitlines() if l.strip()]
    ny = np.array([0 if r["label"] == "vision" else 1 for r in nidx])
    nrows = {r["candidate_id"]: r for r in
             (json.loads(l) for l in Path(args.matched_rows).read_text().splitlines() if l.strip())}
    ncap = [nrows[r["candidate_id"]].get("caption_for_filter", "") for r in nidx]
    nq = [_text(nrows[r["candidate_id"]])[1] for r in nidx]

    print(f"NEW deconf matched: n={len(ny)} V={(ny==0).sum()} T={(ny==1).sum()} chance={max(ny.mean(),1-ny.mean()):.3f}")
    print("\n=== BoW surface floors (NEW) ===")
    print(f"  question+options : {bow_acc(nq, ny):.3f}")
    print(f"  caption          : {bow_acc(ncap, ny):.3f}")
    print(f"  caption+question : {bow_acc([c+' '+q for c, q in zip(ncap, nq)], ny):.3f}")

    # ---- OLD: natural D_M ----
    ost = Path(args.old_store)
    oacts = np.load(ost / "activations.npy", mmap_mode="r")
    oidx = [json.loads(l) for l in (ost / "index.jsonl").read_text().splitlines() if l.strip()]
    omask = np.array([r["label"] in ("vision", "text") for r in oidx])
    oy = np.array([0 if r["label"] == "vision" else 1 for r in oidx])[omask]
    txt = {}
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); txt[r["candidate_id"]] = r
    ocids = [oidx[i]["candidate_id"] for i in range(len(oidx)) if omask[i]]
    o_capq = [(txt[c].get("caption_for_filter", "") + " " + _text(txt[c])[1]) if c in txt else "" for c in ocids]
    old_bow = bow_acc(o_capq, oy)

    print(f"\nOLD natural D_M: n={omask.sum()} V={(oy==0).sum()} T={(oy==1).sum()} chance={max(oy.mean(),1-oy.mean()):.3f}")

    # ---- h_VT per layer, OLD vs NEW ----
    print("\n=== h_VT V/T probe per layer (OLD vs NEW) ===")
    print(f"BoW caption+question         OLD={old_bow:.3f}  NEW={bow_acc([c+' '+q for c,q in zip(ncap,nq)],ny):.3f}")
    print(f"{'L':>3} {'OLD':>8} {'NEW':>8} {'Δ':>8}")
    for L in LAYERS:
        ao = cv_acc(np.asarray(oacts[omask, 0, L, :], np.float32), oy)
        an = cv_acc(np.asarray(nacts[:, 0, L, :], np.float32), ny)
        print(f"{L:>3} {ao:>8.3f} {an:>8.3f} {an-ao:>+8.3f}")

    # ---- condition decomposition (NEW) ----
    print("\n=== condition decomposition (NEW matched; image shared within each V/T pair) ===")
    print(f"{'L':>3} {'VT(img+cap+q)':>14} {'V(img+q,noCap)':>15} {'T(cap+q,noImg)':>15}")
    for L in LAYERS:
        aVT = cv_acc(np.asarray(nacts[:, 0, L, :], np.float32), ny)
        aV  = cv_acc(np.asarray(nacts[:, 1, L, :], np.float32), ny)
        aT  = cv_acc(np.asarray(nacts[:, 2, L, :], np.float32), ny)
        print(f"{L:>3} {aVT:>14.3f} {aV:>15.3f} {aT:>15.3f}")


if __name__ == "__main__":
    main()
