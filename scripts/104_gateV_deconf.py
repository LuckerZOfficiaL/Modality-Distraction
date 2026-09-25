r"""Is the vision-grounding gate reading modality reliance, or question phrasing?

scripts/103 found that a probe predicting VISION-GROUNDING makes the h_V patch look deployable
(AUROC ~1.0 on \dsname). The worry is that the V/T distinction in \dsname{} is partly legible from
question surface form, in which case the gate is a phrasing detector and will not transfer.

We test on the matched-question de-confounded variant (scripts/25--26), which was built to drive a
bag-of-words classifier on the question to chance while preserving the grounding labels. If the
gate's accuracy survives there, it is reading something beyond phrasing; if it collapses, the
\dsname{} number is a surface artifact.

Reported per layer, cross-fitted, alongside:
  bow      bag-of-words logistic on the question text (the surface baseline)
  h_VT     probe on the fused residual (the gate as deployed)
  h_V      probe on the image-only residual (no caption present at all)

Offline, CPU only.

    python scripts/104_gateV_deconf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
FOLDS, SEED = 5, 0
LAYERS = [13, 20, 24, 28, 31]


def crossfit_auroc(X, y, sparse=False):
    p = np.zeros(len(y), dtype=float)
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
        if sparse:
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=2000, C=0.05).fit((X[tr] - mu) / sd, y[tr])
            p[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    pos, neg = p[y == 1], p[y == 0]
    au = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    acc = float(((p > 0.5).astype(int) == y).mean())
    return au, acc


def questions_for(ids):
    """candidate_id -> question text, over both the natural and de-confounded pools."""
    q = {}
    for f in [DR / f"dm/multidomain_v1/splits/{s}.jsonl" for s in ("train", "val", "test")] + \
             list((DR / "dm/multidomain_v1/tcounterparts_v2").rglob("*.jsonl")):
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if "question" in r:
                    q.setdefault(r["candidate_id"], r["question"])
    return [q.get(i, "") for i in ids]


def run(name, store):
    st = DR / "activations" / "qwen" / store
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    y = np.array([r["label"] == "vision" for r in idx]).astype(int)
    acts = np.load(st / "activations.npy", mmap_mode="r")
    ids = [r["candidate_id"] for r in idx]
    print(f"\n===== {name}  n={len(y)}  (V {y.sum()} / T {(1-y).sum()})")

    qs = questions_for(ids)
    have = sum(bool(x) for x in qs)
    bow = None
    if have > 0.5 * len(qs):
        Xb = CountVectorizer(min_df=2, binary=True).fit_transform(qs)
        au, acc = crossfit_auroc(Xb, y, sparse=True)
        bow = dict(auroc=float(au), acc=float(acc), n=len(qs))
        print(f"  {'bag-of-words on the question':34} AUROC {au:.3f}   acc {acc:.3f}   "
              f"({have}/{len(qs)} questions found)")
    else:
        print(f"  bag-of-words: questions unavailable for this store ({have}/{len(qs)})")

    out = {"bow": bow}    # persisted so the paper's BoW numbers are traceable, not log-only
    for L in LAYERS:
        if L >= acts.shape[2]:
            continue
        au_vt, acc_vt = crossfit_auroc(np.asarray(acts[:, 0, L], dtype=np.float32), y)
        au_v, acc_v = crossfit_auroc(np.asarray(acts[:, 1, L], dtype=np.float32), y)
        out[L] = dict(h_vt_auroc=au_vt, h_vt_acc=acc_vt, h_v_auroc=au_v, h_v_acc=acc_v)
        print(f"  L{L:<3} h_VT AUROC {au_vt:.3f} acc {acc_vt:.3f}   |   "
              f"h_V AUROC {au_v:.3f} acc {acc_v:.3f}")
    return out


def main():
    res = {"natural": run("natural \\dsname{} (Qwen2.5-VL-3B)", "dm_multidomain_v1_cf"),
           "deconf": run("de-confounded matched-question variant", "dm_matched_deconf_v2_all")}
    (DR / "diagnostics/gateV_deconf.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {DR / 'diagnostics/gateV_deconf.json'}")


if __name__ == "__main__":
    main()
