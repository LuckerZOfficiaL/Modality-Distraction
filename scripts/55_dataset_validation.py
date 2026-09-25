"""#4 Dataset validation: sanity-check the oracle-generated MCQs for the degenerate
failure modes reviewers look for --- positional bias, length bias, and surface/lexical
answerability (can the correct option be picked from text alone?).

  (1) position bias  : distribution of correct_index over A/B/C/D (+ chi-square vs uniform)
  (2) length bias    : is the correct option systematically the longest / longer?
  (3) lexical leakage: per-option logistic on SURFACE features only (length, question
                       word-overlap, is-numeric -- NO position) predicting is-correct;
                       AUROC near 0.5 == no text tell.

Offline: D_M splits (train+val+test). Complements the de-confounding BoW result
(matched-question variant drives question/caption BoW to chance) reported separately.

    python scripts/55_dataset_validation.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import yaml

WORD = re.compile(r"[a-z0-9]+")


def words(s):
    return set(WORD.findall(str(s).lower()))


def main() -> None:
    cfg = yaml.safe_load(Path("configs/pilot_multidomain_v1.yaml").read_text())
    dm = Path(cfg["paths"]["dm_dir"])
    rows = []
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if len(r.get("options", [])) == 4]
    n = len(rows)
    print(f"D_M items (4-option): {n}\n")

    # (1) position bias
    ci = np.array([r["correct_index"] for r in rows])
    counts = np.bincount(ci, minlength=4)
    exp = n / 4
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    print("=== (1) position bias: correct_index distribution ===")
    print("  " + "  ".join(f"{'ABCD'[i]}={counts[i]} ({counts[i]/n:.3f})" for i in range(4)))
    print(f"  chi-square vs uniform = {chi2:.1f} (df=3; >7.8 => p<0.05 non-uniform)\n")

    # (2) length bias -- report the EXPLOITABLE metric (pick-longest accuracy with
    # random tie-breaking), not P(correct-is-longest) which ties inflate.
    nwords = [[len(WORD.findall(str(o))) for o in r["options"]] for r in rows]
    corr_len = np.array([nwords[i][ci[i]] for i in range(n)])
    dist_len = np.array([np.mean([nwords[i][j] for j in range(4) if j != ci[i]]) for i in range(n)])
    is_longest = np.array([nwords[i][ci[i]] == max(nwords[i]) for i in range(n)])
    def pick_longest_acc(idxs):
        a = 0.0
        for i in idxs:
            mx = max(nwords[i]); tied = [j for j in range(4) if nwords[i][j] == mx]
            a += (1.0 / len(tied)) if ci[i] in tied else 0.0
        return a / len(idxs)
    print("=== (2) length bias ===")
    print(f"  mean words: correct={corr_len.mean():.2f}  distractors={dist_len.mean():.2f}")
    print(f"  P(correct among longest) = {is_longest.mean():.3f} (tie-inflated; not the exploitability)")
    print(f"  pick-longest accuracy (tie-broken) = {pick_longest_acc(range(n)):.3f}  (chance 0.25; "
          f"model single-modality acc is far higher)\n")

    # (3) lexical leakage: per-option surface-feature logistic -> is_correct
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, cross_val_score
    X, y, grp = [], [], []
    for gi, r in enumerate(rows):
        qw = words(r["question"])
        for j, o in enumerate(r["options"]):
            ow = words(o)
            X.append([len(WORD.findall(str(o))), len(str(o)),
                      len(qw & ow) / max(len(ow), 1),
                      1.0 if re.fullmatch(r"[-+]?\d[\d,.]*", str(o).strip()) else 0.0])
            y.append(1 if j == r["correct_index"] else 0); grp.append(gi)
    X = np.array(X); y = np.array(y); grp = np.array(grp)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    auc = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=grp, scoring="roc_auc").mean()
    print("=== (3) lexical leakage: pick correct option from SURFACE features only ===")
    print(f"  per-option AUROC = {auc:.3f}  (0.50 = no text tell; >0.55 = surface leakage)")


if __name__ == "__main__":
    main()
