"""Step 21: question-type confounder analysis via INLP de-confounding.

Uses h_Q (question+options only; scripts/06d) as the pure question-encoding
representation, and asks whether V/T is separable in h_VT BEYOND what the
question text alone encodes.

Per layer:
  1. confounder probe: V/T accuracy from h_Q alone (how much label is in the question).
  2. raw h_VT probe: baseline (matches the point-2 numbers).
  3. INLP de-confounding: iteratively (a) fit a V/T probe on h_Q, (b) project its
     direction out of BOTH h_Q and h_VT, then refit a fresh probe on the
     de-confounded h_VT. Report accuracy vs. #directions removed, on val_pass and
     on the vision-FAIL set. The h_Q probe accuracy (should fall to chance) is the
     sanity that we are actually nullifying the question-type subspace.

Read: if de-confounded h_VT stays well above chance after h_Q is driven to chance,
there is a modality-content axis in h_VT separate from question-type. If it collapses
with h_Q, all the V/T signal in h_VT lived in the question-type subspace.

    python scripts/21_question_confound_probe.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")


def load_store(path: Path, cond: int | None):
    acts = np.load(path / "activations.npy", mmap_mode="r")
    idx = [json.loads(l) for l in (path / "index.jsonl").read_text().splitlines() if l.strip()]
    return acts, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--hq-dir", default="data/activations/qwen/dm_multidomain_v1_hq")
    ap.add_argument("--layers", default="20,28,31")
    ap.add_argument("--rounds", type=int, default=8, help="INLP directions to remove")
    args = ap.parse_args()

    cf = Path(args.cf_dir); hq = Path(args.hq_dir)
    vt_acts = np.load(cf / "activations.npy", mmap_mode="r")            # (N,3,36,d) [VT,V,T]
    vt_idx = [json.loads(l) for l in (cf / "index.jsonl").read_text().splitlines() if l.strip()]
    vt_ll = np.load(cf / "letter_logits.npy")
    q_acts = np.load(hq / "activations.npy", mmap_mode="r")             # (M,36,d)
    q_idx = [json.loads(l) for l in (hq / "index.jsonl").read_text().splitlines() if l.strip()]

    # join by candidate_id
    q_pos = {r["candidate_id"]: i for i, r in enumerate(q_idx)}
    common = [(i, q_pos[r["candidate_id"]]) for i, r in enumerate(vt_idx) if r["candidate_id"] in q_pos]
    vi = np.array([a for a, _ in common]); qi = np.array([b for b, _ in common])
    idx = [vt_idx[a] for a, _ in common]
    print(f"joined {len(common)} items (cf {len(vt_idx)}, hq {len(q_idx)})")

    label = np.array([0 if r["label"] == "vision" else 1 for r in idx])   # 0=V,1=T
    split = np.array([r["split"] for r in idx])
    vtok = np.array([int(vt_ll[a, 0].argmax()) == idx[k]["correct_index"] for k, (a, _) in enumerate(common)])
    train = split == "train_pass"; valp = split == "val_pass"
    visfail = (label == 0) & (~vtok)
    chance = max(label[valp].mean(), 1 - label[valp].mean())
    print(f"train_pass={train.sum()} val_pass={valp.sum()} vis-fail={visfail.sum()} | val chance={chance:.3f}\n")

    def lr():
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    def acc(clf, X, m):
        return (clf.predict(X[m]) == label[m]).mean()

    for L in [int(x) for x in args.layers.split(",")]:
        Xvt = np.asarray(vt_acts[vi, 0, L, :], dtype=np.float32).copy()   # h_VT(L)
        Xq = np.asarray(q_acts[qi, L, :], dtype=np.float32).copy()        # h_Q(L)
        print(f"=== L{L} ===")
        print(f"{'rnd':>3} {'hQ_val':>7} {'hVT_val':>8} {'hVT_visFAIL':>12}")
        for r in range(args.rounds + 1):
            cq = lr().fit(Xq[train], label[train])
            cvt = lr().fit(Xvt[train], label[train])
            print(f"{r:>3} {acc(cq,Xq,valp):>7.3f} {acc(cvt,Xvt,valp):>8.3f} {acc(cvt,Xvt,visfail):>12.3f}")
            # remove the current question-type direction (from hQ probe) from BOTH
            w = cq.coef_[0].astype(np.float32); u = w / (np.linalg.norm(w) + 1e-8)
            Xq = Xq - np.outer(Xq @ u, u)
            Xvt = Xvt - np.outer(Xvt @ u, u)
        print()


if __name__ == "__main__":
    main()
