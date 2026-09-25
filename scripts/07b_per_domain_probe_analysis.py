"""Per-domain probe + contrastive analysis on multidomain_v1.

1. Multi-domain probe accuracy SLICED by domain (does the global probe work for each?)
2. Per-domain probe trained + evaluated within each domain
3. Per-domain contrastive vector norms (proxy for how separable V/T are by mean diff)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent

ACT_DIR = ROOT / "data/activations/qwen/dm_multidomain_v1"
DM_ALL = ROOT / "data/dm/multidomain_v1/dm_all.jsonl"
PROBE_DIR = ROOT / "data/probes/qwen/multidomain_v1"
OUT_PATH = PROBE_DIR / "per_domain.json"

LAYERS_OF_INTEREST = [11, 13, 15, 20, 23, 24, 27, 28]
DOMAINS = ["dci", "vistext", "semart", "roco"]


def main() -> None:
    print("loading activations...")
    acts = np.load(ACT_DIR / "activations.npy")  # (N, 36, D)
    index = [json.loads(l) for l in (ACT_DIR / "index.jsonl").read_text().splitlines() if l.strip()]
    assert acts.shape[0] == len(index)

    # join source field from dm_all
    src_by_cid = {}
    for line in DM_ALL.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            src_by_cid[r["candidate_id"]] = r["source"]

    labels = np.array([0 if r["label"] == "vision" else 1 for r in index])
    splits = np.array([r["split"] for r in index])
    sources = np.array([src_by_cid.get(r["candidate_id"], "?") for r in index])

    train_mask = splits == "train_pass"
    val_mask = splits == "val_pass"

    out: dict = {"layers": LAYERS_OF_INTEREST, "domains": DOMAINS,
                 "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                 "results": []}

    for L in LAYERS_OF_INTEREST:
        X_tr = acts[train_mask, L].astype(np.float32)
        y_tr = labels[train_mask]
        src_tr = sources[train_mask]
        X_va = acts[val_mask, L].astype(np.float32)
        y_va = labels[val_mask]
        src_va = sources[val_mask]

        # (a) Global probe (trained on all domains)
        clf_global = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf_global.fit(X_tr, y_tr)

        row = {"layer": L, "global_val_acc": float(accuracy_score(y_va, clf_global.predict(X_va))),
               "per_domain_global": {}, "per_domain_intrad": {},
               "contrastive_norm": {}}

        for d in DOMAINS:
            # (b) Global probe applied to per-domain val
            m_va = src_va == d
            if m_va.sum() > 0:
                acc_g = float(accuracy_score(y_va[m_va], clf_global.predict(X_va[m_va])))
            else:
                acc_g = float("nan")
            row["per_domain_global"][d] = {"n_val": int(m_va.sum()), "val_acc": acc_g}

            # (c) Intra-domain probe (train + eval only on this domain)
            m_tr = src_tr == d
            if m_tr.sum() > 10 and m_va.sum() > 0 and len(set(y_tr[m_tr])) > 1:
                clf_d = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
                clf_d.fit(X_tr[m_tr], y_tr[m_tr])
                acc_d = float(accuracy_score(y_va[m_va], clf_d.predict(X_va[m_va])))
            else:
                acc_d = float("nan")
            row["per_domain_intrad"][d] = {"n_train": int(m_tr.sum()), "val_acc": acc_d}

            # (d) Contrastive direction norm per domain (from train slice)
            mvis = m_tr & (y_tr == 0)
            mtxt = m_tr & (y_tr == 1)
            if mvis.sum() > 0 and mtxt.sum() > 0:
                vec = X_tr[mvis].mean(0) - X_tr[mtxt].mean(0)
                row["contrastive_norm"][d] = float(np.linalg.norm(vec))
            else:
                row["contrastive_norm"][d] = float("nan")

        # global contrastive (all train)
        vec_g = X_tr[y_tr == 0].mean(0) - X_tr[y_tr == 1].mean(0)
        row["contrastive_norm"]["__global__"] = float(np.linalg.norm(vec_g))

        out["results"].append(row)
        print(f"\n=== L{L:>2} (global val_acc={row['global_val_acc']:.3f}) ===")
        print(f"  {'domain':<8} {'n_val':>5} {'global':>8} {'intrad':>8} {'|μV-μT|':>10}")
        for d in DOMAINS:
            n = row["per_domain_global"][d]["n_val"]
            g = row["per_domain_global"][d]["val_acc"]
            i = row["per_domain_intrad"][d]["val_acc"]
            c = row["contrastive_norm"][d]
            print(f"  {d:<8} {n:>5} {g:>8.3f} {i:>8.3f} {c:>10.2f}")

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
