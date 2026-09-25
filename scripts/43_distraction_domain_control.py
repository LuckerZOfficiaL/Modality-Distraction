"""X1: is the domain-graded v-distraction intrinsic, or a caption-length confound?

The corrected causal asymmetry (memory/log) found v-distraction graded by domain
(roco 11% > vistext 8.5% > semart 7.7% > dci 4.6%). But radiology/chart captions
differ in length & style from photo captions, and the token-ratio analysis already
showed caption length dilutes the image cue. Before leaning on "harder visual domains
are more caption-distractable", we (a) put Wilson CIs on the per-domain rates (roco's
~11% is on few items), and (b) test whether domain survives controlling for caption
token length (logistic: distracted ~ z(caplen) + C(domain)) and whether the grading
persists within matched caption-length strata.

Offline: cf-store index + letter_logits + splits (source, caption_for_filter) + Qwen tokenizer (CPU).

    python scripts/43_distraction_domain_control.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    st = dr / "activations" / "qwen" / args.cf_subdir
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    meta = {}
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); meta[r["candidate_id"]] = r

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)

    ci = np.array([r["correct_index"] for r in idx])
    lab = np.array([r["label"] for r in idx])
    vt_pred = ll[:, 0].argmax(1); v_pred = ll[:, 1].argmax(1)
    v_ok = v_pred == ci
    vt_wrong = vt_pred != ci

    # vision items, V-only correct, joined to a row (for source + caption)
    rows = []
    for i, r in enumerate(idx):
        cid = r["candidate_id"]
        if lab[i] != "vision" or not v_ok[i] or cid not in meta:
            continue
        m = meta[cid]
        caplen = len(tok(m["caption_for_filter"], add_special_tokens=False)["input_ids"])
        rows.append({"source": m["source"], "caplen": caplen, "distracted": int(vt_wrong[i])})
    src = np.array([r["source"] for r in rows])
    caplen = np.array([r["caplen"] for r in rows], float)
    dist = np.array([r["distracted"] for r in rows])
    domains = ["dci", "vistext", "semart", "roco"]

    print(f"\n=== X1: v-distraction by domain (vision items, V-only correct; n={len(rows)}) ===")
    print(f"{'domain':9}{'n':>6}{'distracted':>12}{'rate':>8}{'95% Wilson CI':>18}{'med caplen':>12}")
    for d in domains:
        m = src == d; n = int(m.sum()); k = int(dist[m].sum())
        lo, hi = wilson(k, n)
        print(f"{d:9}{n:>6}{k:>12}{k/max(n,1):>8.3f}   [{lo:.3f},{hi:.3f}]   {np.median(caplen[m]):>9.0f}")

    # logistic: distracted ~ z(caplen) + C(domain), ref=dci
    print("\n=== Logistic: distracted ~ z(caplen) + C(domain), ref=dci ===")
    z = (caplen - caplen.mean()) / caplen.std()
    X_cols = [("z_caplen", z)]
    for d in domains[1:]:
        X_cols.append((f"dom[{d}]", (src == d).astype(float)))
    X = np.column_stack([c for _, c in X_cols])
    names = [n for n, _ in X_cols]
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(X)
        res = sm.Logit(dist, Xc).fit(disp=0)
        coefs = res.params; pvals = res.pvalues
        print(f"{'term':14}{'coef':>9}{'odds-ratio':>12}{'p':>10}")
        print(f"{'const':14}{coefs[0]:>9.3f}{math.exp(coefs[0]):>12.3f}{'':>10}")
        for j, nm in enumerate(names):
            print(f"{nm:14}{coefs[j+1]:>9.3f}{math.exp(coefs[j+1]):>12.3f}{pvals[j+1]:>10.4f}")
        print("(domain coef = log-odds of distraction vs dci, controlling for caption length)")
    except Exception as e:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000).fit(X, dist)
        print(f"(statsmodels unavailable: {e}; sklearn coefs, no p-values)")
        print(f"{'z_caplen':14}{clf.coef_[0][0]:>9.3f}")
        for j, nm in enumerate(names[1:], start=1):
            print(f"{nm:14}{clf.coef_[0][j]:>9.3f}")

    # grading persistence within caption-length strata (global tertiles)
    print("\n=== Distraction rate by GLOBAL caption-length tertile x domain ===")
    qs = np.quantile(caplen, [1/3, 2/3])
    tert = np.digitize(caplen, qs)  # 0,1,2
    tnames = [f"short(<{qs[0]:.0f})", f"mid", f"long(>{qs[1]:.0f})"]
    print(f"{'domain':9}" + "".join(f"{t:>16}" for t in tnames))
    for d in domains:
        cells = []
        for t in range(3):
            m = (src == d) & (tert == t); n = int(m.sum()); k = int(dist[m].sum())
            cells.append(f"{k}/{n} ({k/max(n,1):.2f})")
        print(f"{d:9}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
