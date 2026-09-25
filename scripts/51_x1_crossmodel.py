"""X1 cross-model: replicate the distraction-by-domain + caption-length control
(scripts/43) for EVERY backbone, reading the T8 D_M 3-condition predictions
(data/eval/t8/{model}.jsonl) -- no cf-store needed, so it runs for LLaVA-NeXT /
distracted ~ z(caplen) + C(domain) to show the natural-vs-non-natural split is not a
caption-length artifact in any backbone.

    python scripts/51_x1_crossmodel.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml

DOMAINS = ["dci", "vistext", "semart", "roco"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--t8-dir", default="data/eval/t8")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm = Path(cfg["paths"]["dm_dir"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    caplen = {}
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    caplen[r["candidate_id"]] = len(tok(r["caption_for_filter"], add_special_tokens=False)["input_ids"])

    from sklearn.linear_model import LogisticRegression
    for fp in sorted(glob.glob(str(Path(args.t8_dir) / "*.jsonl"))):
        model = Path(fp).stem
        rows = [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["label"] == "vision" and r["pred_v"] == r["correct_index"]
                and r["candidate_id"] in caplen and r.get("source") in DOMAINS]  # vision V-solvable
        src = np.array([r["source"] for r in rows])
        cl = np.array([caplen[r["candidate_id"]] for r in rows], float)
        dist = np.array([int(r["pred_vt"] != r["correct_index"]) for r in rows])
        print(f"\n=== {model}: v-distraction by domain (vision V-solvable, n={len(rows)}) ===")
        print(f"{'domain':9}{'n':>6}{'distr':>7}{'rate':>8}{'95% Wilson':>18}{'med caplen':>12}")
        for d in DOMAINS:
            m = src == d; n = int(m.sum()); k = int(dist[m].sum()); lo, hi = wilson(k, n)
            print(f"{d:9}{n:>6}{k:>7}{k/max(n,1):>8.3f}   [{lo:.3f},{hi:.3f}]   {np.median(cl[m]) if n else 0:>9.0f}")
        nat = dist[src == "dci"].mean()
        nonnat = dist[np.isin(src, ["vistext", "semart", "roco"])].mean()
        print(f"  natural(dci) {nat:.3f}  vs non-natural {nonnat:.3f}  (ratio {nonnat/max(nat,1e-9):.2f}x)")
        # logistic distracted ~ z(caplen) + C(domain), ref=dci
        z = (cl - cl.mean()) / cl.std()
        X = np.column_stack([z] + [(src == d).astype(float) for d in DOMAINS[1:]])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, dist)
        co = clf.coef_[0]
        print(f"  logistic coefs: z_caplen {co[0]:+.3f} | " +
              " ".join(f"{d} {co[i+1]:+.3f}(OR{math.exp(co[i+1]):.2f})" for i, d in enumerate(DOMAINS[1:])))


if __name__ == "__main__":
    main()
