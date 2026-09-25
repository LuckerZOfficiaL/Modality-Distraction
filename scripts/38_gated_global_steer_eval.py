"""Step 38: cosine-gated global-vector steering eval (offline, after the harness sweep).

Approach 2 confirmatory eval. The GPU harness (scripts/19 --method additive with the
D_M image-use vector, --cell-prefix vgdm_) produced per-cell steered predictions on the
benchmarks. Here we apply the per-input COSINE GATE offline:

  gated_pred_i = steered_i  if  gate(cos(h_img_i, v_global_L))   else  baseline_i

The gate (sign + threshold) and the steering cell (L, alpha) are selected on the dev
split; reported on held-out test. Compared against: baseline, unconditional steer (all),
gated-real, and gated matched-random null. Headline pool = ViLP prior-violating.

    python scripts/38_gated_global_steer_eval.py --benchmark vilp
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def grouped_split(rows, seed, frac):
    g: dict = {}
    for i, r in enumerate(rows):
        g.setdefault(r.get("question_group", r["candidate_id"]), []).append(i)
    keys = sorted(g); np.random.default_rng(seed).shuffle(keys)
    tk = set(keys[:int(round(len(keys) * frac))]); m = np.zeros(len(rows), bool)
    for k, ii in g.items():
        if k in tk:
            for i in ii: m[i] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--evaldir", default="data/diagnostics/benchmark_eval")
    ap.add_argument("--cf-dir", default=None)
    ap.add_argument("--vglobal", default="data/steering/vglobal_dm_image.npz")
    ap.add_argument("--prefix", default="vgdm_")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()

    ev = Path(args.evaldir)
    cf = Path(args.cf_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    base_rows = [json.loads(l) for l in (ev / f"baseline_{args.benchmark}.jsonl").read_text().splitlines() if l.strip()]
    cid_order = [r["candidate_id"] for r in base_rows]
    pos = {c: i for i, c in enumerate(cid_order)}
    ci = np.array([r["correct_index"] for r in base_rows])
    pv = np.array([r.get("prior_consistent") is False for r in base_rows])
    base_pred = np.array([r["pred"] for r in base_rows])

    # align cf-store (h_img + index) to baseline cid order
    cf_idx = [json.loads(l) for l in (cf / "index.jsonl").read_text().splitlines() if l.strip()]
    cf_pos = {r["candidate_id"]: i for i, r in enumerate(cf_idx)}
    take = np.array([cf_pos[c] for c in cid_order])
    h_img = np.load(cf / "h_img.npy")[take]
    sv = np.load(args.vglobal)

    qg = {}
    for l in Path(f"data/benchmarks/{args.benchmark}/rows.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); qg[r["candidate_id"]] = r.get("question_group", r["candidate_id"])
    rows_for_split = [{"candidate_id": c, "question_group": qg.get(c, c)} for c in cid_order]
    is_test = grouped_split(rows_for_split, args.seed, args.test_frac); dev = ~is_test
    bc = base_pred == ci

    def load_cell(path):
        d = {json.loads(l)["candidate_id"]: json.loads(l)["pred"] for l in Path(path).read_text().splitlines() if l.strip()}
        return np.array([d.get(c, base_pred[i]) for i, c in enumerate(cid_order)])

    def acc(pred, m):
        return (pred[m] == ci[m]).mean() if m.sum() else float("nan")

    cells = {}
    for p in glob.glob(str(ev / f"{args.prefix}*_{args.benchmark}.jsonl")):
        m = re.search(rf"{args.prefix}(real|random)_L(\d+)_a([pm])([\d.]+)_{args.benchmark}", p)
        if m:
            cells[(m.group(1), int(m.group(2)), float(m.group(4)) * (1 if m.group(3) == "p" else -1))] = p
    layers = sorted({L for (_, L, _) in cells})
    print(f"=== {args.benchmark} | {len(cells)} cells | layers {layers} ===")
    base_test = acc(base_pred, is_test); base_pv = acc(base_pred, is_test & pv)
    print(f"baseline: test {base_test:.3f}" + (f" | pv {base_pv:.3f}" if pv.any() else ""))

    # cos(h_img, v_global) per layer
    cosL = {L: (h_img[:, L].astype(np.float32) @ sv[f"v_V_L{L}"]) /
               (np.linalg.norm(h_img[:, L].astype(np.float32), axis=1) + 1e-8) for L in layers}

    headline = pv if pv.any() else np.ones(len(ci), bool)   # ViLP: prior-violating; else all
    best = None
    for (tag, L, a), path in cells.items():
        if tag != "real":
            continue
        steered = load_cell(path)
        cosv = cosL[L]
        for sign in (+1, -1):
            for q in np.linspace(0.1, 0.9, 17):
                thr = np.quantile(cosv, q)
                fire = (cosv >= thr) if sign > 0 else (cosv <= thr)
                gated = np.where(fire, steered, base_pred)
                dev_h = acc(gated, dev & headline)
                if best is None or dev_h > best[0]:
                    best = (dev_h, tag, L, a, sign, thr, fire, steered)
    if best is None:
        print("no real cells found"); return
    _, tag, L, a, sign, thr, fire, steered = best
    nullpath = cells.get(("random", L, a))
    gated = np.where(fire, steered, base_pred)
    print(f"\nselected: L{L} a={a} gate cos {'>=' if sign>0 else '<='} {thr:.3f} (fires {fire[is_test].mean():.2f} of test)")
    print(f"{'':18}{'overall':>9}" + ("   pv" if pv.any() else ""))
    def line(name, pred):
        s = f"{name:18}{acc(pred,is_test):>9.3f}"
        if pv.any(): s += f"  {acc(pred,is_test&pv):.3f}"
        print(s)
    line("baseline", base_pred)
    line("uncond steer", steered)
    line("GATED real", gated)
    if nullpath:
        snull = load_cell(nullpath)
        line("GATED null", np.where(fire, snull, base_pred))


if __name__ == "__main__":
    main()
