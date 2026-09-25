"""Step 16b: gated patching analysis (post-processing only, no GPU).

We have full-pool V-patch and T-patch results. The probe predicts V/T from
h_VT with ~1.000 val acc. If we GATE the patch on the probe prediction:
  - probe says V → apply V-patch (use patched_pred)
  - probe says T → do nothing (use baseline_pred)

This is input-conditional patching, the upper bound on any input-conditional
steering. The pool-level Δacc tells us whether gating fixes the catastrophic
T/pass collateral that made unconditional V-patching net-zero past L27.

We approximate the probe by either:
  (a) oracle: use the row's actual label (oracle V/T) as the gate decision.
      This is the ceiling for a perfect classifier.
  (b) real probe: load the L20 probe coefs, evaluate on h_VT(i, L=20) per
      row, use the predicted class as the gate.

If (a) and (b) give the same answer to within ~1pp, the probe is perfect
enough that the difference is moot. If (a) substantially beats (b), we'd
need a better classifier (or per-item α calibration).

Output: stdout table + data/diagnostics/gated_patching_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-patch-dir", default="data/diagnostics/patching_V_full_pool")
    ap.add_argument("--t-patch-dir", default="data/diagnostics/patching_T_full_pool")
    ap.add_argument("--act-dir", default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--probes-dir", default="data/probes/qwen/multidomain_v1")
    ap.add_argument("--probe-layer", type=int, default=20)
    ap.add_argument("--patch-layers", default="27,29,31,33,35")
    ap.add_argument("--out",
                    default="data/diagnostics/gated_patching_summary.json")
    args = ap.parse_args()

    # 06c letter logits → V-only / T-only correctness per cid
    act_dir = Path(args.act_dir)
    logits = np.load(act_dir / "letter_logits.npy")     # (N, 3, 4) [VT, V, T]
    cf_index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    correct = {r["candidate_id"]: r["correct_index"] for r in cf_index}
    v_ok = {cid: bool(logits[i, 1].argmax() == correct[cid]) for cid, i in cid_to_idx.items()}
    t_ok = {cid: bool(logits[i, 2].argmax() == correct[cid]) for cid, i in cid_to_idx.items()}

    # probe prediction at probe_layer (using h_VT, condition 0)
    probe = np.load(Path(args.probes_dir) / f"probe_layer{args.probe_layer:02d}.npz")
    coef = probe["coef"][0].astype(np.float32)
    intercept = float(probe["intercept"][0])
    acts = np.load(act_dir / "activations.npy")  # (N, 3, 36, 2048)

    def probe_predict_label(cid: str) -> str:
        """Return 'vision' or 'text' per the probe applied to h_VT(L_probe)."""
        i = cid_to_idx[cid]
        h = acts[i, 0, args.probe_layer].astype(np.float32)
        score = float(np.dot(coef, h) + intercept)
        # sklearn convention: score>0 -> class 1 (text). class index 1=text.
        return "text" if score > 0 else "vision"

    # load V-patch and T-patch results
    def load_patch(d):
        return [json.loads(l) for l in (Path(d) / "results.jsonl").read_text().splitlines() if l.strip()]
    v_rows = load_patch(args.v_patch_dir)
    t_rows = load_patch(args.t_patch_dir)
    # key by (cid, layer) for fast lookup
    v_idx = {(r["candidate_id"], r["layer"]): r for r in v_rows}
    t_idx = {(r["candidate_id"], r["layer"]): r for r in t_rows}

    # build dedup'd pool: one row per cid (use the V-patch row since it has the same
    # baseline_pred, label, correct_index per cid; layer is grid-scanned separately)
    layers = [int(x) for x in args.patch_layers.split(",")]
    cids_by_layer = defaultdict(set)
    for r in v_rows:
        cids_by_layer[r["layer"]].add(r["candidate_id"])
    for r in t_rows:
        cids_by_layer[r["layer"]].add(r["candidate_id"])

    def subpool(r):
        label = r["label"]; bp = r["baseline_pred"]; ci = r["correct_index"]
        if label == "vision":
            if bp == ci: return "V/pass"
            return "V/fail∧V✓" if v_ok.get(r["candidate_id"], False) else "V/fail∧V✗"
        else:
            if bp == ci: return "T/pass"
            return "T/fail∧T✓" if t_ok.get(r["candidate_id"], False) else "T/fail∧T✗"

    SUBPOOLS = ["V/pass", "V/fail∧V✓", "V/fail∧V✗", "T/pass", "T/fail∧T✓", "T/fail∧T✗"]
    out_summary = {}

    for L in layers:
        if L not in cids_by_layer:
            continue
        # build canonical row set for this layer (one row per unique cid)
        canonical = {}
        for r in v_rows + t_rows:
            if r["layer"] != L: continue
            cid = r["candidate_id"]
            if cid not in canonical:
                canonical[cid] = r

        per_subpool = defaultdict(lambda: {"n": 0,
                                           "base": 0,
                                           "uncond_V": 0,
                                           "uncond_T": 0,
                                           "gated_oracle": 0,
                                           "gated_probe": 0,
                                           "sym_oracle": 0,
                                           "sym_probe": 0})

        for cid, r in canonical.items():
            sp = subpool(r)
            d = per_subpool[sp]
            d["n"] += 1
            base = (r["baseline_pred"] == r["correct_index"])
            d["base"] += int(base)
            # unconditional V-patch
            v_r = v_idx.get((cid, L))
            uV = (v_r["patched_pred"] == r["correct_index"]) if v_r is not None else base
            d["uncond_V"] += int(uV)
            # unconditional T-patch
            t_r = t_idx.get((cid, L))
            uT = (t_r["patched_pred"] == r["correct_index"]) if t_r is not None else base
            d["uncond_T"] += int(uT)
            # gated by oracle label (asymmetric): V-label → V-patch, T-label → nothing
            gated_o = uV if r["label"] == "vision" else base
            d["gated_oracle"] += int(gated_o)
            # gated by probe prediction (asymmetric)
            probe_label = probe_predict_label(cid)
            gated_p = uV if probe_label == "vision" else base
            d["gated_probe"] += int(gated_p)
            # SYMMETRIC: V-label → V-patch, T-label → T-patch
            sym_o = uV if r["label"] == "vision" else uT
            d["sym_oracle"] += int(sym_o)
            sym_p = uV if probe_label == "vision" else uT
            d["sym_probe"] += int(sym_p)

        # pool-level rollup
        cells_summary = {}
        KEYS = ["n", "base", "uncond_V", "uncond_T",
                "gated_oracle", "gated_probe", "sym_oracle", "sym_probe"]
        totals = {k: 0 for k in KEYS}
        for sp in SUBPOOLS:
            if sp not in per_subpool: continue
            d = per_subpool[sp]
            cells_summary[sp] = d
            for k in totals: totals[k] += d[k]

        n = totals["n"]
        if n == 0: continue
        out_summary[L] = {
            "n": n,
            "baseline_acc": totals["base"] / n,
            "uncond_V_acc": totals["uncond_V"] / n,
            "uncond_T_acc": totals["uncond_T"] / n,
            "gated_oracle_acc": totals["gated_oracle"] / n,
            "gated_probe_acc":  totals["gated_probe"]  / n,
            "sym_oracle_acc":   totals["sym_oracle"]   / n,
            "sym_probe_acc":    totals["sym_probe"]    / n,
            "cells": cells_summary,
        }

    # print pool-level table
    print(f"\n=== Pool-level acc (probe@L{args.probe_layer}) ===")
    print(f"{'L':<5s}{'base':>8s}{'unV':>8s}{'unT':>8s}{'gV-o':>8s}{'gV-p':>8s}{'sym-o':>8s}{'sym-p':>8s}"
          f"{'ΔgV':>8s}{'Δsym':>8s}")
    for L in sorted(out_summary):
        s = out_summary[L]
        b = s["baseline_acc"]
        print(f"L{L:<4d}{b:>8.3f}{s['uncond_V_acc']:>8.3f}{s['uncond_T_acc']:>8.3f}"
              f"{s['gated_oracle_acc']:>8.3f}{s['gated_probe_acc']:>8.3f}"
              f"{s['sym_oracle_acc']:>8.3f}{s['sym_probe_acc']:>8.3f}"
              f"{s['gated_probe_acc']-b:>+8.3f}{s['sym_probe_acc']-b:>+8.3f}")
    print("  Legend: unV/unT=unconditional V/T patch; gV-o/gV-p=asymmetric gate by oracle/probe (V-label→V-patch, T-label→nothing); "
          "sym-o/sym-p=symmetric gate (V-label→V-patch, T-label→T-patch).")

    # stratified breakdown for the best symmetric-gated layer
    best_L = max(out_summary, key=lambda L: out_summary[L]["sym_probe_acc"])
    print(f"\n=== Stratified breakdown at best symmetric-gated layer L{best_L} ===")
    print(f"{'subpool':<14s}{'n':>5s}{'baseline':>10s}{'uncondV':>10s}{'gatedV(prb)':>13s}{'symGate(prb)':>14s}")
    for sp in SUBPOOLS:
        c = out_summary[best_L]["cells"].get(sp)
        if not c: continue
        print(f"  {sp:<12s}{c['n']:>5d}{c['base']/c['n']:>10.3f}"
              f"{c['uncond_V']/c['n']:>10.3f}{c['gated_probe']/c['n']:>13.3f}{c['sym_probe']/c['n']:>14.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_summary, indent=2, default=lambda o: list(o) if isinstance(o, set) else o))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
