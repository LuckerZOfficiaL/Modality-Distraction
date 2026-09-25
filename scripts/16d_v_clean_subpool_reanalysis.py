"""Step 16d: re-analyze the existing 12-cell steering sweep on the V/fail ∧ V✓
subpool only. Compare to the random-vector null on the same filtered subpool.

The full V/fail pool mixed V/fail ∧ V✓ (~117 items, causally recoverable per
patching) with V/fail ∧ V✗ (~228 items, not modality-routing failures). Prior
steering results were dominated by noise-floor recovery on V/fail ∧ V✗ — which
is why random matched real on Δpool. On the V/fail ∧ V✓ subset alone we can
ask: does any prior method have CAUSAL recovery beyond the random null?

For each (method, variant, L, α) cell:
  V/fail_clean Δacc = mean[steered_correct - 0]  over V/fail ∧ V✓ items
For random null, same.
Report the largest real-vs-random gap on V/fail ∧ V✓ per layer.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--steering-root",
                    default="data/steering/qwen_multidomain_v1")
    args = ap.parse_args()

    # V-only correctness per cid (defines V✓)
    act_dir = Path(args.act_dir)
    logits = np.load(act_dir / "letter_logits.npy")
    cf_index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    correct_by_cid = {r["candidate_id"]: r["correct_index"] for r in cf_index}
    v_only_arg = {cid: int(logits[i, 1].argmax()) for cid, i in cid_to_idx.items()}
    v_ok = {cid: v_only_arg[cid] == correct_by_cid[cid] for cid in cid_to_idx}

    PAIRS = [
        ("probe",       "plain",    "probe_unseen",                       "random_probe_unseen_seed0"),
        ("contrastive", "plain",    "contrastive_unseen",                 "random_contrastive_unseen_seed0"),
        ("SAE",         "plain",    "sae_unseen",                         "random_sae_unseen_seed0"),
        ("probe",       "bcast+nm", "probe_unseen_bcast_normmatch",       "random_probe_unseen_bcast_normmatch_seed0"),
    ]

    def vfail_clean_per_cell(d):
        """For each (L, α) in run d, return V/fail∧V✓ recovery rate."""
        rows = [json.loads(l) for l in (Path(args.steering_root) / d / "results.jsonl").read_text().splitlines() if l.strip()]
        base = {r["candidate_id"]: r for r in rows if r["layer"] == -1}
        item = {}
        for r in rows:
            if r["layer"] == -1: continue
            cid = r["candidate_id"]
            if cid not in base: continue
            if cid in item: continue
            item[cid] = {"label": r["label"], "ci": r["correct_index"],
                         "bc": base[cid]["pred"] == r["correct_index"]}
        cells = defaultdict(list)
        for r in rows:
            if r["layer"] == -1: continue
            cid = r["candidate_id"]
            m = item.get(cid)
            if not m or m["label"] != "vision" or m["bc"]: continue   # V/fail only
            if not v_ok.get(cid, False): continue                      # V✓ only
            cells[(r["layer"], r["alpha"])].append(int(r["pred"] == m["ci"]))
        return {k: (sum(v) / len(v), len(v)) for k, v in cells.items()}

    for method, variant, real_d, rand_d in PAIRS:
        real = vfail_clean_per_cell(real_d)
        rand = vfail_clean_per_cell(rand_d)
        common = sorted({k for k in real if k in rand and k[1] != 0.0},
                        key=lambda k: (k[0], abs(k[1]), k[1]))
        if not common:
            print(f"{method}/{variant}: no common cells"); continue
        print(f"\n=== {method} / {variant} | V/fail∧V✓ recovery (n on first match: {real[common[0]][1]}) ===")
        # group by layer
        by_L = defaultdict(list)
        for L, a in common:
            real_acc, n_real = real[(L, a)]
            rand_acc, n_rand = rand[(L, a)]
            by_L[L].append((a, real_acc, rand_acc, real_acc - rand_acc))
        # print per layer
        for L in sorted(by_L):
            print(f"  L{L}:")
            print(f"  {'α':>8s}{'real':>10s}{'random':>10s}{'real−rand':>14s}")
            # sort by α
            for a, r_acc, n_acc, gap in sorted(by_L[L]):
                marker = " *" if abs(gap) > 0.10 else ""
                print(f"  {a:>+8g}{r_acc:>10.3f}{n_acc:>10.3f}{gap:>+14.3f}{marker}")
        # best real-vs-random gap (real beats random) across all common cells
        best = max(by_L.values(), key=lambda lst: max(g[3] for g in lst))
        # actually we want the max gap across ALL (L, α), not just within one L
        all_gaps = [(L, a, r, n, g) for L, lst in by_L.items() for (a, r, n, g) in lst]
        all_gaps.sort(key=lambda x: -x[4])
        L_b, a_b, r_b, n_b, g_b = all_gaps[0]
        print(f"  -> max gap: L{L_b}, α={a_b:+g}: real={r_b:.3f}, random={n_b:.3f}, gap={g_b:+.3f}")


if __name__ == "__main__":
    main()
