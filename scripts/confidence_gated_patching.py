"""Step 16c: modality-confidence gating, post-processing only.

Alternative gate to the L20 probe (16b). For each pool item, use the stored
letter-logit softmax probs P(correct|VT), P(correct|V), P(correct|T) from
06c's letter_logits.npy to decide who needs intervention:

  - If P(correct|V) > P(correct|VT) by margin ε → V-distractable → V-patch.
  - If P(correct|T) > P(correct|VT) by margin ε → T-distractable → T-patch.
  - If neither — item isn't a routing failure → leave at baseline.
  - If both — pick the larger improvement.

The signal is item-specific (per-row) and uses information the question-text
probe doesn't see: single-modality behavioral confidence. Could pick up
t_distracted items (V-style question content but Qwen empirically needs the
caption) that probe@L20 misclassifies as V.

Compared to probe gate (16b), confidence gate is:
  - More expressive: per-item modality-routing diagnosis, not question type.
  - More expensive at deployment: requires V-only AND T-only forward passes
    to know P(correct|·), so 3 forward passes per item even before patching.
  - But upper-bound-relevant: tells us the ceiling for "best possible gate."

Output: stdout + data/diagnostics/confidence_gated_summary.json
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
    ap.add_argument("--v-patch-dir", default="data/diagnostics/patching_V_full_pool")
    ap.add_argument("--t-patch-dir", default="data/diagnostics/patching_T_full_pool")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--probes-dir", default="data/probes/qwen/multidomain_v1")
    ap.add_argument("--probe-layer", type=int, default=20)
    ap.add_argument("--patch-layers", default="27,29,31,33,35")
    ap.add_argument("--margins", default="0.0,0.05,0.1,0.2",
                    help="confidence-margin thresholds ε to sweep "
                         "(P(X) - P(VT) > ε required to gate to X-patch)")
    ap.add_argument("--out",
                    default="data/diagnostics/confidence_gated_summary.json")
    args = ap.parse_args()

    # 06c letter logits per item per condition
    act_dir = Path(args.act_dir)
    logits = np.load(act_dir / "letter_logits.npy")     # (N, 3, 4) [VT, V, T]
    cf_index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    correct = {r["candidate_id"]: r["correct_index"] for r in cf_index}

    # Per-cid: P(correct|VT), P(correct|V), P(correct|T) and argmaxes
    p_vt, p_v, p_t = {}, {}, {}
    arg_v, arg_t = {}, {}
    for cid, i in cid_to_idx.items():
        ci = correct[cid]
        for cond, store, arg_store in [(0, p_vt, None), (1, p_v, arg_v), (2, p_t, arg_t)]:
            probs = softmax(logits[i, cond].astype(np.float32))
            store[cid] = float(probs[ci])
            if arg_store is not None:
                arg_store[cid] = int(probs.argmax())
    v_ok = {cid: arg_v[cid] == correct[cid] for cid in cid_to_idx}
    t_ok = {cid: arg_t[cid] == correct[cid] for cid in cid_to_idx}

    # Probe predictions for the comparison
    probe = np.load(Path(args.probes_dir) / f"probe_layer{args.probe_layer:02d}.npz")
    coef = probe["coef"][0].astype(np.float32)
    intercept = float(probe["intercept"][0])
    acts = np.load(act_dir / "activations.npy")
    probe_label = {}
    for cid, i in cid_to_idx.items():
        h = acts[i, 0, args.probe_layer].astype(np.float32)
        score = float(np.dot(coef, h) + intercept)
        probe_label[cid] = "text" if score > 0 else "vision"

    # Load patching results
    def load_patch(d):
        return [json.loads(l) for l in (Path(d) / "results.jsonl").read_text().splitlines() if l.strip()]
    v_rows = load_patch(args.v_patch_dir)
    t_rows = load_patch(args.t_patch_dir)
    v_idx = {(r["candidate_id"], r["layer"]): r for r in v_rows}
    t_idx = {(r["candidate_id"], r["layer"]): r for r in t_rows}

    layers = [int(x) for x in args.patch_layers.split(",")]
    margins = [float(x) for x in args.margins.split(",")]

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
        canonical = {}
        for r in v_rows + t_rows:
            if r["layer"] != L: continue
            cid = r["candidate_id"]
            if cid not in canonical:
                canonical[cid] = r

        # for each margin threshold, compute pool acc under confidence-gate
        per_eps = {eps: {"n": 0, "base": 0, "gate": 0,
                         "n_route_V": 0, "n_route_T": 0, "n_route_none": 0,
                         "per_subpool": defaultdict(lambda: {"n": 0, "base": 0, "gate": 0})}
                   for eps in margins}
        probe_pool = {"n": 0, "base": 0, "gate": 0,
                      "per_subpool": defaultdict(lambda: {"n": 0, "base": 0, "gate": 0})}

        for cid, r in canonical.items():
            ci = r["correct_index"]
            base = (r["baseline_pred"] == ci)
            sp = subpool(r)
            v_r = v_idx.get((cid, L))
            t_r = t_idx.get((cid, L))
            v_correct = (v_r["patched_pred"] == ci) if v_r is not None else base
            t_correct = (t_r["patched_pred"] == ci) if t_r is not None else base

            p_VT = p_vt[cid]; p_V = p_v[cid]; p_T = p_t[cid]

            # Probe-gate (asymmetric, V-patch only when probe says V)
            probe_gate_correct = v_correct if probe_label[cid] == "vision" else base
            probe_pool["n"] += 1
            probe_pool["base"] += int(base)
            probe_pool["gate"] += int(probe_gate_correct)
            probe_pool["per_subpool"][sp]["n"] += 1
            probe_pool["per_subpool"][sp]["base"] += int(base)
            probe_pool["per_subpool"][sp]["gate"] += int(probe_gate_correct)

            # Confidence-gate per margin
            d_V = p_V - p_VT
            d_T = p_T - p_VT
            for eps in margins:
                if d_V > eps and d_V >= d_T:
                    pred_correct = v_correct
                    route = "V"
                elif d_T > eps and d_T > d_V:
                    pred_correct = t_correct
                    route = "T"
                else:
                    pred_correct = base
                    route = "none"
                d = per_eps[eps]
                d["n"] += 1
                d["base"] += int(base)
                d["gate"] += int(pred_correct)
                d[f"n_route_{route}"] += 1
                d["per_subpool"][sp]["n"] += 1
                d["per_subpool"][sp]["base"] += int(base)
                d["per_subpool"][sp]["gate"] += int(pred_correct)

        out_summary[L] = {
            "probe": {**probe_pool,
                      "pool_acc": probe_pool["gate"]/probe_pool["n"],
                      "baseline_acc": probe_pool["base"]/probe_pool["n"],
                      "delta": (probe_pool["gate"] - probe_pool["base"]) / probe_pool["n"],
                      "per_subpool": {k: dict(v) for k, v in probe_pool["per_subpool"].items()}},
            "confidence": {f"eps={eps}": {
                **{k: v for k, v in d.items() if k != "per_subpool"},
                "pool_acc": d["gate"]/d["n"],
                "baseline_acc": d["base"]/d["n"],
                "delta": (d["gate"] - d["base"]) / d["n"],
                "per_subpool": {k: dict(v) for k, v in d["per_subpool"].items()},
            } for eps, d in per_eps.items()},
        }

    # Print headline table
    print("\n=== Pool Δacc by gate type (per layer) ===")
    print(f"{'L':<5s}{'base':>8s}{'probe-gate':>12s}", end="")
    for eps in margins: print(f"{'conf ε='+str(eps):>14s}", end="")
    print()
    for L in sorted(out_summary):
        s = out_summary[L]
        b = s["probe"]["baseline_acc"]
        print(f"L{L:<4d}{b:>8.3f}{s['probe']['pool_acc']:>12.3f}", end="")
        for eps in margins:
            print(f"{s['confidence'][f'eps={eps}']['pool_acc']:>14.3f}", end="")
        print()

    print("\n=== Δacc (gate − baseline) ===")
    print(f"{'L':<5s}{'probe':>10s}", end="")
    for eps in margins: print(f"{'conf ε='+str(eps):>14s}", end="")
    print()
    for L in sorted(out_summary):
        s = out_summary[L]
        print(f"L{L:<4d}{s['probe']['delta']:>+10.3f}", end="")
        for eps in margins:
            print(f"{s['confidence'][f'eps={eps}']['delta']:>+14.3f}", end="")
        print()

    # routing breakdown at best (L, eps)
    best = None
    for L in out_summary:
        for eps in margins:
            d = out_summary[L]["confidence"][f"eps={eps}"]
            if best is None or d["delta"] > best[2]:
                best = (L, eps, d["delta"], d)
    print(f"\n=== Best confidence-gate operating point: L{best[0]}, ε={best[1]} ===")
    d = best[3]
    print(f"  pool Δacc = {d['delta']:+.3f} ({d['baseline_acc']:.3f} → {d['pool_acc']:.3f})")
    print(f"  routing: V-patch={d['n_route_V']}  T-patch={d['n_route_T']}  none={d['n_route_none']}  (total {d['n']})")
    print(f"\n  per-subpool:")
    print(f"  {'subpool':<14s}{'n':>5s}{'baseline':>10s}{'gated':>10s}")
    for sp in SUBPOOLS:
        c = d["per_subpool"].get(sp)
        if not c: continue
        print(f"  {sp:<12s}{c['n']:>5d}{c['base']/c['n']:>10.3f}{c['gate']/c['n']:>10.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_summary, indent=2,
                                          default=lambda o: dict(o) if hasattr(o, 'keys') else o))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
