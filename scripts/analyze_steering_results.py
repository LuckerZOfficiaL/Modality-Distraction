"""Aggregate steering summary.json files (probe / contrastive / SAE × variants) on
multidomain D_M unseen pool. For each (method, variant, layer, alpha) compute:
  - V/fail delta_acc
  - V/pass delta_acc
  - T/pass delta_acc
  - net pool delta (weighted by cell sizes)
Then report the best operating point per (method, variant) and a comparison table.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

ROOT = ROOT / "data/steering"
DIRS = {
    ("probe",       "plain"):           ROOT / "qwen_multidomain_v1" / "probe_unseen",
    ("probe",       "bcast"):           ROOT / "qwen_multidomain_v1" / "probe_unseen_bcast",
    ("probe",       "normmatch"):       ROOT / "qwen_multidomain_v1" / "probe_unseen_normmatch",
    ("probe",       "bcast+normmatch"): ROOT / "qwen_multidomain_v1" / "probe_unseen_bcast_normmatch",
    ("contrastive", "plain"):           ROOT / "qwen_multidomain_v1" / "contrastive_unseen",
    ("contrastive", "bcast"):           ROOT / "qwen_multidomain_v1" / "contrastive_unseen_bcast",
    ("contrastive", "normmatch"):       ROOT / "qwen_multidomain_v1" / "contrastive_unseen_normmatch",
    ("contrastive", "bcast+normmatch"): ROOT / "qwen_multidomain_v1" / "contrastive_unseen_bcast_normmatch",
    ("SAE",         "plain"):           ROOT / "qwen" / "diversified_v1" / "sae_unseen",
    ("SAE",         "bcast"):           ROOT / "qwen" / "diversified_v1" / "sae_unseen_bcast",
    ("SAE",         "normmatch"):       ROOT / "qwen" / "diversified_v1" / "sae_unseen_normmatch",
    ("SAE",         "bcast+normmatch"): ROOT / "qwen" / "diversified_v1" / "sae_unseen_bcast_normmatch",
}


def parse_summary(p: Path) -> dict:
    s = json.load(open(p / "summary.json"))
    bv = s["baseline|vision"]; bt = s["baseline|text"]
    n_v_pass, n_v_fail = bv["n_pass"], bv["n_fail"]
    n_t_pass, n_t_fail = bt["n_pass"], bt["n_fail"]
    n_total = n_v_pass + n_v_fail + n_t_pass + n_t_fail
    # baseline acc
    baseline_acc = (n_v_pass + n_t_pass) / n_total
    rows = []
    for k, v in s.items():
        if not "|" in k or k.startswith("baseline"):
            continue
        parts = k.split("|")
        if len(parts) < 4:
            continue
        L_str = parts[0]              # "L13"
        a_str = parts[1].split("=")[1] # "+5.0"
        label = parts[2]              # "vision"/"text"
        split = parts[3]              # "pass"/"fail"
        L = int(L_str[1:])
        a = float(a_str)
        rows.append({"L": L, "alpha": a, "label": label, "split": split,
                     "delta_acc": v["delta_acc"], "n": v["n"]})
    return {"n_v_pass": n_v_pass, "n_v_fail": n_v_fail,
            "n_t_pass": n_t_pass, "n_t_fail": n_t_fail,
            "baseline_acc": baseline_acc,
            "rows": rows}


def aggregate(rows: list, n_v_pass, n_v_fail, n_t_pass, n_t_fail) -> dict:
    """Group rows by (L, alpha); compute V/pass, V/fail, T/pass deltas + net pool delta."""
    bucket = defaultdict(dict)
    for r in rows:
        bucket[(r["L"], r["alpha"])][(r["label"], r["split"])] = r["delta_acc"]
    out = {}
    total = n_v_pass + n_v_fail + n_t_pass + n_t_fail
    for (L, a), cells in bucket.items():
        dvp = cells.get(("vision","pass"), 0.0)
        dvf = cells.get(("vision","fail"), 0.0)
        dtp = cells.get(("text","pass"), 0.0)
        dtf = cells.get(("text","fail"), 0.0)
        # weighted net pool delta (V/pass + V/fail + T/pass + T/fail) — T/fail typically 0 items, contributes 0
        net = (dvp * n_v_pass + dvf * n_v_fail + dtp * n_t_pass + dtf * n_t_fail) / total
        out[(L, a)] = {"V/fail": dvf, "V/pass": dvp, "T/pass": dtp, "T/fail": dtf, "net": net}
    return out


def main():
    methods = defaultdict(dict)  # method -> variant -> {(L,a): metrics}
    base_meta = None
    for (method, variant), d in DIRS.items():
        if not (d / "summary.json").exists():
            print(f"  missing: {d}")
            continue
        s = parse_summary(d)
        if base_meta is None:
            base_meta = s
        else:
            assert s["n_v_pass"] == base_meta["n_v_pass"]
        agg = aggregate(s["rows"], s["n_v_pass"], s["n_v_fail"],
                        s["n_t_pass"], s["n_t_fail"])
        methods[method][variant] = agg
    print(f"\npool: V/pass={base_meta['n_v_pass']}  V/fail={base_meta['n_v_fail']}  "
          f"T/pass={base_meta['n_t_pass']}  T/fail={base_meta['n_t_fail']}")
    print(f"baseline acc: {base_meta['baseline_acc']:.3f}")

    # Best operating point per (method, variant) — max net pool delta with V/pass collateral <= 0.10 cap
    print("\n=== Best operating point per (method, variant), constraint: V/pass Δ >= -0.10 ===")
    print(f"{'method':<12s} {'variant':<18s} {'(L,α)':<14s} {'V/fail':>8s} {'V/pass':>8s} {'T/pass':>8s} {'net':>8s}")
    best_global = []
    for method in ("probe", "contrastive", "SAE"):
        for variant in ("plain", "bcast", "normmatch", "bcast+normmatch"):
            if variant not in methods.get(method, {}):
                continue
            agg = methods[method][variant]
            cand = [(La, m) for La, m in agg.items() if m["V/pass"] >= -0.10]
            if not cand:
                cand = list(agg.items())
            best = max(cand, key=lambda x: x[1]["net"])
            (L, a), m = best
            best_global.append((method, variant, L, a, m))
            print(f"{method:<12s} {variant:<18s} L{L}/α={a:+g}    "
                  f"{m['V/fail']:+8.3f} {m['V/pass']:+8.3f} {m['T/pass']:+8.3f} {m['net']:+8.3f}")

    # Best operating point per layer per (method,variant) — for "fixed alpha" question
    print("\n=== Top-3 net-pool ops per (method, variant) ===")
    for method in ("probe", "contrastive", "SAE"):
        for variant in ("plain", "bcast", "normmatch", "bcast+normmatch"):
            if variant not in methods.get(method, {}):
                continue
            agg = methods[method][variant]
            top3 = sorted(agg.items(), key=lambda x: -x[1]["net"])[:3]
            print(f"\n  {method} / {variant}:")
            for (L, a), m in top3:
                print(f"    L{L} α={a:+g}  V/fail={m['V/fail']:+.3f} V/pass={m['V/pass']:+.3f} "
                      f"T/pass={m['T/pass']:+.3f} net={m['net']:+.3f}")

    # Save full table
    out = []
    for method, vmap in methods.items():
        for variant, agg in vmap.items():
            for (L, a), m in agg.items():
                out.append({"method": method, "variant": variant, "layer": L, "alpha": a,
                            "V_fail": m["V/fail"], "V_pass": m["V/pass"],
                            "T_pass": m["T/pass"], "T_fail": m["T/fail"], "net": m["net"]})
    out_path = ROOT / "qwen_multidomain_v1_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nfull table -> {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
