"""Per-domain breakdown of steering results on multidomain D_M unseen pool.

For each (method, variant, layer, alpha), compute per-source × per-cell accuracy
deltas. Then for the headline operating point (bcast+normmatch, α=+0.5) and per-
domain optima, report a breakdown by source (DCI / VisText / SemArt / ROCO).
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

ROOT = ROOT / "data/steering"
DM_ALL = ROOT / "data/dm/multidomain_v1/dm_all.jsonl"

DIRS = {
  ("probe", "plain"):           ROOT/"qwen_multidomain_v1"/"probe_unseen",
  ("probe", "bcast"):           ROOT/"qwen_multidomain_v1"/"probe_unseen_bcast",
  ("probe", "normmatch"):       ROOT/"qwen_multidomain_v1"/"probe_unseen_normmatch",
  ("probe", "bcast+normmatch"): ROOT/"qwen_multidomain_v1"/"probe_unseen_bcast_normmatch",
  ("contrastive", "plain"):           ROOT/"qwen_multidomain_v1"/"contrastive_unseen",
  ("contrastive", "bcast"):           ROOT/"qwen_multidomain_v1"/"contrastive_unseen_bcast",
  ("contrastive", "normmatch"):       ROOT/"qwen_multidomain_v1"/"contrastive_unseen_normmatch",
  ("contrastive", "bcast+normmatch"): ROOT/"qwen_multidomain_v1"/"contrastive_unseen_bcast_normmatch",
  ("SAE", "plain"):           ROOT/"qwen"/"diversified_v1"/"sae_unseen",
  ("SAE", "bcast"):           ROOT/"qwen"/"diversified_v1"/"sae_unseen_bcast",
  ("SAE", "normmatch"):       ROOT/"qwen"/"diversified_v1"/"sae_unseen_normmatch",
  ("SAE", "bcast+normmatch"): ROOT/"qwen"/"diversified_v1"/"sae_unseen_bcast_normmatch",
}


def load_source_map():
    src = {}
    for line in DM_ALL.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            src[r["candidate_id"]] = r["source"]
    return src


def parse_results(d: Path, src_by_cid: dict) -> dict:
    """Parse results.jsonl into per-(L, alpha, source, label, baseline_outcome) accuracy.

    Returns dict[(L, alpha)] -> dict[(source, label, baseline_outcome)] = {"n": int, "acc": float, "delta_acc": float}
    baseline_outcome ∈ {"pass","fail"} based on layer=-1 alpha=0 row per candidate
    """
    # Read all rows; collect baseline (layer=-1, alpha=0) per candidate
    rows = []
    for line in (d / "results.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    base = {}  # cid -> bool correct
    for r in rows:
        if r["layer"] == -1 and r["alpha"] == 0.0:
            base[r["candidate_id"]] = (r["pred"] == r["correct_index"])
    # Per-(L, alpha, source, label, base_pass/fail) → list of correct flags
    by_key = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["layer"] == -1: continue  # skip baseline rows
        cid = r["candidate_id"]
        if cid not in base: continue
        src = src_by_cid.get(cid, "?")
        label = r["label"]
        base_outcome = "pass" if base[cid] else "fail"
        L, a = int(r["layer"]), float(r["alpha"])
        correct = (r["pred"] == r["correct_index"])
        by_key[(L, a)][(src, label, base_outcome)].append(correct)
    # Compute acc + delta
    agg = {}
    for (L, a), cells in by_key.items():
        out = {}
        for key, lst in cells.items():
            n = len(lst)
            acc = sum(lst) / n if n else 0.0
            # baseline acc for V/pass = 1.0, V/fail = 0.0, T/pass = 1.0, T/fail = 0.0
            src_, label_, base_ = key
            base_acc = 1.0 if base_ == "pass" else 0.0
            out[key] = {"n": n, "acc": acc, "delta_acc": acc - base_acc}
        agg[(L, a)] = out
    return agg


def compute_per_domain_net(agg_la: dict) -> dict:
    """For a given (L,α), compute per-domain net Δ accuracy across V/fail + V/pass + T/pass + T/fail.

    Returns dict[source] = {"n": int, "net_delta": float, "Vf": delta, "Vp": delta, "Tp": delta, "n_Vf": ..., ...}
    """
    per_src = defaultdict(lambda: {"correct": 0, "n": 0, "by_cell": defaultdict(lambda: {"correct": 0, "n": 0})})
    # We re-aggregate from per (src, label, base) cells
    for (src, label, base_), c in agg_la.items():
        n = c["n"]; n_correct = int(round(c["acc"] * n))
        baseline_correct = n if base_ == "pass" else 0
        per_src[src]["correct"] += n_correct
        per_src[src]["n"] += n
        per_src[src]["by_cell"][(label, base_)]["correct"] = n_correct
        per_src[src]["by_cell"][(label, base_)]["n"] = n
        per_src[src]["by_cell"][(label, base_)]["baseline_correct"] = baseline_correct
    # Compute net delta per source
    out = {}
    for src, d in per_src.items():
        baseline_pass = sum(c["baseline_correct"] for c in d["by_cell"].values())
        net_delta = (d["correct"] - baseline_pass) / max(d["n"], 1)
        cells_d = {}
        for (label, base_), c in d["by_cell"].items():
            cells_d[f"{label[0].upper()}{base_[0]}"] = (c["correct"] - c["baseline_correct"]) / max(c["n"], 1)
            cells_d[f"n_{label[0].upper()}{base_[0]}"] = c["n"]
        out[src] = {"n": d["n"], "net_delta": net_delta, **cells_d}
    return out


def main():
    src_by_cid = load_source_map()

    # Pick the 3 headline operating points from the 12-cell analysis
    op_points = [
        ("contrastive", "bcast+normmatch", 20, 0.5),
        ("SAE",         "bcast+normmatch", 13, 0.5),
        ("probe",       "plain",           13, 150.0),
        ("probe",       "bcast+normmatch", 20, 0.5),
        ("SAE",         "bcast+normmatch", 20, 0.5),
    ]

    print(f"{'method':<13s}{'variant':<19s}{'(L,α)':<14s}{'source':<10s}{'n':>5s}{'V/fail':>8s}{'V/pass':>8s}{'T/pass':>8s}{'net':>8s}")
    print("-" * 95)
    for method, variant, L, a in op_points:
        d = DIRS[(method, variant)]
        agg = parse_results(d, src_by_cid)
        if (L, a) not in agg:
            print(f"missing op: {method}/{variant} L{L} α={a}")
            continue
        # Get per-domain breakdown
        per_src = compute_per_domain_net(agg[(L, a)])
        # Also aggregate overall
        all_correct = sum(c["correct"] for c in per_src.values()) if False else None  # placeholder
        first = True
        for src in ("dci", "vistext", "semart", "roco"):
            d_ = per_src.get(src)
            if not d_: continue
            tag_m = method if first else ""
            tag_v = variant if first else ""
            tag_la = f"L{L}/α={a:+g}" if first else ""
            vf = d_.get("Vf", 0); vp = d_.get("Vp", 0); tp = d_.get("Tp", 0)
            print(f"{tag_m:<13s}{tag_v:<19s}{tag_la:<14s}{src:<10s}{d_['n']:>5d}{vf:+8.3f}{vp:+8.3f}{tp:+8.3f}{d_['net_delta']:+8.3f}")
            first = False
        # Also compute overall
        n_total = sum(d_["n"] for d_ in per_src.values())
        # weighted net
        net_overall = sum(d_["net_delta"] * d_["n"] for d_ in per_src.values()) / n_total
        print(f"{'':<13s}{'':<19s}{'':<14s}{'TOTAL':<10s}{n_total:>5d}{'':>8s}{'':>8s}{'':>8s}{net_overall:+8.3f}")
        print()


if __name__ == "__main__":
    main()
