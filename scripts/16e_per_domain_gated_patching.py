"""Step 16e: per-domain breakdown of the gated patching result.

The +10.4 pp pool gain (probe-gated V-patch at L31) is averaged across
DCI / VisText / SemArt / ROCO. Worth checking if any domain drives most of
the gain or any is destructive — a domain-specific destruction would
qualify the headline claim.

Output: stdout table.
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
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--probes-dir", default="data/probes/qwen/multidomain_v1")
    ap.add_argument("--probe-layer", type=int, default=20)
    ap.add_argument("--patch-layer", type=int, default=31)
    ap.add_argument("--dm-all", default="data/dm/multidomain_v1/dm_all.jsonl")
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    logits = np.load(act_dir / "letter_logits.npy")
    cf_index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    correct_by_cid = {r["candidate_id"]: r["correct_index"] for r in cf_index}
    v_ok = {cid: bool(logits[i, 1].argmax() == correct_by_cid[cid]) for cid, i in cid_to_idx.items()}

    # source per cid from dm_all
    src_by_cid = {}
    for line in Path(args.dm_all).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        src_by_cid[r["candidate_id"]] = r.get("source", "?")

    # probe predictions
    probe = np.load(Path(args.probes_dir) / f"probe_layer{args.probe_layer:02d}.npz")
    coef = probe["coef"][0].astype(np.float32)
    intercept = float(probe["intercept"][0])
    acts = np.load(act_dir / "activations.npy")
    probe_label = {}
    for cid, i in cid_to_idx.items():
        h = acts[i, 0, args.probe_layer].astype(np.float32)
        score = float(np.dot(coef, h) + intercept)
        probe_label[cid] = "text" if score > 0 else "vision"

    # V-patch results at the chosen layer
    rows = [json.loads(l) for l in (Path(args.v_patch_dir) / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["layer"] == args.patch_layer]
    canonical = {}
    for r in rows:
        cid = r["candidate_id"]
        if cid not in canonical:
            canonical[cid] = r

    def subpool(r):
        label = r["label"]; bp = r["baseline_pred"]; ci = r["correct_index"]
        if label == "vision":
            if bp == ci: return "V/pass"
            return "V/fail∧V✓" if v_ok.get(r["candidate_id"], False) else "V/fail∧V✗"
        else:
            if bp == ci: return "T/pass"
            return "T/fail"

    per_src = defaultdict(lambda: {"n": 0, "base": 0, "gated": 0,
                                    "per_subpool": defaultdict(lambda: {"n": 0, "base": 0, "gated": 0})})
    for cid, r in canonical.items():
        src = src_by_cid.get(cid, "?")
        sp = subpool(r)
        ci = r["correct_index"]
        base = (r["baseline_pred"] == ci)
        uV = (r["patched_pred"] == ci)
        # probe-gated: apply V-patch if probe says V; else baseline
        gated = uV if probe_label[cid] == "vision" else base
        d = per_src[src]
        d["n"] += 1
        d["base"] += int(base)
        d["gated"] += int(gated)
        d["per_subpool"][sp]["n"] += 1
        d["per_subpool"][sp]["base"] += int(base)
        d["per_subpool"][sp]["gated"] += int(gated)

    print(f"=== Per-domain pool acc at L{args.patch_layer}, probe-gated V-patch ===\n")
    print(f"{'source':<10s}{'n':>5s}{'base':>8s}{'gated':>8s}{'Δ':>9s}")
    for src in sorted(per_src):
        d = per_src[src]
        if d["n"] == 0: continue
        b = d["base"] / d["n"]; g = d["gated"] / d["n"]
        print(f"{src:<10s}{d['n']:>5d}{b:>8.3f}{g:>8.3f}{g-b:>+9.3f}")
    # overall
    n = sum(d["n"] for d in per_src.values())
    b_tot = sum(d["base"] for d in per_src.values()) / n
    g_tot = sum(d["gated"] for d in per_src.values()) / n
    print(f"{'TOTAL':<10s}{n:>5d}{b_tot:>8.3f}{g_tot:>8.3f}{g_tot-b_tot:>+9.3f}")

    # Per-domain stratified breakdown
    SUBPOOLS = ["V/pass", "V/fail∧V✓", "V/fail∧V✗", "T/pass", "T/fail"]
    print(f"\n=== Stratified per-domain (gated acc) ===\n")
    print(f"{'source':<10s}", end="")
    for sp in SUBPOOLS: print(f"{sp:>14s}", end="")
    print()
    for src in sorted(per_src):
        print(f"{src:<10s}", end="")
        for sp in SUBPOOLS:
            c = per_src[src]["per_subpool"].get(sp)
            if c and c["n"] > 0:
                print(f"  {c['gated']}/{c['n']}={c['gated']/c['n']:.2f}", end="")
            else:
                print(f"          -", end="")
        print()


if __name__ == "__main__":
    main()
