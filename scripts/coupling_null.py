"""W1(b): coupling-aware null for the grounding-strength law (reviewer Q1).

Question: how much of the cross-cell correlation r(grounding, distraction) is already implied by
margin-crossing alone — i.e., by each cell's REAL margin distribution pushed through a SINGLE,
cell-independent perturbation distribution G (no caption-, cell-, or model-specific effect)?

Per vision cell (8 backbones x [4 D_M domains + assembled]): grounding g = P(m>0) (real);
simulated distraction = P(m + delta < 0 | m > 0) with delta ~ empirical pooled-G. We report the
correlation between g and the SIMULATED rates (analytic expectation + simulation CI), next to the
observed r. If they match, the law is exactly what the margin model predicts (the paper's claim);
the residual gap is what "grounding" contributes beyond mechanical coupling.

    python scripts/coupling_null.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
MODELS = ["qwen", "Qwen_Qwen2.5-VL-7B-Instruct", "Qwen_Qwen2-VL-2B-Instruct",
          "OpenGVLab_InternVL3-8B-hf", "llava-hf_llava-1.5-7b-hf",
          "llava-hf_llava-onevision-qwen2-7b-ov-hf", "llavanext"]
DOMS = ["dci", "vistext", "semart", "roco"]
VIOL = set(json.load(open(DR / "diagnostics/assembled_caption_certification.json"))["violations"])
RNG = np.random.default_rng(0)


def split_sources():
    src = {}
    for sp in ("train", "val", "test"):
        p = DR / "dm/multidomain_v1/splits" / f"{sp}.jsonl"
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); src[r["candidate_id"]] = r.get("source")
    return src


def cell_margins():
    """-> list of (cell_key, margins_all_items, deltas_on_solvable)"""
    SRC = split_sources()
    cells = {}
    for m in MODELS:
        for sub, tag in [("dm_multidomain_v1_cf", "dm"), ("merged_aokvqa_racehigh_cf", "asm")]:
            st = DR / "activations" / m / sub
            if not (st / "letter_logits.npy").exists():
                continue
            ll = np.load(st / "letter_logits.npy")
            idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
            for i, r in enumerate(idx):
                if r["label"] != "vision":
                    continue
                if tag == "asm" and r["candidate_id"] in VIOL:
                    continue
                dom = (r.get("source") or SRC.get(r["candidate_id"])) if tag == "dm" else "aokvqa"
                if tag == "dm" and dom not in DOMS:
                    continue
                ci = int(r["correct_index"])
                lv = ll[i, 1].astype(np.float64); lvt = ll[i, 0].astype(np.float64)
                mv = lv[ci] - np.delete(lv, ci).max()
                key = (m, dom)
                cells.setdefault(key, {"m": [], "d": []})
                cells[key]["m"].append(mv)
                if mv > 0:
                    mvt = lvt[ci] - np.delete(lvt, ci).max()
                    cells[key]["d"].append(mvt - mv)
    return cells


def main() -> None:
    cells = cell_margins()
    print(f"vision cells: {len(cells)}")
    # pooled, cell-independent perturbation distribution (z-scored? no: raw pooled deltas)
    all_d = np.concatenate([np.array(c["d"]) for c in cells.values()])
    print(f"pooled deltas: n={len(all_d)}  mean={all_d.mean():.2f}")
    g, obs, sim = [], [], []
    for key, c in cells.items():
        marg = np.array(c["m"]); pos = marg[marg > 0]
        if len(pos) < 20:
            continue
        g.append((marg > 0).mean())
        obs.append(np.mean(np.array(c["d"]) + pos[: len(c["d"])] < 0) if False else
                   np.mean((np.array(c["d"]) + pos[np.arange(len(c["d"])) % len(pos)]) < 0))
        # analytic expectation under pooled G: mean_i G(-m_i)
        Gsort = np.sort(all_d)
        sim.append(float(np.searchsorted(Gsort, -pos, side="right").mean() / len(Gsort)))
    g, sim = np.array(g), np.array(sim)
    # observed distraction rates: real flips (mvt<0) per cell
    obs2 = []
    for key, c in cells.items():
        marg = np.array(c["m"]); pos = marg[marg > 0]
        if len(pos) < 20:
            continue
        d = np.array(c["d"])
        obs2.append(float((d + pos[: len(d)] < 0).mean()) if len(d) == len(pos) else
                    float(np.mean(np.array(c["d"]) < -pos[: len(c["d"])])))
    obs2 = np.array(obs2)
    r_obs = np.corrcoef(g, obs2)[0, 1]
    r_sim = np.corrcoef(g, sim)[0, 1]
    # simulation CI (resample deltas per cell)
    rs = []
    Gpool = all_d
    for _ in range(500):
        simr = []
        for key, c in cells.items():
            marg = np.array(c["m"]); pos = marg[marg > 0]
            if len(pos) < 20:
                continue
            dd = RNG.choice(Gpool, size=len(pos))
            simr.append(float(((pos + dd) < 0).mean()))
        rs.append(np.corrcoef(g, simr)[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    print(f"\nr(grounding, OBSERVED distraction)   = {r_obs:+.3f}   (vision cells, margin-level recompute)")
    print(f"r(grounding, MARGIN-MODEL predicted)  = {r_sim:+.3f}   (single pooled G, no cell effect)")
    print(f"simulation 95% CI over draws          = [{lo:+.3f},{hi:+.3f}]")
    out = {"r_observed_vision_cells": float(r_obs), "r_margin_model": float(r_sim),
           "sim_ci": [float(lo), float(hi)], "n_cells": int(len(g))}
    (DR / "diagnostics/coupling_null.json").write_text(json.dumps(out, indent=1))
    print(f"-> {DR/'diagnostics/coupling_null.json'}")


if __name__ == "__main__":
    main()
