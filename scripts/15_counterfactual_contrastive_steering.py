"""Step 15: two-vector counterfactual contrastive steering.

Build per-modality causal steering directions from the V/V+T/T-only paired
activations collected by 06c, then sweep (alpha_V, alpha_T) at each layer.

Directions (under the existing +alpha = vision sign convention):
  v_V_causal(L) = mean_i [ h_V(i, L) - h_VT(i, L) ]  over V-grounded train_pass items
  v_T_causal(L) = mean_i [ h_T(i, L) - h_VT(i, L) ]  over T-grounded train_pass items
Each unit-normed. Applied via AdditiveTwoHook.

Sweep design (compact: 48 cells total):
  - V-only push:   alpha_V in alpha_grid, alpha_T = 0
  - T-only push:   alpha_V = 0,           alpha_T in alpha_grid
  - Both:          alpha_V = alpha_T in alpha_grid (diagonal)

Outputs go to data/steering/qwen_multidomain_v1/cfcontrastive_<variant>/
with results.jsonl carrying (alpha_v, alpha_t) instead of a single alpha.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sae_steering.models import load_qwen_vl
from sae_steering.steering import (
    POOL_CHOICES, AdditiveTwoHook, baseline_rows_needed,
    fix_negative_args, get_letter_token_ids, load_distraction_pool,
    load_oracle_passed, score_row, select_pool,
)


def compute_causal_vectors(act_dir: Path, layers, device, splits=("train_pass",)):
    """Return v_V_causal, v_T_causal at each layer (unit-normed)."""
    acts = np.load(act_dir / "activations.npy")  # (N, 3, n_layers, d) — [VT, V, T]
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    assert acts.shape[0] == len(index)
    n_total = len(index)
    keep_v = np.array([i for i, r in enumerate(index)
                       if r["split"] in splits and r["label"] == "vision"])
    keep_t = np.array([i for i, r in enumerate(index)
                       if r["split"] in splits and r["label"] == "text"])
    print(f"counterfactual source rows: V={len(keep_v)} T={len(keep_t)} (splits={splits})")
    v_V, v_T = {}, {}
    for L in layers:
        # h_V - h_VT  over V-grounded train_pass items  -> push toward V
        h_VT_V = acts[keep_v, 0, L].astype(np.float32)
        h_V_V  = acts[keep_v, 1, L].astype(np.float32)
        delta_V = (h_V_V - h_VT_V).mean(axis=0)
        nV = float(np.linalg.norm(delta_V))
        # h_T - h_VT  over T-grounded items  -> push toward T
        h_VT_T = acts[keep_t, 0, L].astype(np.float32)
        h_T_T  = acts[keep_t, 2, L].astype(np.float32)
        delta_T = (h_T_T - h_VT_T).mean(axis=0)
        nT = float(np.linalg.norm(delta_T))
        cos = float(np.dot(delta_V, delta_T) / (nV * nT))
        print(f"  L{L}: ||v_V_causal||={nV:.3f}  ||v_T_causal||={nT:.3f}  cos(v_V,v_T)={cos:+.3f}")
        v_V[L] = torch.from_numpy(delta_V / nV).to(device).to(torch.float32)
        v_T[L] = torch.from_numpy(delta_T / nT).to(device).to(torch.float32)
    return v_V, v_T


def build_sweep(alpha_grid: list[float]) -> list[tuple[float, float]]:
    """Return ordered list of (alpha_V, alpha_T) operating points."""
    points: list[tuple[float, float]] = []
    # V-only push (alpha_T = 0)
    for aV in alpha_grid:
        if aV == 0: continue
        points.append((aV, 0.0))
    # T-only push (alpha_V = 0)
    for aT in alpha_grid:
        if aT == 0: continue
        points.append((0.0, aT))
    # diagonal (both at once)
    for a in alpha_grid:
        if a == 0: continue
        points.append((a, a))
    return points


def load_resume_state_2d(out_path: Path) -> tuple[set, dict[str, int]]:
    """Like load_resume_state but with (alpha_v, alpha_t) keys."""
    done: set = set()
    baseline_preds: dict[str, int] = {}
    if not out_path.exists():
        return done, baseline_preds
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        L = r["layer"]
        if L == -1:
            baseline_preds[r["candidate_id"]] = r["pred"]
            done.add((r["candidate_id"], -1, 0.0, 0.0))
        else:
            done.add((r["candidate_id"], L, r["alpha_v"], r["alpha_t"]))
    return done, baseline_preds


def main():
    fix_negative_args(flags=("--alphas",))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--layers", default="13,20,23,28")
    ap.add_argument("--alphas", default="-3,-1,-0.5,-0.2,0.2,0.5,1,3",
                    help="grid used in (V-only, T-only, diagonal) sweeps under norm-match")
    ap.add_argument("--pool", choices=POOL_CHOICES, default="unseen")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-suffix", default="bcast_normmatch",
                    help="appended to default out-dir for clarity")
    ap.add_argument("--broadcast", action="store_true", default=True)
    ap.add_argument("--norm-match", action="store_true", default=True)
    ap.add_argument("--no-broadcast", dest="broadcast", action="store_false")
    ap.add_argument("--no-norm-match", dest="norm_match", action="store_false")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--baseline-source", default=None,
                    help="path to an existing results.jsonl with baseline rows (layer=-1) for "
                         "the same pool; we'll copy them into this run's results.jsonl instead "
                         "of re-running baseline. Useful since baselines are deterministic.")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the baseline phase entirely (assumes baseline_preds is "
                         "already populated from resume state or --baseline-source).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    layers = [int(x) for x in args.layers.split(",")]
    alpha_grid = [float(x) for x in args.alphas.split(",")]

    all_rows = load_oracle_passed(data_root, baseline_rows_needed(args.pool), dm_dir=dm_dir)
    print(f"loaded {len(all_rows)} oracle-passed items")
    if args.distraction_pool:
        dist_rows = load_distraction_pool(Path(args.distraction_pool))
        print(f"loaded {len(dist_rows)} distraction items")
        all_rows = all_rows + dist_rows

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = data_root / "steering" / "qwen_multidomain_v1" / f"cfcontrastive_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done, baseline_preds = load_resume_state_2d(out_path)
    if done:
        print(f"resume: {len(done)} records present ({len(baseline_preds)} baselines)")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    v_V, v_T = compute_causal_vectors(Path(args.act_dir), layers, args.device)
    np.savez(out_dir / "steering_vectors.npz",
             **{f"v_V_L{L:02d}": v_V[L].cpu().numpy() for L in layers},
             **{f"v_T_L{L:02d}": v_T[L].cpu().numpy() for L in layers})

    out_f = out_path.open("a")

    # --baseline-source: bulk-copy baseline rows from another completed run
    if args.baseline_source and not args.skip_baseline:
        src_lines = Path(args.baseline_source).read_text().splitlines()
        src_base = []
        for line in src_lines:
            if not line.strip(): continue
            r = json.loads(line)
            if r["layer"] != -1: continue
            if (r["candidate_id"], -1, 0.0, 0.0) in done: continue
            new = {**r, "alpha_v": 0.0, "alpha_t": 0.0}
            new.pop("alpha", None)
            out_f.write(json.dumps(new) + "\n")
            baseline_preds[r["candidate_id"]] = r["pred"]
            done.add((r["candidate_id"], -1, 0.0, 0.0))
            src_base.append(r["candidate_id"])
        out_f.flush()
        print(f"copied {len(src_base)} baselines from {args.baseline_source}")

    # inline baseline phase (skipped if --skip-baseline or fully resumed)
    todo_base = [] if args.skip_baseline else [
        r for r in all_rows if (r["candidate_id"], -1, 0.0, 0.0) not in done]
    print(f"baseline phase: {len(todo_base)} forward passes")
    pbar = tqdm(total=len(todo_base), desc="baseline")
    for row in todo_base:
        pred = score_row(model, processor, row, None, None, letter_ids)
        out_f.write(json.dumps({
            "candidate_id": row["candidate_id"], "label": row["label"],
            "correct_index": row["correct_index"], "split": row["__split"],
            "layer": -1, "alpha_v": 0.0, "alpha_t": 0.0, "pred": pred,
        }) + "\n")
        out_f.flush()
        baseline_preds[row["candidate_id"]] = pred
        pbar.update(1)
    pbar.close()

    pool = select_pool(all_rows, baseline_preds, args.pool)
    n_v = sum(1 for r in pool if r["label"] == "vision")
    n_t = sum(1 for r in pool if r["label"] == "text")
    print(f"steering pool ({args.pool}): {len(pool)} items  V={n_v} T={n_t}")

    sweep = build_sweep(alpha_grid)
    print(f"sweep: {len(sweep)} (aV, aT) points x {len(layers)} layers x {len(pool)} pool "
          f"= {len(sweep)*len(layers)*len(pool)} forward passes")

    todo = []
    for row in pool:
        for L in layers:
            for aV, aT in sweep:
                if (row["candidate_id"], L, aV, aT) not in done:
                    todo.append((row, L, aV, aT))
    pbar = tqdm(total=len(todo), desc="cf-contrastive")
    for row, L, aV, aT in todo:
        hook = AdditiveTwoHook(v_V[L], v_T[L], aV, aT,
                               broadcast=args.broadcast, norm_match=args.norm_match)
        pred = score_row(model, processor, row, hook, L, letter_ids)
        out_f.write(json.dumps({
            "candidate_id": row["candidate_id"], "label": row["label"],
            "correct_index": row["correct_index"], "split": row["__split"],
            "layer": L, "alpha_v": aV, "alpha_t": aT, "pred": pred,
        }) + "\n")
        out_f.flush()
        pbar.update(1)
    pbar.close()
    out_f.close()

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
