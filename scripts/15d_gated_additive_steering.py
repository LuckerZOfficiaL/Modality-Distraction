"""Step 15d: probe-gated additive steering with the counterfactual direction.

For each pool item, predict V vs T using the probe at L_probe applied to the
STORED h_VT(L_probe) from the counterfactual file (no forward pass needed
for the gate). Then:
  - probe says V → forward pass with additive hook at L_patch:
                   h_VT(L_patch) ← h_VT(L_patch) + α · v_V_causal(L_patch)
  - probe says T → use baseline prediction (no hook)

v_V_causal(L) = unit(mean_i[h_V(i,L) - h_VT(i,L)]) over V-grounded train_pass.
Same direction as script 15's V-only sweep, applied with the gate.

Resume-safe via per-(cid, L, alpha) key. Baseline preds read from a
specified results.jsonl (probe_unseen by default; deterministic argmax).

Output: data/steering/qwen_multidomain_v1/gated_additive_<tag>/results.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from moground.models import load_qwen_vl
from moground.steering import (
    AdditiveHook, baseline_rows_needed, fix_negative_args,
    get_letter_token_ids, load_distraction_pool, load_oracle_passed,
    score_row, select_pool,
)


def compute_v_V_causal(act_dir: Path, layers, device):
    """v_V_causal(L) = unit(mean_i[h_V - h_VT]) over V-grounded train_pass."""
    acts = np.load(act_dir / "activations.npy")  # (N, 3, n_layers, d) [VT, V, T]
    idx = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    keep = np.array([i for i, r in enumerate(idx)
                     if r["split"] == "train_pass" and r["label"] == "vision"])
    print(f"v_V_causal source: {len(keep)} vision train_pass rows")
    out = {}
    for L in layers:
        h_VT = acts[keep, 0, L].astype(np.float32)
        h_V  = acts[keep, 1, L].astype(np.float32)
        v = (h_V - h_VT).mean(axis=0)
        n = float(np.linalg.norm(v))
        out[L] = torch.from_numpy(v / n).to(device).to(torch.float32)
        print(f"  L{L}: ||raw delta|| = {n:.3f}")
    return out


def probe_predict_labels(act_dir: Path, probes_dir: Path, probe_layer: int):
    """Return dict[cid] -> 'vision' or 'text' from probe applied to stored h_VT(L_probe)."""
    acts = np.load(act_dir / "activations.npy")
    idx = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    probe = np.load(probes_dir / f"probe_layer{probe_layer:02d}.npz")
    coef = probe["coef"][0].astype(np.float32)
    intercept = float(probe["intercept"][0])
    preds = {}
    for i, r in enumerate(idx):
        h = acts[i, 0, probe_layer].astype(np.float32)
        score = float(np.dot(coef, h) + intercept)
        # sklearn: class 1 (text) when score > 0; class 0 (vision) otherwise
        preds[r["candidate_id"]] = "text" if score > 0 else "vision"
    return preds


def main():
    fix_negative_args(flags=("--alphas",))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--layers", default="27,29,31,33,35",
                    help="patch layers (additive hook injection points)")
    ap.add_argument("--alphas", default="-0.5,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.5",
                    help="norm-match alphas. v_V_causal natural magnitude is "
                         "~5-10%% of ||h||, so small alphas (0.05-0.2) are the "
                         "meaningful range under norm-match.")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--probes-dir", default="data/probes/qwen/multidomain_v1")
    ap.add_argument("--probe-layer", type=int, default=20)
    ap.add_argument("--baseline-source",
                    default="data/steering/qwen_multidomain_v1/probe_unseen/results.jsonl",
                    help="results.jsonl with baseline V+T preds (layer=-1)")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--broadcast", action="store_true", default=True)
    ap.add_argument("--no-broadcast", dest="broadcast", action="store_false")
    ap.add_argument("--norm-match", action="store_true", default=True)
    ap.add_argument("--no-norm-match", dest="norm_match", action="store_false")
    ap.add_argument("--out-dir",
                    default="data/steering/qwen_multidomain_v1/gated_additive_v_causal")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    # Build the unseen pool
    all_rows = load_oracle_passed(data_root, baseline_rows_needed("unseen"), dm_dir=dm_dir)
    if args.distraction_pool:
        all_rows = all_rows + load_distraction_pool(Path(args.distraction_pool))

    # Baseline preds (deterministic argmax; copy from probe_unseen)
    baseline_preds = {}
    for line in Path(args.baseline_source).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["layer"] == -1:
            baseline_preds[r["candidate_id"]] = r["pred"]
    print(f"loaded {len(baseline_preds)} baseline preds")

    pool = select_pool(all_rows, baseline_preds, "unseen")
    # dedup by cid to avoid double-scoring distraction collisions
    seen = set(); pool_dedup = []
    for r in pool:
        if r["candidate_id"] in seen: continue
        seen.add(r["candidate_id"]); pool_dedup.append(r)
    pool = pool_dedup
    print(f"pool: {len(pool)} unique items")

    # Probe predictions (pre-computed from stored h_VT(L_probe))
    probe_pred = probe_predict_labels(Path(args.act_dir), Path(args.probes_dir),
                                       args.probe_layer)
    n_pred_V = sum(1 for r in pool if probe_pred.get(r["candidate_id"]) == "vision")
    n_pred_T = sum(1 for r in pool if probe_pred.get(r["candidate_id"]) == "text")
    print(f"probe@L{args.probe_layer} predicts: V={n_pred_V}  T={n_pred_T}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            done.add((r["candidate_id"], r["layer"], r["alpha"]))
        print(f"resume: {len(done)} (cid, L, α) entries done")

    # Load model + compute v_V_causal at requested layers
    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    vecs = compute_v_V_causal(Path(args.act_dir), layers, args.device)
    np.savez(out_dir / "steering_vectors.npz",
             **{f"v_V_L{L:02d}": vecs[L].cpu().numpy() for L in layers})

    out_f = out_path.open("a")
    todo = []
    for row in pool:
        for L in layers:
            for a in alphas:
                if (row["candidate_id"], L, a) not in done:
                    todo.append((row, L, a))
    print(f"todo: {len(todo)} forward passes ({n_pred_V} V-items × {len(layers)*len(alphas)} cells "
          f"if no resume; T-items skipped since they use baseline)")

    # Filter todo: T-predicted items don't need a hook — write baseline pred directly
    todo_hooked = []
    n_t_written = 0
    for row, L, a in todo:
        cid = row["candidate_id"]
        if probe_pred.get(cid) == "text":
            # Write baseline pred for this cell (no hook applied at inference)
            bp = baseline_preds.get(cid)
            if bp is None: continue
            out_f.write(json.dumps({
                "candidate_id": cid, "label": row["label"],
                "correct_index": row["correct_index"], "split": row["__split"],
                "layer": L, "alpha": a, "pred": bp,
                "probe_gate": "text", "gated_applied": False,
            }) + "\n")
            n_t_written += 1
        else:
            todo_hooked.append((row, L, a))
    out_f.flush()
    print(f"T-gated items (skip hook, use baseline): {n_t_written} rows written directly")
    print(f"V-gated items (apply hook): {len(todo_hooked)} forward passes to do")

    pbar = tqdm(total=len(todo_hooked), desc="gated-add")
    for row, L, a in todo_hooked:
        hook = AdditiveHook(vecs[L], a, broadcast=args.broadcast,
                            norm_match=args.norm_match)
        pred = score_row(model, processor, row, hook, L, letter_ids)
        out_f.write(json.dumps({
            "candidate_id": row["candidate_id"], "label": row["label"],
            "correct_index": row["correct_index"], "split": row["__split"],
            "layer": L, "alpha": a, "pred": pred,
            "probe_gate": "vision", "gated_applied": True,
        }) + "\n")
        out_f.flush()
        pbar.update(1)
    pbar.close()
    out_f.close()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
