"""Step 18: canonical-setup eval for the post-pivot headline methods.

Re-runs the headline methods on the 948-cid patching pool using 06c's
forward-pass setup (build_messages with max_pixels=1024*1024). Baseline is
read from cf-store letter_logits.npy (also produced by 06c), so baseline
and steered preds come from the same forward-pass setup → no cross-script
drift.

Methods (selectable via --methods):
  baseline           verifies cf-store baseline matches a fresh no-hook run
  additive_v_causal  L28, αV=+1, bcast+normmatch (using saved v_V_L28)
  probe_gated_patch  L31, V-patch when probe@L20 predicts vision
  random_matched     random unit vec, L28, αV=+1, bcast+normmatch (seed 0)
  random_best        random unit vec, L13, α=+3, normmatch (seed 0)

Output: data/diagnostics/canonical_eval/<method>.jsonl  (cid, pred)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sae_steering.canonical_eval import (
    SetActivationHook,
    load_canonical_pool,
    load_pool_rows_jsonl,
    load_cf_activations,
    run_canonical_forward as run_forward,
)
from sae_steering.models import load_qwen_vl
from sae_steering.steering import AdditiveHook, get_letter_token_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir",
                    default="dm_multidomain_v1_counterfactual",
                    help="cf-store subdir under data/activations/qwen/")
    ap.add_argument("--act-dir", default=None,
                    help="optional override for cf-store dir (otherwise derived from --cf-subdir)")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--methods", default="all",
                    help="comma-separated subset of: baseline, additive_v_causal, "
                         "probe_gated_patch, random_matched, random_best. "
                         "Or 'all'.")
    ap.add_argument("--out-dir", default="data/diagnostics/canonical_eval")
    ap.add_argument("--pool", default="unseen",
                    help="pool spec (unseen/test/composite) OR, with --rows-jsonl, "
                         "just the output-file suffix label")
    ap.add_argument("--rows-jsonl", default=None,
                    help="load the eval pool directly from this rows JSONL "
                         "(bypasses multidomain split structure; for assembled/cross-domain sets)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steering-vectors",
                    default="data/steering/canonical/v_causal_vectors.npz")
    ap.add_argument("--probe-layer", type=int, default=20)
    ap.add_argument("--probes-dir", default="data/probes/qwen/multidomain_v1")
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # ----- Pool -----
    if args.rows_jsonl is not None:
        pool = load_pool_rows_jsonl(Path(args.rows_jsonl), data_root, args.cf_subdir)
        print(f"pool (rows-jsonl {args.rows_jsonl}, tag={args.pool}): {len(pool)} cids")
    else:
        pool = load_canonical_pool(data_root, dm_dir, Path(args.distraction_pool),
                                    pool=args.pool, cf_subdir=args.cf_subdir)
        print(f"pool ({args.pool}): {len(pool)} cids")

    # ----- Load cf-store activations (used for h_V patching and probe input) -----
    activations, cid_to_cf = load_cf_activations(data_root, cf_subdir=args.cf_subdir)

    # ----- Load probe weights -----
    probe_npz = np.load(Path(args.probes_dir) / f"probe_layer{args.probe_layer:02d}.npz")
    probe_coef = torch.tensor(probe_npz["coef"][0], device=args.device, dtype=torch.float32)
    probe_inter = float(probe_npz["intercept"][0])

    # ----- Load v_V_causal at L28 -----
    sv = np.load(args.steering_vectors)
    v_V_L28 = torch.tensor(sv["v_V_L28"], device=args.device, dtype=torch.float32)

    # ----- Random vectors (seed 0) -----
    rng = np.random.default_rng(args.random_seed)
    def random_unit(d, seed_offset=0):
        rng_local = np.random.default_rng(args.random_seed + seed_offset)
        v = rng_local.standard_normal(d).astype(np.float32)
        v /= np.linalg.norm(v)
        return torch.tensor(v, device=args.device, dtype=torch.float32)
    v_rand_matched = random_unit(2048, seed_offset=0)
    v_rand_best = random_unit(2048, seed_offset=1)

    # Decide methods
    all_methods = ["baseline", "additive_v_causal", "probe_gated_patch",
                   "random_matched", "random_best"]
    methods = all_methods if args.methods == "all" else args.methods.split(",")
    methods = [m.strip() for m in methods]
    print(f"Methods to run: {methods}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load model -----
    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    # Pre-compute probe predictions from cf-store h_VT(L20)
    probe_pred = {}
    coef_np = probe_npz["coef"][0].astype(np.float32)
    for r in pool:
        cid = r["candidate_id"]
        i = cid_to_cf[cid]
        h = activations[i, 0, args.probe_layer].astype(np.float32)
        s = float(np.dot(coef_np, h) + probe_inter)
        probe_pred[cid] = "text" if s > 0 else "vision"

    # Also pre-compute cf-store baseline pred from letter_logits
    from pathlib import Path as _P
    cf_dir = _P("data/activations/qwen") / args.cf_subdir
    letter_logits = np.load(cf_dir / "letter_logits.npy")
    cfstore_baseline = {}
    for r in pool:
        cid = r["candidate_id"]; i = cid_to_cf[cid]
        cfstore_baseline[cid] = int(letter_logits[i, 0].argmax())

    # ----- Run each method -----
    for method in methods:
        suffix = "" if args.pool == "unseen" else f"_{args.pool}"
        out_path = out_dir / f"{method}{suffix}.jsonl"
        done = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["candidate_id"])
            print(f"{method}: resume — {len(done)} already done")
        f = out_path.open("a")

        # Build hook/config per method
        if method == "baseline":
            hook = None; layer_idx = None
        elif method == "additive_v_causal":
            hook = AdditiveHook(v_V_L28, alpha=1.0, broadcast=True, norm_match=True)
            layer_idx = 28
        elif method == "random_matched":
            hook = AdditiveHook(v_rand_matched, alpha=1.0, broadcast=True, norm_match=True)
            layer_idx = 28
        elif method == "random_best":
            hook = AdditiveHook(v_rand_best, alpha=3.0, broadcast=False, norm_match=True)
            layer_idx = 13
        elif method == "probe_gated_patch":
            hook = None; layer_idx = 31  # handled specially below
        else:
            raise ValueError(method)

        for row in tqdm(pool, desc=method):
            cid = row["candidate_id"]
            if cid in done: continue

            if method == "probe_gated_patch":
                if probe_pred[cid] == "text":
                    # T-routed: no hook, equivalent to baseline forward
                    pred = run_forward(model, processor, row, letter_ids)
                else:
                    # V-routed: SetActivationHook with h_V(L31)
                    h_V = activations[cid_to_cf[cid], 1, 31].astype(np.float32)
                    target = torch.tensor(h_V, device=args.device, dtype=torch.float32)
                    h = SetActivationHook(target)
                    pred = run_forward(model, processor, row, letter_ids,
                                       hook=h, layer_idx=31)
            else:
                pred = run_forward(model, processor, row, letter_ids,
                                   hook=hook, layer_idx=layer_idx)

            f.write(json.dumps({
                "candidate_id": cid,
                "label": row["label"],
                "correct_index": row["correct_index"],
                "pred": pred,
                "probe_pred": probe_pred[cid],
                "cfstore_baseline": cfstore_baseline[cid],
            }) + "\n")
            f.flush()
        f.close()
        print(f"{method}: wrote {out_path}")


if __name__ == "__main__":
    main()
