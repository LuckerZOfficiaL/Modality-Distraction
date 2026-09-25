"""Step 16: activation-patching diagnostic — does any layer causally route modality?

For each V/fail item, run the V+T forward pass but REPLACE the last-token
residual at layer L with the stored last-token residual from the V-only
forward pass on the same item. Measure flip rate to correct.

This is the gold-standard layer-localization test. Unlike steering (which
adds a delta), patching SETS the activation to a known value the model
produces under a different condition. If patching at no layer flips
V/fail items to correct, then fixed-vector additive steering at any
layer is intrinsically inadequate — the layer simply isn't where modality
routing is causally encoded for these failures.

Pool: V/fail items from the same unseen pool steering uses
(val + train V+T-fail + distraction with label=vision).

Output:
  data/diagnostics/patching_V_to_VT/results.jsonl   per (cid, layer) row
  data/diagnostics/patching_V_to_VT/summary.json    per-layer flip rate

Resume-safe via results.jsonl.
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
    SteeringHook, baseline_rows_needed, build_messages,
    get_letter_token_ids, load_distraction_pool, load_oracle_passed,
    select_pool,
)
from qwen_vl_utils import process_vision_info


class SetActivationHook(SteeringHook):
    """Override the residual at last_idx with a stored target vector."""
    def __init__(self, target: torch.Tensor):
        super().__init__(broadcast=False)
        self.target = target   # [D]

    def edit(self, h):
        # h is [B, D] (last-token slice); target is [D]. Set all rows to target.
        return self.target.to(h.dtype).expand_as(h)


@torch.inference_mode()
def score_with_patch(model, processor, row, layer_idx, target_vec, letter_token_ids):
    msgs = build_messages(row)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
    hook = SetActivationHook(target_vec) if target_vec is not None else None
    handle = None
    if hook is not None:
        hook.last_idx = last_idx
        layer = model.model.language_model.layers[layer_idx]
        handle = layer.register_forward_hook(hook)
    try:
        out = model(**inputs)
    finally:
        if handle is not None:
            handle.remove()
            hook.last_idx = None
    logits = out.logits[0, last_idx]
    return int(logits[letter_token_ids].argmax().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--baseline-source",
                    default="data/steering/qwen_multidomain_v1/probe_unseen/results.jsonl",
                    help="path to a results.jsonl with baseline V+T preds (for V/fail selection)")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--layers", default="all",
                    help="comma-separated layers to scan, or 'all' for 0..n_layers-1")
    ap.add_argument("--patch-source", choices=["V", "T"], default="V",
                    help="V: patch h_V(i,L) into V+T pass (recover image-only behavior). "
                         "T: patch h_T(i,L) into V+T pass (recover caption-only behavior).")
    ap.add_argument("--pool-filter",
                    choices=["v_fail", "v_pass", "all", "v_all", "t_all"], default="v_fail",
                    help="which subpool to patch on. v_fail/v_pass = vision items only; "
                         "v_all = all vision; t_all = all text; all = the whole pool")
    ap.add_argument("--out-dir",
                    default="data/diagnostics/patching_V_to_VT")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # Pool: same as steering's "unseen"
    all_rows = load_oracle_passed(data_root, baseline_rows_needed("unseen"), dm_dir=dm_dir)
    if args.distraction_pool:
        all_rows = all_rows + load_distraction_pool(Path(args.distraction_pool))

    # baseline preds for V+T (for V/fail selection)
    baseline_preds: dict[str, int] = {}
    for line in Path(args.baseline_source).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["layer"] == -1:
            baseline_preds[r["candidate_id"]] = r["pred"]
    print(f"loaded {len(baseline_preds)} baseline preds")

    pool = select_pool(all_rows, baseline_preds, "unseen")
    # filter to V/fail (label=vision AND baseline wrong) by default
    def is_target(row):
        bp = baseline_preds.get(row["candidate_id"])
        if bp is None: return False
        pf = args.pool_filter
        if pf == "v_fail":
            return row["label"] == "vision" and bp != row["correct_index"]
        if pf == "v_pass":
            return row["label"] == "vision" and bp == row["correct_index"]
        if pf == "v_all":
            return row["label"] == "vision"
        if pf == "t_all":
            return row["label"] == "text"
        if pf == "all":
            return True
        return False
    target_rows = [r for r in pool if is_target(r)]
    print(f"target subpool ({args.pool_filter}): {len(target_rows)} items")

    # Load counterfactual activations + index, build cid → row-index map
    acts = np.load(Path(args.act_dir) / "activations.npy")  # (N, 3, n_layers, d)
    cf_index = [json.loads(l) for l in (Path(args.act_dir) / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    n_layers = acts.shape[2]
    print(f"counterfactual acts: {acts.shape}")

    # Filter to items present in the counterfactual store
    target_rows = [r for r in target_rows if r["candidate_id"] in cid_to_idx]
    print(f"after counterfactual-store filter: {len(target_rows)} items")

    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    cond_idx = 1 if args.patch_source == "V" else 2  # [VT=0, V=1, T=2]
    print(f"scanning layers {layers}  patch_source={args.patch_source} (cond_idx={cond_idx})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"

    # resume
    done: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            done.add((r["candidate_id"], r["layer"]))
        print(f"resume: {len(done)} (cid, layer) entries already done")

    print(f"loading model on {args.device}")
    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)

    todo = [(row, L) for row in target_rows for L in layers
            if (row["candidate_id"], L) not in done]
    print(f"todo: {len(todo)} forward passes")

    out_f = out_path.open("a")
    pbar = tqdm(total=len(todo), desc="patch")
    for row, L in todo:
        cf_i = cid_to_idx[row["candidate_id"]]
        target_vec = torch.from_numpy(acts[cf_i, cond_idx, L].astype(np.float32)).to(args.device)
        pred = score_with_patch(model, processor, row, L, target_vec, letter_ids)
        out_f.write(json.dumps({
            "candidate_id": row["candidate_id"],
            "label": row["label"],
            "correct_index": row["correct_index"],
            "split": row["__split"],
            "layer": L,
            "patched_pred": pred,
            "baseline_pred": baseline_preds.get(row["candidate_id"]),
            "flipped_to_correct": pred == row["correct_index"],
        }) + "\n")
        out_f.flush()
        pbar.update(1)
    pbar.close()
    out_f.close()

    # per-layer flip rate
    from collections import defaultdict
    rows_log = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    by_layer = defaultdict(list)
    for r in rows_log:
        by_layer[r["layer"]].append(r["flipped_to_correct"])
    summary = {}
    print(f"\n=== Per-layer flip rate ({args.patch_source} → V+T, {args.pool_filter}) ===")
    print(f"{'layer':<8s}{'n':>6s}{'flips':>8s}{'rate':>8s}")
    for L in sorted(by_layer):
        flips = sum(by_layer[L])
        n = len(by_layer[L])
        rate = flips / n
        summary[L] = {"n": n, "flips": int(flips), "rate": float(rate)}
        print(f"L{L:<7d}{n:>6d}{int(flips):>8d}{rate:>8.3f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
