"""Step 13b: per-item SAE-decoded Δ injection ("SAE-mediated patching").

For each pool item, look up the stored h_V(i, L) from the counterfactual file.
At inference:
  z_VT = SAE.encode(h_VT(i, L))
  z_V  = SAE.encode(h_V(i, L))      [precomputed; no second forward pass]
  δ    = SAE.decode(z_V) - SAE.decode(z_VT)
  h_VT(i, L) ← h_VT(i, L) + α · δ

α=1 is "full SAE-projected patching" — δ is the SAE's reconstruction of the
caption-removal counterfactual for THIS item. α<1 lets us titrate between
no intervention and full SAE patching.

Compared to:
  - 13_sae_steering.py multiplicative hook: per-item, no fixed atom selection,
    uses the full SAE encoding instead of top-K atom multiplication.
  - 16_patching_diagnostic.py: same input (h_V) but BYPASSES the SAE; this
    script's δ is constrained to the SAE manifold.

Output:
  data/steering/qwen_multidomain_v1/sae_per_item_delta_L<L>/results.jsonl
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
from sae_steering.sae import TopKSAE, SAEConfig
from sae_steering.steering import (
    SteeringHook, baseline_rows_needed, fix_negative_args,
    get_letter_token_ids, load_distraction_pool, load_oracle_passed,
    score_row, select_pool,
)


def load_sae(data_root, variant, layer, device):
    ckpt = torch.load(data_root / f"sae/qwen/{variant}/layer{layer}/sae.pt",
                      map_location=device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ckpt["d_model"], d_sae=ckpt["d_sae"], k=ckpt["k"]))
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device).to(torch.float32).eval()
    return sae


class SAEPerItemDeltaHook(SteeringHook):
    """Inject α · (SAE.decode(z_V) - SAE.decode(z_VT)) at last_idx.

    The h_V activation is provided per-item via set_target() before forward.
    z_V is precomputed at construction (or set_target) for efficiency."""
    def __init__(self, sae, alpha: float, broadcast: bool = False):
        super().__init__(broadcast=broadcast)
        self.sae = sae
        self.alpha = alpha
        self.h_V_target: torch.Tensor | None = None
        self.delta_norms: list[float] = []

    def set_target(self, h_V_at_L: torch.Tensor):
        self.h_V_target = h_V_at_L  # [d_model]

    def _compute_delta(self, h_at_last: torch.Tensor) -> torch.Tensor:
        x_VT = h_at_last.to(torch.float32)
        x_V  = self.h_V_target.unsqueeze(0).to(torch.float32).to(x_VT.device)
        with torch.no_grad():
            z_VT = self.sae(x_VT)["z"]
            z_V  = self.sae(x_V)["z"]
            delta = self.alpha * (self.sae.decode(z_V) - self.sae.decode(z_VT))
        self.delta_norms.append(float(delta.norm().item()))
        return delta.to(h_at_last.dtype)

    def edit(self, h_at_last):
        if self.alpha == 0.0 or self.h_V_target is None:
            return None
        return h_at_last + self._compute_delta(h_at_last)

    def __call__(self, module, inputs, output):
        if not self.broadcast:
            return super().__call__(module, inputs, output)
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.last_idx is None or self.alpha == 0.0 or self.h_V_target is None:
            return output
        delta = self._compute_delta(h[:, self.last_idx, :])  # [B, D]
        h2 = h + delta.unsqueeze(1).to(h.dtype)
        return (h2,) + output[1:] if is_tuple else h2


def main():
    fix_negative_args()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--alphas", default="0.1,0.25,0.5,0.75,1.0,1.5")
    ap.add_argument("--broadcast", action="store_true", default=True)
    ap.add_argument("--no-broadcast", dest="broadcast", action="store_false")
    ap.add_argument("--distraction-pool",
                    default="data/dm/multidomain_v1/distraction_pool.jsonl")
    ap.add_argument("--baseline-source",
                    default="data/steering/qwen_multidomain_v1/probe_unseen/results.jsonl")
    ap.add_argument("--skip-baseline", action="store_true", default=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])

    # Load counterfactual activations (we need h_V at the chosen layer per cid)
    act_dir = Path(args.act_dir)
    acts = np.load(act_dir / "activations.npy", mmap_mode="r")  # (N, 3, n_layers, d)
    cf_index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_cf = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    print(f"counterfactual store: {acts.shape}; using h_V at L{args.layer}")

    # Pool
    all_rows = load_oracle_passed(data_root, baseline_rows_needed("unseen"), dm_dir=dm_dir)
    if args.distraction_pool:
        all_rows = all_rows + load_distraction_pool(Path(args.distraction_pool))
    baseline_preds = {}
    for line in Path(args.baseline_source).read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["layer"] == -1: baseline_preds[r["candidate_id"]] = r["pred"]
    pool = select_pool(all_rows, baseline_preds, "unseen")
    seen = set(); pool_dedup = []
    for r in pool:
        if r["candidate_id"] in seen: continue
        seen.add(r["candidate_id"]); pool_dedup.append(r)
    pool = [r for r in pool_dedup if r["candidate_id"] in cid_to_cf]
    print(f"pool (deduped, with h_V available): {len(pool)}")

    # Output dir
    out_dir = Path(args.out_dir) if args.out_dir else \
              data_root / "steering" / "qwen_multidomain_v1" / f"sae_per_item_delta_L{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            done.add((r["candidate_id"], r["layer"], r["alpha"]))
        print(f"resume: {len(done)} entries done")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids = get_letter_token_ids(processor, args.device)
    sae = load_sae(data_root, args.variant, args.layer, args.device)

    alphas = [float(x) for x in args.alphas.split(",")]
    out_f = out_path.open("a")

    # Optionally write baseline rows from baseline_source
    if not args.skip_baseline:
        for row in pool:
            cid = row["candidate_id"]
            if (cid, -1, 0.0) in done: continue
            bp = baseline_preds.get(cid)
            if bp is None: continue
            out_f.write(json.dumps({
                "candidate_id": cid, "label": row["label"],
                "correct_index": row["correct_index"], "split": row["__split"],
                "layer": -1, "alpha": 0.0, "pred": bp,
            }) + "\n")
            done.add((cid, -1, 0.0))
        out_f.flush()

    todo = [(row, a) for row in pool for a in alphas
            if (row["candidate_id"], args.layer, a) not in done]
    print(f"todo: {len(todo)} forward passes (L={args.layer})")

    pbar = tqdm(total=len(todo), desc="sae-perit")
    for row, a in todo:
        cid = row["candidate_id"]
        h_V = torch.from_numpy(acts[cid_to_cf[cid], 1, args.layer].astype(np.float32))
        hook = SAEPerItemDeltaHook(sae, a, broadcast=args.broadcast)
        hook.set_target(h_V)
        pred = score_row(model, processor, row, hook, args.layer, letter_ids)
        out_f.write(json.dumps({
            "candidate_id": cid, "label": row["label"],
            "correct_index": row["correct_index"], "split": row["__split"],
            "layer": args.layer, "alpha": a, "pred": pred,
        }) + "\n")
        out_f.flush()
        pbar.update(1)
    pbar.close()
    out_f.close()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
