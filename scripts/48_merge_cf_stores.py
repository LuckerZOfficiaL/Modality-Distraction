"""Merge counterfactual cf-stores (concatenate along the item axis) so analyses can
run on the WHOLE dataset (no train/val/test split) per the resource-paper principle.

    python scripts/48_merge_cf_stores.py \
        --stores dm_multidomain_v1_counterfactual,dm_multidomain_v1_counterfactual_test \
        --out dm_multidomain_v1_counterfactual_full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--stores", required=True, help="comma-separated cf-subdir names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(yaml.safe_load(Path(args.config).read_text())["paths"]["data_root"]) / "activations" / "qwen"
    stores = [base / s for s in args.stores.split(",")]
    acts, lls, idx_lines = [], [], []
    for s in stores:
        a = np.load(s / "activations.npy", mmap_mode="r")
        l = np.load(s / "letter_logits.npy")
        lines = (s / "index.jsonl").read_text().splitlines()
        assert len(lines) == a.shape[0] == l.shape[0], f"{s}: row mismatch"
        print(f"{s.name}: {a.shape[0]} items  acts{tuple(a.shape)}  ll{tuple(l.shape)}")
        acts.append(a); lls.append(l); idx_lines += [ln for ln in lines if ln.strip()]
    # shapes must match on non-item axes
    assert len({a.shape[1:] for a in acts}) == 1, "activation shapes differ on non-item axes"
    assert len({l.shape[1:] for l in lls}) == 1, "letter_logit shapes differ on non-item axes"

    out = base / args.out; out.mkdir(parents=True, exist_ok=True)
    merged_acts = np.concatenate([np.asarray(a) for a in acts], axis=0)
    merged_ll = np.concatenate([np.asarray(l) for l in lls], axis=0)
    np.save(out / "activations.npy", merged_acts)
    np.save(out / "letter_logits.npy", merged_ll)
    (out / "index.jsonl").write_text("\n".join(idx_lines) + "\n")
    print(f"\n-> {out}  ({merged_acts.shape[0]} items, acts {tuple(merged_acts.shape)})")


if __name__ == "__main__":
    main()
