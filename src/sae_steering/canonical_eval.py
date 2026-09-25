"""Canonical evaluation utilities.

Every new steering method should use these helpers so its numbers are
within-script clean against the canonical baseline (pool acc 0.671 on the
948-cid multidomain unseen pool). Importing from here guarantees:

- Same prompt setup as 06c (build_messages with max_pixels=1024^2).
- Same pool construction (948 unique cids: val ∪ train-fail ∪ distraction,
  filtered to cf-store presence).
- Same baseline source (cf-store letter_logits.npy V+T argmax).
- Same forward-pass boilerplate (run_canonical_forward).

Workflow for a new method:
    1. Copy scripts/eval_template.py to scripts/<new>.py.
    2. Replace `make_hook(...)` to produce your AdditiveHook / SAEHook / custom.
    3. Always include a random matched-magnitude control with the same hook
       structure (use random_matched_unit_vector()).
    4. Run; numbers in <method>.jsonl are directly comparable to
       data/diagnostics/canonical_eval/* via compare_to_baseline().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from sae_steering.steering import (
    CANONICAL_CF_SUBDIR,
    SteeringHook,
    build_messages,
    load_canonical_baseline,
    load_distraction_pool,
)


CANONICAL_BASELINE_DIR = Path("data/diagnostics/canonical_eval")


class SetActivationHook(SteeringHook):
    """Override last_idx residual with a stored target vector.

    Used for activation patching (replace h_VT(L) with h_V(L) or h_T(L)).
    """
    def __init__(self, target: torch.Tensor):
        super().__init__(broadcast=False)
        self.target = target

    def edit(self, h):
        return self.target.to(h.dtype).expand_as(h)


NAMED_POOL_ALIASES = {
    # "unseen" is pinned to the exact 948 cids in data/pools/unseen.txt so the
    # project_log's headline numbers reproduce. New compositions should use
    # the explicit "<split>_<filter>" component syntax.
    "unseen":    "@data/pools/unseen.txt",
    "test":      "test_all",
    "all":       "train_all,val_all,test_all,distraction",
    # Convenience presets:
    "val":       "val_all",
    "val_pass":  "val_pass_all",
    "train":     "train_all",
    "train_pass":"train_pass_all",
}

# Components recognized by `load_canonical_pool`:
#   <split>_<filter>   where split ∈ {train, val, test, train_pass, val_pass}
#                       and  filter ∈ {all, fails, pass}
#   distraction        (no filter; all distraction-pool items)


def canonical_baseline_path(pool: str = "unseen") -> Path:
    """Return the canonical baseline jsonl path for a given pool."""
    suffix = "" if pool == "unseen" else f"_{pool}"
    return CANONICAL_BASELINE_DIR / f"baseline{suffix}.jsonl"


def _parse_pool_spec(spec: str | list[str]) -> list[str]:
    """Normalize a pool spec into a list of component strings.

    A component is one of:
      - "<split>_<filter>"  where split ∈ train/val/test/train_pass/val_pass
                            and filter ∈ all/fails/pass
      - "distraction"
      - "@<path>.txt"       a pinned cid list (one cid per line)
    """
    if isinstance(spec, list):
        return [c.strip() for c in spec if c.strip()]
    spec = spec.strip()
    if spec in NAMED_POOL_ALIASES:
        spec = NAMED_POOL_ALIASES[spec]
    return [c.strip() for c in spec.replace("+", ",").split(",") if c.strip()]


def load_pool_rows_jsonl(
    path: Path,
    data_root: Path | None = None,
    cf_subdir: str | None = None,
) -> list[dict]:
    """Load an evaluation pool directly from a rows JSONL (e.g. an assembled /
    cross-domain test set), bypassing the multidomain split structure.

    Rows must carry candidate_id, image_path, caption_for_filter, question,
    options, correct_index, label. Deduped by candidate_id. If a cf_subdir is
    given, filters to rows present in that cf-store (required for patching /
    probe-input methods) and warns about any missing.
    """
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        cid = r["candidate_id"]
        if cid in seen: continue
        seen.add(cid)
        r.setdefault("__split", "custom")
        out.append(r)
    if cf_subdir is not None and data_root is not None:
        cf_dir = Path(data_root) / "activations" / "qwen" / cf_subdir
        cf_index = [json.loads(l) for l in (cf_dir / "index.jsonl").read_text().splitlines() if l.strip()]
        cf_cids = {x["candidate_id"] for x in cf_index}
        kept = [r for r in out if r["candidate_id"] in cf_cids]
        if len(kept) < len(out):
            print(f"[load_pool_rows_jsonl] {len(out)-len(kept)}/{len(out)} rows missing "
                  f"from cf-store {cf_subdir!r}; excluded. Run 06c --rows-jsonl on this "
                  f"pool first if you need counterfactual (patching/probe) methods.")
        out = kept
    return out


def load_canonical_pool(
    data_root: Path,
    dm_dir: Path,
    distraction_pool_path: Path | None = None,
    pool: str | list[str] = "unseen",
    cf_subdir: str = CANONICAL_CF_SUBDIR,
) -> list[dict]:
    """Build a canonical pool deduped by candidate_id and filtered to cf-store.

    `pool` may be either:
      - a named alias: "unseen" (default), "test", "all", "val", "train", "val_pass", "train_pass"
      - a comma-or-+-separated component spec, e.g. "val_all,train_fails,distraction"
        or "val_all + test_all".
      - a list of component strings, e.g. ["val_all", "test_all"].

    Components have the form `<split>_<filter>`:
      split  ∈ {train, val, test, train_pass, val_pass}
      filter ∈ {all, fails, pass}  (fails / pass use the canonical baseline)

    Plus the special component "distraction" (all distraction-pool items;
    no filter because they're V+T-fail by construction).

    cf-store filter: items without h_V / h_T activations are dropped, with a
    warning. Counterfactual methods (patching) cannot run on items missing
    from cf_subdir.
    """
    components = _parse_pool_spec(pool)
    if not components:
        raise ValueError(f"Empty pool spec: {pool!r}")

    # Figure out which split files we need to load.
    base_splits: set[str] = set()
    pass_splits_needed: set[str] = set()
    needs_distraction = False
    pinned_cid_files: list[Path] = []
    for comp in components:
        if comp.startswith("@"):
            pinned_cid_files.append(Path(comp[1:]))
            # we don't yet know which splits these cids belong to; load all.
            base_splits.update({"train", "val", "test"})
            needs_distraction = True
            continue
        if comp == "distraction":
            needs_distraction = True
            continue
        if "_" not in comp:
            raise ValueError(f"Component {comp!r} has no filter; use <split>_<filter>")
        split, _, filt = comp.rpartition("_")
        if filt not in ("all", "fails", "pass"):
            raise ValueError(f"Component {comp!r}: unknown filter {filt!r}; "
                             f"use one of all/fails/pass")
        if split.endswith("_pass"):
            base = split.removesuffix("_pass")
            base_splits.add(base)
            pass_splits_needed.add(base)
        else:
            base_splits.add(split)

    # Load oracle survivors for the splits we need.
    rows: list[dict] = []
    for sp in base_splits:
        for line in (Path(dm_dir) / "splits" / f"{sp}.jsonl").read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line); r["__split"] = sp
            rows.append(r)
    # And the Qwen-pass subsets if any component asked for them.
    pass_rows: dict[str, set[str]] = {sp: set() for sp in pass_splits_needed}
    for sp in pass_splits_needed:
        pass_path = Path(dm_dir) / "pass" / f"{sp}.jsonl"
        if not pass_path.exists():
            raise FileNotFoundError(f"Pass split file missing: {pass_path}")
        for line in pass_path.read_text().splitlines():
            if not line.strip(): continue
            pass_rows[sp].add(json.loads(line)["candidate_id"])

    # Load distraction rows separately (they have a different schema).
    distraction_rows: list[dict] = []
    if needs_distraction and distraction_pool_path is not None:
        distraction_rows = load_distraction_pool(Path(distraction_pool_path))

    # Build per-row lookup
    rows_by_cid: dict[str, dict] = {r["candidate_id"]: r for r in rows}
    for r in distraction_rows:
        rows_by_cid.setdefault(r["candidate_id"], r)

    # Load canonical baseline if any component needs filtering.
    needs_baseline = any(comp.endswith("_fails") or comp.endswith("_pass") and comp != "val_pass" and comp != "train_pass" for comp in components)
    # Simpler: any *_fails or any non-pass-split *_pass.
    needs_baseline = any(
        comp.endswith("_fails") or
        (comp.endswith("_pass") and not (comp.startswith("val_pass") or comp.startswith("train_pass")))
        for comp in components
    )
    canonical_base: dict[str, int] = {}
    if needs_baseline:
        canonical_base, _, _ = load_canonical_baseline(data_root, cf_subdir)

    # Build the cid set per component, then union.
    selected_cids: set[str] = set()
    for comp in components:
        if comp.startswith("@"):
            path = Path(comp[1:])
            for line in path.read_text().splitlines():
                cid = line.strip()
                if cid: selected_cids.add(cid)
            continue
        if comp == "distraction":
            selected_cids.update(r["candidate_id"] for r in distraction_rows)
            continue
        split, _, filt = comp.rpartition("_")
        if split.endswith("_pass"):
            base = split.removesuffix("_pass")
            candidate = [r for r in rows
                         if r["__split"] == base and r["candidate_id"] in pass_rows[base]]
        else:
            candidate = [r for r in rows if r["__split"] == split]

        if filt == "all":
            selected_cids.update(r["candidate_id"] for r in candidate)
        elif filt == "fails":
            for r in candidate:
                bp = canonical_base.get(r["candidate_id"])
                if bp is not None and bp != r["correct_index"]:
                    selected_cids.add(r["candidate_id"])
        elif filt == "pass":
            for r in candidate:
                bp = canonical_base.get(r["candidate_id"])
                if bp is not None and bp == r["correct_index"]:
                    selected_cids.add(r["candidate_id"])

    # Resolve to row objects (dedup automatic via cid set).
    pool_rows = [rows_by_cid[c] for c in selected_cids if c in rows_by_cid]

    # cf-store filter
    cf_dir = Path(data_root) / "activations" / "qwen" / cf_subdir
    cf_index = [json.loads(l) for l in (cf_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cf_cids = {r["candidate_id"] for r in cf_index}
    out = [r for r in pool_rows if r["candidate_id"] in cf_cids]

    if len(out) < len(pool_rows):
        missing = len(pool_rows) - len(out)
        print(f"[canonical_pool] WARNING: {missing}/{len(pool_rows)} pool items "
              f"missing from cf-store (cf_subdir={cf_subdir!r}); excluded.")
        contains_test = any("test" in c for c in components)
        if contains_test:
            print(f"[canonical_pool]   The held-out test split was excluded from 06c per "
                  f"the 2026-06-04 data policy. Run scripts/collect_cf_activations.py "
                  f"--splits test (or equivalent) to extend the cf-store before evaluating on test.")
    return out


def load_cf_activations(data_root: Path, cf_subdir: str = CANONICAL_CF_SUBDIR):
    """Return (activations [N,3,n_layers,d], cid_to_index dict)."""
    cf_dir = Path(data_root) / "activations" / "qwen" / cf_subdir
    activations = np.load(cf_dir / "activations.npy", mmap_mode="r")
    cf_index = [json.loads(l) for l in (cf_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    cid_to_cf = {r["candidate_id"]: i for i, r in enumerate(cf_index)}
    return activations, cid_to_cf


@torch.inference_mode()
def run_canonical_forward(model, processor, row, letter_ids,
                           hook=None, layer_idx: int | None = None) -> int:
    """One canonical forward pass with optional hook. Returns letter argmax index."""
    msgs = build_messages(row)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
    handle = None
    if hook is not None and layer_idx is not None:
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
    return int(logits[letter_ids].argmax().item())


def random_matched_unit_vector(d_model: int, seed: int, device: str | torch.device) -> torch.Tensor:
    """Single seed-deterministic random unit vector. Used as matched-magnitude null."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d_model).astype(np.float32)
    v /= np.linalg.norm(v)
    return torch.tensor(v, device=device, dtype=torch.float32)


def compare_to_baseline(method_jsonl: Path, pool: str = "unseen",
                          baseline_jsonl: Path | str | None = None) -> dict:
    """Return pool acc, Δ vs canonical baseline.

    By default looks up the canonical baseline file for the given pool
    (unseen / test / etc). Override `baseline_jsonl` to supply a custom path.

    Errors if the canonical baseline for the pool hasn't been computed yet.
    """
    if baseline_jsonl is None:
        baseline_jsonl = canonical_baseline_path(pool)
    baseline_jsonl = Path(baseline_jsonl)
    if not baseline_jsonl.exists():
        raise FileNotFoundError(
            f"Canonical baseline for pool='{pool}' not found at {baseline_jsonl}. "
            f"Run scripts/18_canonical_eval.py --methods baseline --pool {pool} first."
        )
    base = {r["candidate_id"]: r for r in (json.loads(l) for l in baseline_jsonl.read_text().splitlines() if l.strip())}
    meth = {r["candidate_id"]: r for r in (json.loads(l) for l in method_jsonl.read_text().splitlines() if l.strip())}
    common = set(base) & set(meth)
    base_correct = sum(1 for c in common if base[c]["pred"] == base[c]["correct_index"])
    meth_correct = sum(1 for c in common if meth[c]["pred"] == base[c]["correct_index"])
    n = len(common)
    return {
        "n": n,
        "baseline_pool_acc": base_correct / n,
        "method_pool_acc": meth_correct / n,
        "delta_pool": (meth_correct - base_correct) / n,
    }
