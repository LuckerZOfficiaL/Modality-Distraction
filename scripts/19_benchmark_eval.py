"""Step 19: additive-method eval + sweep on external MCQ benchmarks.

Runs on a materialized benchmark pool (data/benchmarks/<name>/rows.jsonl from
19a) under the canonical forward-pass setup. No cf-store / probe / patching:
external benchmarks have no separable text modality, so the V-only counterfactual
h_V is undefined and probe-gated patching is degenerate (h_V == h_VT). The
runnable family is the additive v_V_causal direction and its matched random null.

Per-row option arity is handled: letters are sliced to len(row["options"]).

Two modes:

  (default) headline -- baseline + additive_v_causal + random_matched at a single
      (layer, alpha), the held-out-test operating point (L28, a=+1, bcast+nm):
        CUDA_VISIBLE_DEVICES=0 python scripts/19_benchmark_eval.py --benchmark mmstar

  --sweep -- additive real + matched random across --layers x --alphas (bcast+nm),
      same grid convention as 18b. Ensures baseline exists, then writes one jsonl
      per (kind, layer, alpha):
        CUDA_VISIBLE_DEVICES=0 python scripts/19_benchmark_eval.py --benchmark mmstar --sweep

(The L31 SAE method slots into make_hook once its checkpoint exists.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from moground.canonical_eval import (
    load_pool_rows_jsonl,
    random_matched_unit_vector,
    run_canonical_forward as run_forward,
)
from moground.models import load_qwen_vl
from moground.sae import SAEConfig, TopKSAE
from moground.steering import AdditiveHook, SAEHook, get_letter_token_ids


METHODS = ["baseline", "additive_v_causal", "random_matched"]
SWEEP_LAYERS = [13, 20, 23, 28]          # layers present in v_causal_vectors.npz
SWEEP_ALPHAS = [-3, -1, -0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5, 1, 3]  # NM grid


def _alpha_tag(a: float) -> str:
    return f"{a:+g}".replace("+", "p").replace("-", "m")


def _load_sae(data_root: Path, variant: str, layer: int, device: str) -> TopKSAE:
    ck = torch.load(data_root / f"sae/qwen/{variant}/layer{layer}/sae.pt",
                    map_location=device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
    sae.load_state_dict(ck["state_dict"])
    sae.to(device).to(torch.float32).eval()
    return sae


def _load_top_feature_idx(features_dir: Path, layer: int, top_n: int, device: str):
    """Top-N vision- and text-preferring latent indices (discriminative rho)."""
    feats = json.loads((features_dir / f"layer{layer}" / "top_features.json").read_text())
    v = torch.tensor([r["k"] for r in feats["top_vision"][:top_n]], device=device, dtype=torch.long)
    t = torch.tensor([r["k"] for r in feats["top_text"][:top_n]], device=device, dtype=torch.long)
    return v, t


def _random_alive_idx(features_dir: Path, layer: int, n_v: int, n_t: int,
                      seed: int, device: str):
    """Matched null for the SAE method: random ALIVE latents (split into a
    pseudo-vision and pseudo-text group of the same sizes as the real atoms)."""
    import csv
    rows = list(csv.DictReader(open(features_dir / f"layer{layer}" / "feature_stats.csv")))
    alive = [int(r["k"]) for r in rows if str(r.get("alive", "")).lower() in ("true", "1")]
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.array(alive), size=min(n_v + n_t, len(alive)), replace=False)
    v = torch.tensor(pick[:n_v], device=device, dtype=torch.long)
    t = torch.tensor(pick[n_v:n_v + n_t], device=device, dtype=torch.long)
    return v, t


def _run_cell(model, processor, pool, letter_ids_all, hook, layer_idx,
              out_path: Path, desc: str) -> None:
    """Forward the whole pool through one hook config, resumable, per-row arity."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["candidate_id"])
    if done:
        print(f"  {out_path.name}: resume — {len(done)} done")
    f = out_path.open("a")
    for row in tqdm(pool, desc=desc, leave=False):
        cid = row["candidate_id"]
        if cid in done:
            continue
        letter_ids = letter_ids_all[:len(row["options"])]
        pred = run_forward(model, processor, row, letter_ids,
                           hook=hook, layer_idx=layer_idx)
        rec = {
            "candidate_id": cid,
            "source": row.get("source", "?"),
            "label": row.get("label", "vision"),
            "correct_index": row["correct_index"],
            "pred": pred,
        }
        for k in ("variant", "prior_consistent", "group", "category"):
            if k in row:
                rec[k] = row[k]
        f.write(json.dumps(rec) + "\n")
        f.flush()
    f.close()


def summarize(out_dir: Path, tag: str, methods: list[str]) -> None:
    """Per-source accuracy + Δ vs baseline for the named headline methods."""
    def load(m):
        p = out_dir / f"{m}_{tag}.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else None

    base = load("baseline")
    if base is None:
        print("(no baseline file; skipping summary)")
        return
    correct = {r["candidate_id"]: r["correct_index"] for r in base}

    def acc(rows, subset=None):
        rows = [r for r in rows if subset is None or subset(r)]
        if not rows:
            return float("nan"), 0
        return sum(1 for r in rows if r["pred"] == correct[r["candidate_id"]]) / len(rows), len(rows)

    sources = sorted({r["source"] for r in base})
    print(f"\n=== Benchmark eval summary [{tag}] ===")
    print(f"{'method':<20} " + "  ".join(f"{s:>14}" for s in sources) + f"{'OVERALL':>16}")
    ba, _ = acc(base)
    base_by_src = {s: acc(base, lambda r, s=s: r['source'] == s)[0] for s in sources}
    for m in methods:
        rows = load(m)
        if rows is None:
            continue
        cells = []
        for s in sources:
            a, nsub = acc(rows, lambda r, s=s: r["source"] == s)
            cells.append(f"{a:.3f}({a-base_by_src[s]:+.3f})" if m != "baseline" else f"{a:.3f}[{nsub}]")
        a_all, n_all = acc(rows)
        tail = f"{a_all:.3f}({a_all-ba:+.3f})" if m != "baseline" else f"{a_all:.3f}[{n_all}]"
        print(f"{m:<20} " + "  ".join(f"{c:>14}" for c in cells) + f"{tail:>16}")

    if any(r["source"] == "vilp" for r in base):
        print("\n-- ViLP prior-consistent (do-no-harm) vs prior-violating (recovery) --")
        for m in methods:
            rows = load(m)
            if rows is None:
                continue
            rows = [r for r in rows if r["source"] == "vilp"]
            pc, _ = acc(rows, lambda r: r.get("prior_consistent") is True)
            pv, _ = acc(rows, lambda r: r.get("prior_consistent") is False)
            print(f"  {m:<20} prior-consistent={pc:.3f}  prior-violating={pv:.3f}")


def summarize_sweep(out_dir: Path, tag: str, layers, alphas, kinds, cell_prefix: str = "") -> None:
    """Grid of overall pool-acc Δ vs baseline for each (kind, layer, alpha)."""
    bp = out_dir / f"baseline_{tag}.jsonl"
    if not bp.exists():
        print("(no baseline file; skipping sweep summary)")
        return
    base = [json.loads(l) for l in bp.read_text().splitlines() if l.strip()]
    correct = {r["candidate_id"]: r["correct_index"] for r in base}
    ba = sum(1 for r in base if r["pred"] == correct[r["candidate_id"]]) / len(base)

    def cell_acc(kind, L, a):
        p = out_dir / f"{cell_prefix}{kind}_L{L}_a{_alpha_tag(a)}_{tag}.jsonl"
        if not p.exists():
            return None
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            return None
        return sum(1 for r in rows if r["pred"] == correct[r["candidate_id"]]) / len(rows)

    print(f"\n=== Additive sweep [{tag}] — Δ pool acc vs baseline {ba:.3f} (n={len(base)}) ===")
    best = (None, -1.0)
    for kind in kinds:
        print(f"\n{kind}:")
        print("  L \\ a | " + " ".join(f"{a:>+6g}" for a in alphas))
        for L in layers:
            cells = []
            for a in alphas:
                ac = cell_acc(kind, L, a)
                if ac is None:
                    cells.append("   .  ")
                else:
                    d = ac - ba
                    cells.append(f"{d:>+6.3f}")
                    if kind == "real" and d > best[1]:
                        best = ((L, a), d)
            print(f"  L{L:<3d} | " + " ".join(cells))
    if best[0]:
        print(f"\nbest real cell: L{best[0][0]} α={best[0][1]:+g} → Δ {best[1]:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=None,
                    help="name under data/benchmarks/ (mmstar/vilp/naturalbench)")
    ap.add_argument("--rows-jsonl", default=None, help="explicit rows.jsonl (overrides --benchmark)")
    ap.add_argument("--tag", default=None, help="output suffix (defaults to --benchmark)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out-dir", default="data/diagnostics/benchmark_eval")
    ap.add_argument("--steering-vectors", default="data/steering/canonical/v_causal_vectors.npz")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--layer", type=int, default=28, help="headline-mode layer")
    ap.add_argument("--alpha", type=float, default=1.0, help="headline-mode alpha")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    # sweep mode
    ap.add_argument("--sweep", action="store_true",
                    help="sweep across --layers x --alphas (bcast+nm) with a matched null")
    ap.add_argument("--method", default="additive", choices=["additive", "sae"],
                    help="additive: v_V_causal vs random unit vector; "
                         "sae: discriminative modality atoms vs random alive atoms")
    ap.add_argument("--layers", default=",".join(str(L) for L in SWEEP_LAYERS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in SWEEP_ALPHAS))
    ap.add_argument("--kinds", default="real,random")
    # sae-method options
    ap.add_argument("--sae-variant", default="diversified_v1")
    ap.add_argument("--sae-layers", default="31", help="SAE layers (method=sae); overrides --layers")
    ap.add_argument("--features-dir", default=None,
                    help="default data_root/features/qwen/<sae-variant>")
    ap.add_argument("--top-n", type=int, default=20)
    # prompt-conditioning baseline (orthogonal to steering; no hook). Directive is
    # modality-matched by row label: V rows get --directive-v, T rows get --directive-t
    # (parallels routing the intervention by modality). On the all-vision benchmarks
    # only --directive-v applies.
    ap.add_argument("--prompt-cond", action="store_true",
                    help="run the no-hook label-matched prompt-conditioning baseline")
    ap.add_argument("--directive-v", default="Focus on the image to answer the question.")
    ap.add_argument("--directive-t", default="Focus on the text to answer the question.")
    # output-name prefix for sweep cells, so different steering vectors don't collide
    # (e.g. --cell-prefix instr_ for the instruction-contrastive vector)
    ap.add_argument("--cell-prefix", default="")
    args = ap.parse_args()

    if args.rows_jsonl is None:
        if args.benchmark is None:
            ap.error("provide --benchmark or --rows-jsonl")
        args.rows_jsonl = f"data/benchmarks/{args.benchmark}/rows.jsonl"
    tag = args.tag or args.benchmark or Path(args.rows_jsonl).parent.name

    pool = load_pool_rows_jsonl(Path(args.rows_jsonl))  # no cf-store filtering
    print(f"pool ({tag}): {len(pool)} rows from {args.rows_jsonl}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    model, processor = load_qwen_vl(device=args.device)
    letter_ids_all = get_letter_token_ids(processor, args.device, n_letters=6)

    # ===== prompt-conditioning baseline (no hook, modality-matched directive) =====
    if args.prompt_cond:
        for r in pool:
            r["directive"] = args.directive_v if r.get("label") == "vision" else args.directive_t
        _run_cell(model, processor, pool, letter_ids_all, None, None,
                  out_dir / f"prompt_directive_{tag}.jsonl", desc=f"prompt_directive [{tag}]")
        print(f"prompt_directive: done (V={args.directive_v!r}  T={args.directive_t!r})")
        summarize(out_dir, tag, ["prompt_directive"])  # vs existing baseline_{tag}
        return

    sv = np.load(args.steering_vectors)
    d_model = int(sv[f"v_V_L{args.layer}"].shape[-1])
    v_rand = random_matched_unit_vector(d_model, args.random_seed, args.device)

    if not args.sweep:
        # ===== headline mode (single operating point) =====
        v_V = torch.tensor(sv[f"v_V_L{args.layer}"], device=args.device, dtype=torch.float32)
        methods = [m.strip() for m in args.methods.split(",")]

        def make_hook(method):
            if method == "baseline":
                return None, None
            if method == "additive_v_causal":
                return AdditiveHook(v_V, alpha=args.alpha, broadcast=True, norm_match=True), args.layer
            if method == "random_matched":
                return AdditiveHook(v_rand, alpha=args.alpha, broadcast=True, norm_match=True), args.layer
            raise ValueError(f"unknown method {method!r}")

        for method in methods:
            hook, layer_idx = make_hook(method)
            _run_cell(model, processor, pool, letter_ids_all, hook, layer_idx,
                      out_dir / f"{method}_{tag}.jsonl", desc=f"{method} [{tag}]")
            print(f"{method}: done")
        summarize(out_dir, tag, methods)
        return

    # ===== sweep mode =====
    # One shared loop over layers x alphas x kinds. The only thing a steering
    # method customizes is (a) its layers, (b) per-layer setup, (c) the hook it
    # builds for (kind, alpha), (d) its output cell prefix. Adding a new
    # vector-based baseline needs no code here: point --steering-vectors at a
    # different npz and pass --cell-prefix.
    alphas = [float(x) for x in args.alphas.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]
    layers, cell_prefix, setup_layer, build_hook = _make_sweep_method(
        args, sv, v_rand, data_root)
    print(f"sweep {args.method} (prefix={cell_prefix!r}): layers={layers} alphas={alphas} "
          f"kinds={kinds} ({len(layers)*len(alphas)*len(kinds)} cells + baseline)")

    _run_cell(model, processor, pool, letter_ids_all, None, None,
              out_dir / f"baseline_{tag}.jsonl", desc=f"baseline [{tag}]")
    for L in layers:
        ctx = setup_layer(L)
        for a in alphas:
            for kind in kinds:
                hook = build_hook(ctx, kind, a)
                _run_cell(model, processor, pool, letter_ids_all, hook, L,
                          out_dir / f"{cell_prefix}{kind}_L{L}_a{_alpha_tag(a)}_{tag}.jsonl",
                          desc=f"{args.method} {kind} L{L} α={a} [{tag}]")
    summarize_sweep(out_dir, tag, layers, alphas, kinds, cell_prefix=cell_prefix)


def _make_sweep_method(args, sv, v_rand, data_root):
    """Return (layers, cell_prefix, setup_layer, build_hook) for the chosen method.

    setup_layer(L) -> ctx (heavy per-layer load);  build_hook(ctx, kind, alpha)
    -> SteeringHook, where kind in {real, random}. All sweep methods share the
    loop in main(); only these four pieces differ."""
    device = args.device
    prefix = args.cell_prefix

    if args.method == "additive":
        # Any V-direction npz with keys v_V_L{L}: v_V_causal, the instruction-
        # contrastive vector, etc. real = the vector; random = matched unit null.
        layers = [int(x) for x in args.layers.split(",")]
        v_V = {L: torch.tensor(sv[f"v_V_L{L}"], device=device, dtype=torch.float32) for L in layers}

        def setup_layer(L):
            return v_V[L]

        def build_hook(ctx, kind, a):
            vec = ctx if kind == "real" else v_rand
            return AdditiveHook(vec, alpha=a, broadcast=True, norm_match=True)

        return layers, prefix, setup_layer, build_hook

    if args.method == "sae":
        # real = discriminative modality atoms; random = matched random alive atoms.
        layers = [int(x) for x in args.sae_layers.split(",")]
        feat_dir = Path(args.features_dir) if args.features_dir \
            else data_root / "features" / "qwen" / args.sae_variant
        prefix = prefix or "sae_"

        def setup_layer(L):
            sae = _load_sae(data_root, args.sae_variant, L, device)
            v_idx, t_idx = _load_top_feature_idx(feat_dir, L, args.top_n, device)
            rv_idx, rt_idx = _random_alive_idx(feat_dir, L, v_idx.numel(), t_idx.numel(),
                                               args.random_seed, device)
            return (sae, v_idx, t_idx, rv_idx, rt_idx)

        def build_hook(ctx, kind, a):
            sae, v_idx, t_idx, rv_idx, rt_idx = ctx
            vi, ti = (v_idx, t_idx) if kind == "real" else (rv_idx, rt_idx)
            return SAEHook(sae, vi, ti, alpha=a, broadcast=True, norm_match=True)

        return layers, prefix, setup_layer, build_hook

    raise ValueError(f"unknown --method {args.method!r}")


if __name__ == "__main__":
    main()
