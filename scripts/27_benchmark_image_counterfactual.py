"""Step 27: image-counterfactual steering on the natural benchmarks.

Decisive transfer test (project_log "De-Confounding v2" verdict + pivot): does the
image's OWN causal contribution, surfaced per-item from hidden space, recover the
prior-violating failures on natural VQA -- where D_M's caption-vs-image construct
does not apply?

Per item we define the counterfactual direction
    delta_i(L) = h_img(L) - h_noimg(L)            (last-token residual at layer L)
where h_img = forward(image + question), h_noimg = forward(question only, image
dropped). delta_i is the image's per-item contribution to the residual stream --
no caption, no D_M, definable on any VQA row. Steering amplifies it:
    h <- h + alpha * ||h_last|| * delta_i/||delta_i||   (broadcast + norm-match)
the established best variant. The matched null replaces delta_i/||delta_i|| with a
per-item RANDOM unit vector (same magnitude), so any real lift must beat noise.

We sweep (layer, alpha) and report accuracy on the prior-violating subpopulation
(ViLP: where the image is required and the prior answer is a distractor -- the only
real headroom) vs the prior-consistent items (collateral), and overall (MMStar).

Stage A (collect): per row, two forwards (image / no-image) -> delta_i for all
layers + baseline letter logits. GPU. Stage B (steer): layer x alpha sweep reusing
the cached deltas. GPU.

    # ViLP (prior tags) -- collect then sweep:
    CUDA_VISIBLE_DEVICES=0 python scripts/27_benchmark_image_counterfactual.py \
        --benchmark vilp --stage all
    # MMStar (overall only):
    CUDA_VISIBLE_DEVICES=0 python scripts/27_benchmark_image_counterfactual.py \
        --benchmark mmstar --stage all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from sae_steering.models import load_qwen_vl
from sae_steering.steering import (
    build_messages, AdditiveHook, get_letter_token_ids, CANONICAL_MAX_PIXELS, score_row,
)

SWEEP_LAYERS = [13, 20, 23, 28, 31]
SWEEP_ALPHAS = [0.5, 1.0, 2.0, 4.0]   # norm-match fractional coefficients (O(1))


def build_messages_noimg(row: dict) -> list[dict]:
    """Same canonical prompt as build_messages, with the image block removed."""
    msgs = build_messages(row)
    content = [c for c in msgs[0]["content"] if c.get("type") != "image"]
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def _last_token_hidden(model, processor, msgs) -> tuple[np.ndarray, int, torch.Tensor]:
    """Forward `msgs`; return (per-layer last-token hidden [n_layers, D], last_idx, letter_logits row)."""
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    if image_inputs is not None and len(image_inputs) == 0:
        image_inputs = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
    out = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple(n_layers+1) of [B, T, D]; drop the embedding layer (index 0)
    hs = torch.stack([h[0, last_idx] for h in out.hidden_states[1:]], dim=0)  # [n_layers, D]
    return hs.float().cpu().numpy(), last_idx, out.logits[0, last_idx]


def _model_dims(model) -> tuple[int, int]:
    """(n_layers, d_model) for the language model, robust to nested VL configs."""
    layers = model.model.language_model.layers
    n_layers = len(layers)
    tcfg = getattr(model.config, "text_config", model.config)
    d_model = getattr(tcfg, "hidden_size", None) or layers[0].input_layernorm.weight.shape[0]
    return n_layers, int(d_model)


def _margin(letter_logits: torch.Tensor) -> float:
    """Top1 - top2 softmax prob over the option letters (confidence proxy)."""
    p = torch.softmax(letter_logits.float(), dim=-1)
    top = torch.topk(p, min(2, p.numel())).values
    return float(top[0] - top[1]) if top.numel() > 1 else float(top[0])


def stage_collect(model, processor, rows, letter_ids_all, out_dir: Path):
    n_layers, d_model = _model_dims(model)
    deltas = np.zeros((len(rows), n_layers, d_model), dtype=np.float16)
    h_img_all = np.zeros((len(rows), n_layers, d_model), dtype=np.float16)   # with-image state (h_noimg = h_img - delta)
    base_pred = np.zeros(len(rows), dtype=np.int64)        # image+question answer
    noimg_pred = np.zeros(len(rows), dtype=np.int64)       # question-only (prior) answer
    base_margin = np.zeros(len(rows), dtype=np.float32)    # confidence with image
    noimg_margin = np.zeros(len(rows), dtype=np.float32)   # confidence without image (prior strength)
    for i, row in enumerate(tqdm(rows, desc="collect delta_i")):
        h_img, _, logits_img = _last_token_hidden(model, processor, build_messages(row))
        h_noimg, _, logits_noimg = _last_token_hidden(model, processor, build_messages_noimg(row))
        deltas[i] = (h_img - h_noimg).astype(np.float16)
        h_img_all[i] = h_img.astype(np.float16)
        k = len(row["options"])
        li, ln = logits_img[letter_ids_all[:k]], logits_noimg[letter_ids_all[:k]]
        base_pred[i] = int(li.argmax().item());  noimg_pred[i] = int(ln.argmax().item())
        base_margin[i] = _margin(li);            noimg_margin[i] = _margin(ln)
    np.save(out_dir / "delta.npy", deltas)
    np.save(out_dir / "h_img.npy", h_img_all)             # h_noimg = h_img - delta
    np.save(out_dir / "baseline_pred.npy", base_pred)
    np.save(out_dir / "noimg_pred.npy", noimg_pred)
    np.save(out_dir / "base_margin.npy", base_margin)
    np.save(out_dir / "noimg_margin.npy", noimg_margin)
    (out_dir / "index.jsonl").write_text(
        "".join(json.dumps({"candidate_id": r["candidate_id"],
                            "correct_index": r["correct_index"],
                            "prior_consistent": r.get("prior_consistent"),
                            "question_group": r.get("question_group")}) + "\n" for r in rows))
    print(f"stage A: delta {deltas.shape} + baseline preds -> {out_dir}")


def _acc(pred, ci, mask):
    m = mask if mask is not None else np.ones(len(ci), bool)
    return float((pred[m] == ci[m]).mean()) if m.sum() else float("nan")


def stage_steer(model, processor, rows, letter_ids_all, out_dir: Path, args):
    deltas = np.load(out_dir / "delta.npy")
    base_pred = np.load(out_dir / "baseline_pred.npy")
    # metadata from the in-memory rows (same order as collection) -> no re-collect needed
    ci = np.array([r["correct_index"] for r in rows])
    pv = np.array([r.get("prior_consistent") is False for r in rows])   # prior-violating
    pc = np.array([r.get("prior_consistent") is True for r in rows])    # prior-consistent
    has_prior = pv.any() or pc.any()

    # grouped dev/test split: pick (layer, alpha) on dev, report on held-out test.
    # group by question_group (ViLP variants of one image stay together) else candidate_id.
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("question_group", r["candidate_id"]), []).append(i)
    keys = sorted(groups); g_rng = np.random.default_rng(args.random_seed); g_rng.shuffle(keys)
    test_keys = set(keys[:int(round(len(keys) * args.test_frac))])
    is_test = np.zeros(len(rows), bool)
    for g, ii in groups.items():
        if g in test_keys:
            is_test[ii] = True
    dev = ~is_test
    sel = "pv" if has_prior else "overall"
    print(f"split: {len(groups)} groups -> dev {dev.sum()} / test {is_test.sum()} rows; "
          f"select on dev by '{sel}'")

    rng = np.random.default_rng(args.random_seed)
    rand_unit = rng.standard_normal((len(rows), deltas.shape[-1])).astype(np.float32)
    rand_unit /= (np.linalg.norm(rand_unit, axis=1, keepdims=True) + 1e-8)

    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    results = []
    pred_store, pred_meta = [], []   # saved so any re-split is offline (no GPU re-run)

    def accs(pred, m):
        out = {"overall": _acc(pred, ci, m)}
        if has_prior:
            out["pv"] = _acc(pred, ci, m & pv)
            out["pc"] = _acc(pred, ci, m & pc)
        return out

    def run(vecs, L, alpha, tag):
        hook = AdditiveHook(torch.zeros(deltas.shape[-1], device=model.device),
                            alpha=alpha, broadcast=True, norm_match=True)
        pred = np.zeros(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            v = torch.tensor(vecs[i], device=model.device, dtype=torch.float32)
            hook.v_unit = v / (v.norm() + 1e-8)
            k = len(row["options"])
            pred[i] = score_row(model, processor, row, hook, L, letter_ids_all[:k])
        cell = {"tag": tag, "layer": L, "alpha": alpha,
                "dev": accs(pred, dev), "test": accs(pred, is_test)}
        results.append(cell)
        pred_store.append(pred.copy()); pred_meta.append({"tag": tag, "layer": L, "alpha": alpha})
        return cell

    results.append({"tag": "baseline", "layer": -1, "alpha": 0.0,
                    "dev": accs(base_pred, dev), "test": accs(base_pred, is_test)})

    for L in layers:
        dL = deltas[:, L, :].astype(np.float32)
        for a in alphas:
            for tag, vecs in (("real", dL), ("random", rand_unit)):
                c = run(vecs, L, a, tag)
                print(f"L{L} a={a} {tag:6} dev: " +
                      "  ".join(f"{k}={v:.3f}" for k, v in c["dev"].items()))
    (out_dir / "sweep_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    np.save(out_dir / "sweep_preds.npy", np.stack(pred_store))   # (n_cells, n_rows)
    (out_dir / "sweep_pred_meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in pred_meta))

    # dev-selected headline, reported on held-out test (real vs matched null at the same cell)
    real = [c for c in results if c["tag"] == "real"]
    best = max(real, key=lambda c: c["dev"][sel])
    null = next(c for c in results if c["tag"] == "random"
                and c["layer"] == best["layer"] and c["alpha"] == best["alpha"])
    base = results[0]
    print(f"\n=== dev-selected cell: L{best['layer']} alpha={best['alpha']} (by dev {sel}) ===")
    print(f"{'':10}{'overall':>9}" + ("".join(f"{k:>9}" for k in ('pv', 'pc')) if has_prior else ""))
    for name, c in (("baseline", base), ("real", best), ("null", null)):
        row = c["test"]
        print(f"{name:10}{row['overall']:>9.3f}" +
              ("".join(f"{row[k]:>9.3f}" for k in ('pv', 'pc')) if has_prior else "")
              + ("   <- HELD-OUT TEST"))
    print(f"stage B: {len(results)} cells -> {out_dir/'sweep_results.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="benchmark name; rows read from data/benchmarks/<name>/rows.jsonl unless --rows-jsonl")
    ap.add_argument("--stage", choices=["collect", "steer", "all"], default="all")
    ap.add_argument("--rows-jsonl", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--layers", default=",".join(str(L) for L in SWEEP_LAYERS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in SWEEP_ALPHAS))
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.5,
                    help="fraction of groups held out for the reported test (rest = dev for layer/alpha selection)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows_jsonl = Path(args.rows_jsonl or f"data/benchmarks/{args.benchmark}/rows.jsonl")
    rows = [json.loads(l) for l in rows_jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"no rows in {rows_jsonl}")
    out_dir = Path(args.out_dir or f"data/diagnostics/img_counterfactual/{args.benchmark}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"benchmark={args.benchmark} rows={len(rows)} -> {out_dir}")

    model, processor = load_qwen_vl(device=args.device)
    letter_ids_all = get_letter_token_ids(processor, args.device, n_letters=6)

    if args.stage in ("collect", "all"):
        stage_collect(model, processor, rows, letter_ids_all, out_dir)
    if args.stage in ("steer", "all"):
        stage_steer(model, processor, rows, letter_ids_all, out_dir, args)


if __name__ == "__main__":
    main()
