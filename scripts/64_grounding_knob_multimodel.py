"""#2 (multi-model): causal grounding-knob via DOWNSCALE only, for any model, on D_M and/or assembled.

Reuses scripts/61's adapters + row loaders (download-run-delete). For one model, sweeps image
downscale fractions on vision items of the chosen pool(s) and records (grounding, v-distraction)
per level -> data/diagnostics/grounding_knob_{key}_downscale[_assembled].json. The combined overlay
figure (all models/pools on the cross-model law) is built afterward by paper/make_figs.py.

    CUDA_VISIBLE_DEVICES=0 python scripts/64_grounding_knob_multimodel.py --model Qwen/Qwen2.5-VL-7B-Instruct
    CUDA_VISIBLE_DEVICES=0 python scripts/64_grounding_knob_multimodel.py --model qwen2.5-vl-3b --key qwen  # match law cell key
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s61", ROOT / "61_behavioral_eval.py")
s61 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s61)  # type: ignore
MAKE = {"qwen": s61.make_qwen, "hf": s61.make_hf, "llava": s61.make_llava, "gemini": s61.make_gemini}


def resolve(model):
    spec = s61.REGISTRY.get(model)
    if spec is None:
        p = model.split(":")
        if p[0] == "qwen":
            spec = {"type": "qwen", "model_id": p[1], "cls": p[2]}
        elif p[0] == "gemini":
            spec = {"type": "gemini", "model_id": p[1]}
        elif p[0] == "hf":
            spec = {"type": "hf", "model_id": p[1]}
        else:
            spec = {"type": "hf", "model_id": model}
    return spec


def run_knob(predict3, pool, key, cfg, levels, limit, seed, tmp, dump_dir=None):
    rows = [r for r in s61.load_rows(pool, cfg) if r["label"] == "vision"]
    rng = np.random.default_rng(seed); rng.shuffle(rows); rows = rows[:limit]
    print(f"--- pool={pool}  vision items={len(rows)}  levels={levels} ---")
    traj = []
    for lv in levels:
        pv, pvt, ci = [], [], []
        items = []                      # per-item rows, written only when --dump-dir is given
        for r in tqdm(rows, desc=f"{pool} downscale={lv}", leave=False):
            if lv >= 1.0:
                row = r
            else:
                img = Image.open(r["image_path"]).convert("RGB")
                w, h = img.size
                img.resize((max(1, int(w * lv)), max(1, int(h * lv)))).save(tmp, quality=95)
                row = {**r, "image_path": str(tmp)}
            preds, logs = predict3(row)         # (pred_vt, pred_v, pred_t)
            pvt.append(preds[0]); pv.append(preds[1]); ci.append(r["correct_index"])
            if dump_dir is not None:
                # same schema as data/eval/t8/*.jsonl, so the margin decomposition reads the input
                # knob exactly like a weight-space operating point
                it = {"candidate_id": r["candidate_id"], "source": r.get("source"),
                      "label": r["label"], "correct_index": r["correct_index"],
                      "pred_vt": preds[0], "pred_v": preds[1]}
                if logs is not None:
                    it["logits_vt"] = [float(x) for x in logs[0]]
                    it["logits_v"] = [float(x) for x in logs[1]]
                items.append(it)
        pv, pvt, ci = map(np.array, (pv, pvt, ci)); vok = pv == ci
        traj.append({"level": lv, "grounding": round(float(vok.mean()), 4),
                     "v_distraction": round(float(((pv == ci) & (pvt != ci)).sum() / max(vok.sum(), 1)), 4),
                     "vt_acc": round(float((pvt == ci).mean()), 4), "n": len(rows)})
        if dump_dir is not None:
            dd = Path(dump_dir); dd.mkdir(parents=True, exist_ok=True)
            fp = dd / f"{key}_downscale{'_assembled' if pool == 'assembled' else ''}_lv{lv}.jsonl"
            fp.write_text("".join(json.dumps(x) + "\n" for x in items))
            print(f"  -> {fp}")
        print(f"  downscale={lv:>5}: grounding={traj[-1]['grounding']:.3f}  v-distraction={traj[-1]['v_distraction']:.3f}")
    suffix = "_assembled" if pool == "assembled" else ""
    out = Path(cfg["paths"]["data_root"]) / "diagnostics" / f"grounding_knob_{key}_downscale{suffix}.json"
    out.write_text(json.dumps({"model": key, "key": key, "pool": pool, "knob": "downscale", "trajectory": traj}, indent=2))
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=None,
                    help="also write per-item predictions+letter logits per level, in the t8 schema "
                         "(needed to decompose a flip into margin vs caption pressure)")
    ap.add_argument("--conds", default="",
                    help="comma list of conditions (VT,V,T); the knob itself only needs VT,V, so "
                         "restricting saves a third of the passes")
    ap.add_argument("--model", required=True, help="HF path or REGISTRY key (see scripts/61)")
    ap.add_argument("--key", default=None, help="override output key (default: sanitized --model; use to match law cell, e.g. qwen)")
    ap.add_argument("--pool", default="both", choices=["dm", "assembled", "both"])
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--levels", default="1.0,0.7,0.5,0.35,0.25,0.15")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true", help="keep HF checkpoint (default: delete after)")
    args = ap.parse_args()

    if args.conds:
        want = {c.strip().upper() for c in args.conds.split(",") if c.strip()}
        s61.COND = [c for c in s61.ALL_COND if c[0] in want]
        assert [c[0] for c in s61.COND][:2] == ["VT", "V"], \
            "the knob indexes preds[0]=VT and preds[1]=V, so VT and V must come first"
        print(f"[conds] running only {[c[0] for c in s61.COND]}")

    cfg = yaml.safe_load(Path(args.config).read_text())
    key = args.key or re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    levels = [float(x) for x in args.levels.split(",")]
    pools = ["dm", "assembled"] if args.pool == "both" else [args.pool]
    spec = resolve(args.model)
    print(f"=== knob/downscale {args.model} (type={spec['type']}, key={key}) pools={pools} ===")
    predict3 = MAKE[spec["type"]](spec, args.device)

    tmp = Path(tempfile.mkdtemp()) / "deg.jpg"
    for pool in pools:
        run_knob(predict3, pool, key, cfg, levels, args.limit, args.seed, tmp, args.dump_dir)

    if not args.keep and spec["type"] in ("hf", "qwen") and spec.get("model_id"):
        del predict3; s61.delete_hf_cache(spec["model_id"])


if __name__ == "__main__":
    main()
