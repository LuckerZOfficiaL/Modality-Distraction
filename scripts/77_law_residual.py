"""Distraction-aware finetuning: did it beat the law, and preserve capability?

Reads a finetuned model's 3-condition eval (data/eval/t8/{key}.jsonl + t8_assembled/{key}.jsonl,
from scripts/61 with --adapter) and its base counterpart. D_M metrics are computed on the TEST split
only (finetuning used train -> leakage-free); the assembled pool is inherently held-out. Reports:
  (1) LAW RESIDUAL: per vision cell, residual = actual v-distraction - law-predicted (8-model
      cross-model vision fit). Residual < 0 = LESS distracted than grounding predicts = the finetune
      DECOUPLED distraction from grounding (non-trivial win). ~0 = only slid along the law (trivial).
  (2) PRESERVATION: general multimodal (A-OKVQA V+T), text-only (RACE T-only), caption-use (RACE V+T);
      must stay within ~1-2pp of base.

    python scripts/77_law_residual.py --ft ft_dropout_qwen --base qwen
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parent.parent


def test_ids():
    p = RT / "data/dm/multidomain_v1/splits/test.jsonl"
    return {json.loads(l)["candidate_id"] for l in p.read_text().splitlines() if l.strip()}


def vision_cells(key, tids):
    """[(name, grounding=V-only acc, v-distraction)]: D_M-test per domain + assembled A-OKVQA."""
    out = []
    p = RT / f"data/eval/t8/{key}.jsonl"
    if p.exists():
        rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        by = collections.defaultdict(list)
        for r in rs:
            if r["candidate_id"] in tids and r["label"] == "vision":
                by[r["source"]].append(r)
        for dom, items in by.items():
            if len(items) < 20:
                continue
            ci = np.array([r["correct_index"] for r in items])
            v = np.array([r["pred_v"] for r in items]); vt = np.array([r["pred_vt"] for r in items])
            vsolv = v == ci
            out.append((f"dm/{dom}", float(vsolv.mean()),
                        float(((vsolv) & (vt != ci)).sum() / max(vsolv.sum(), 1))))
    pa = RT / f"data/eval/t8_assembled/{key}.jsonl"
    if pa.exists():
        rs = [json.loads(l) for l in pa.read_text().splitlines() if l.strip()]
        ci = np.array([r["correct_index"] for r in rs]); lab = np.array([r["label"] for r in rs])
        v = np.array([r["pred_v"] for r in rs]); vt = np.array([r["pred_vt"] for r in rs])
        m = lab == "vision"; vsolv = m & (v == ci)
        out.append(("assembled/aokvqa", float((v[m] == ci[m]).mean()),
                    float((vsolv & (vt != ci)).sum() / max(vsolv.sum(), 1))))
    return out


def preservation(key):
    p = RT / f"data/eval/t8_assembled/{key}.jsonl"
    if not p.exists():
        return None
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rs]); lab = np.array([r["label"] for r in rs])
    vt = np.array([r["pred_vt"] for r in rs]); t = np.array([r["pred_t"] for r in rs])
    V, T = lab == "vision", lab == "text"
    return {"aokvqa_vt": float((vt[V] == ci[V]).mean()),
            "race_t": float((t[T] == ci[T]).mean()),
            "race_vt": float((vt[T] == ci[T]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True); ap.add_argument("--base", default="qwen")
    args = ap.parse_args()

    pts = json.loads((RT / "data/diagnostics/grounding_strength.json").read_text())
    gv = np.array([p["ground"] for p in pts if p["modality"] == "vision"])
    dv = np.array([p["distr"] for p in pts if p["modality"] == "vision"])
    b, a = np.polyfit(gv, dv, 1)
    print(f"8-model vision law: v-distr = {a:.3f} + {b:.3f}*grounding\n")
    tids = test_ids()

    stats = {}
    for key in (args.base, args.ft):
        cells = vision_cells(key, tids)
        if not cells:
            print(f"[{key}] no eval found (run scripts/61 --key {key} --adapter ...)"); continue
        g = np.array([c[1] for c in cells]); d = np.array([c[2] for c in cells])
        resid = d - (a + b * g); stats[key] = (float(d.mean()), float(resid.mean()))
        print(f"[{'BASE' if key == args.base else 'FT  '} {key}]  mean grounding={g.mean():.3f}  "
              f"mean v-distraction={d.mean():.3f}  mean law-residual={resid.mean():+.3f}")
        for (name, gg, dd), r in sorted(zip(cells, resid)):
            print(f"     {name:20} ground={gg:.3f}  v-distr={dd:.3f}  residual={r:+.3f}")

    pb, pf = preservation(args.base), preservation(args.ft)
    if pb and pf:
        print("\n=== preservation (base -> FT, want |Δ| <~ 0.02) ===")
        for k, lbl in [("aokvqa_vt", "general multimodal (A-OKVQA V+T)"),
                       ("race_t", "text-only (RACE T-only)"), ("race_vt", "caption-use (RACE V+T)")]:
            print(f"  {lbl:34} {pb[k]:.3f} -> {pf[k]:.3f}  ({pf[k]-pb[k]:+.3f})")
        if args.ft in stats and args.base in stats:
            (dvb, _), (dvf, rf) = stats[args.base], stats[args.ft]
            beat = rf < -0.02; pres_ok = all(pf[k] - pb[k] > -0.025 for k in pb)
            print(f"\nVERDICT: v-distraction {dvb:.3f}->{dvf:.3f}; FT law-residual {rf:+.3f} "
                  f"({'BELOW law = decoupled' if beat else 'on/above law = trivial'}); "
                  f"preservation {'OK' if pres_ok else 'REGRESSED'}")


if __name__ == "__main__":
    main()
