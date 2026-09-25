"""#3 Grounding-strength law: distraction susceptibility scales inversely with
grounding strength in the target modality.

Grounding strength (per backbone x domain x modality) = single-modality accuracy
(V-only for vision cells, T-only for text cells) -- "how well the model answers from
the grounding modality alone." Distraction = the distraction rate in that cell
(v-distraction for vision, t-distraction for text). The law predicts a NEGATIVE
relationship: weaker grounding -> more distractible. We pool all cells from D_M
(per-domain) and the ceiling-free assembled pool (A-OKVQA vision / RACE text), and
report the correlation + a scatter.

Offline: data/eval/t8/{model}.jsonl + data/eval/t8_assembled/{model}.jsonl.

    python scripts/54_grounding_strength.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOMS = ["dci", "vistext", "semart", "roco"]


# The assembled pool ships post-audit: the irrelevance judge flagged 239 VISION items whose
# retrieved caption asserts or contradicts an option, and every assembled statistic in the paper is
# computed on the audited subset (2261 vision + 2496 text = 4757). Eval files also carry a handful
# of duplicated candidate_ids from resumed runs, so dedupe before counting.
_VIOL = set(json.loads((ROOT / "data/diagnostics/assembled_caption_certification.json")
                       .read_text())["violations"])


def cells_from(path, by_source):
    seen, rs = set(), []
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cid = r["candidate_id"]
        if cid in seen or (not by_source and cid in _VIOL):
            continue
        seen.add(cid); rs.append(r)
    ci = np.array([r["correct_index"] for r in rs]); lab = np.array([r["label"] for r in rs])
    vt = np.array([r["pred_vt"] for r in rs]); v = np.array([r["pred_v"] for r in rs])
    t = np.array([r["pred_t"] for r in rs]); src = np.array([r.get("source") for r in rs])
    out = []
    groups = ([(d, "vision", src == d) for d in DOMS] if by_source
              else [("aokvqa", "vision", lab == "vision")])
    groups += ([(d, "text", src == d) for d in DOMS] if by_source
               else [("race", "text", lab == "text")])
    for name, mod, srcmask in groups:
        if mod == "vision":
            m = srcmask & (lab == "vision"); ground = (v[m] == ci[m]).mean() if m.sum() else np.nan
            solv = m & (v == ci); distr = (solv & (vt != ci)).sum() / max(solv.sum(), 1)
        else:
            m = srcmask & (lab == "text"); ground = (t[m] == ci[m]).mean() if m.sum() else np.nan
            solv = m & (t == ci); distr = (solv & (vt != ci)).sum() / max(solv.sum(), 1)
        if m.sum() >= 20:
            out.append({"domain": name, "modality": mod, "n": int(m.sum()),
                        "ground": float(ground), "distr": float(distr)})
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# The seven paper backbones, as the keys of the canonical (current-environment) base runs.
# These are the `_w0.0` files: the same evaluation the mitigation section uses for w=0, so every
# number in the paper traces to one run. The older un-suffixed keys are a stale June evaluation
# made under transformers 4.48; they differ for the Qwen family only (see scripts/110).
CANONICAL = ["internvl_w0.0", "qwen7b_w0.0", "qwen3b_w0.0", "qwen2b_w0.0",
             "llavaov_w0.0", "next_w0.0", "llava15_w0.0"]


def main() -> None:
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=",".join(CANONICAL),
                    help="comma list of eval keys; 'ALL' globs every file (legacy behaviour)")
    ap.add_argument("--out", default="data/diagnostics/grounding_strength.json",
                    help="output json, relative to the repo root. Override when adding a backbone "
                         "that is not yet part of the paper cohort, so the paper's own input file "
                         "(the default) is never rewritten.")
    args = ap.parse_args()
    pts = []
    keys = (sorted({Path(p).stem for p in glob.glob(str(ROOT / "data/eval/t8/*.jsonl"))}
                   | {Path(p).stem for p in glob.glob(str(ROOT / "data/eval/t8_assembled/*.jsonl"))})
            if args.keys == "ALL" else args.keys.split(","))
    print(f"{'model':28}{'dataset':10}{'domain':9}{'mod':7}{'n':>6}{'ground':>8}{'distr':>8}")
    for m in keys:
        for ds, path, by_src in [("dm", ROOT / f"data/eval/t8/{m}.jsonl", True),
                                 ("assembled", ROOT / f"data/eval/t8_assembled/{m}.jsonl", False)]:
            if not Path(path).exists():
                continue
            for c in cells_from(path, by_src):
                c.update(model=m, dataset=ds); pts.append(c)
                print(f"{m:28}{ds:10}{c['domain']:9}{c['modality']:7}{c['n']:>6}{c['ground']:>8.3f}{c['distr']:>8.3f}")

    g = np.array([p["ground"] for p in pts]); d = np.array([p["distr"] for p in pts])
    def report(mask, label):
        if mask.sum() < 3:
            print(f"  {label:34} n={int(mask.sum())} (too few)"); return
        pear = float(np.corrcoef(g[mask], d[mask])[0, 1]); spr = spearman(g[mask], d[mask])
        print(f"  {label:34} n={int(mask.sum()):>3}  Pearson r={pear:+.3f}  Spearman={spr:+.3f}")

    mod = np.array([p["modality"] for p in pts]); dsv = np.array([p["dataset"] for p in pts])
    print("\n=== grounding-strength law: corr(grounding, distraction) -- expect NEGATIVE ===")
    report(np.ones(len(pts), bool), "ALL cells")
    report(mod == "vision", "vision cells")
    report((mod == "text"), "text cells (incl. saturated D_M)")
    report((mod == "text") & (dsv == "assembled"), "text cells, assembled (ceiling-free)")
    report(dsv == "assembled", "assembled only (V+T, non-saturated)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pts, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
