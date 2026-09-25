"""Regenerate paper/tables/{leaderboard,assembled}.tex from data/eval/t8[_assembled]/*.jsonl
so the cross-model tables always match the data. Re-run after a model sweep.

    python scripts/62_make_tables.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
DOMS = ["dci", "vistext", "semart", "roco"]
NAMES = {  # sanitized key -> display name
    "qwen": "Qwen2.5-VL-3B", "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "Qwen_Qwen2-VL-2B-Instruct": "Qwen2-VL-2B", "OpenGVLab_InternVL3-8B-hf": "InternVL3-8B",
    "llava-hf_llava-1.5-7b-hf": "LLaVA-1.5-7B",
    "llava-hf_llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B",
}


def rows_of(d):
    rs = [json.loads(l) for l in Path(d).read_text().splitlines() if l.strip()]
    return (np.array([r["correct_index"] for r in rs]), np.array([r["label"] for r in rs]),
            np.array([r["pred_vt"] for r in rs]), np.array([r["pred_v"] for r in rs]),
            np.array([r["pred_t"] for r in rs]), np.array([r.get("source") for r in rs]))


def vdistr(isV, ci, vt, v):
    s = isV & (v == ci); return (s & (vt != ci)).sum() / max(s.sum(), 1)


def disp(k):
    return NAMES.get(k, k.replace("_", "\\_"))


def leaderboard():
    out = []
    for f in glob.glob(str(DR / "eval/t8/*.jsonl")):
        k = Path(f).stem
        if k not in NAMES:  # skip stray generic-hf duplicates
            continue
        ci, lab, vt, v, t, src = rows_of(f)
        isV = lab == "vision"
        vtv = (vt[isV] == ci[isV]).mean()
        pd = {dm: vdistr(isV & (src == dm), ci, vt, v) for dm in DOMS}
        vall = vdistr(isV, ci, vt, v)
        isT = lab == "text"; ts = isT & (t == ci); tall = (ts & (vt != ci)).sum() / max(ts.sum(), 1)
        ratio = np.mean([pd[d] for d in DOMS[1:]]) / max(pd["dci"], 1e-9)
        out.append((vtv, k, pd, vall, tall, ratio))
    out.sort(key=lambda r: -r[0])
    body = "\n".join(
        f"{disp(k)} & {vtv:.3f} & {pd['dci']:.3f} & {pd['vistext']:.3f} & {pd['semart']:.3f} & "
        f"{pd['roco']:.3f} & {vall:.3f} & {tall:.3f} & ${ratio:.2f}\\times$ \\\\"
        for vtv, k, pd, vall, tall, ratio in out)
    tex = (r"""\begin{tabular}{l|c|cccc|cc|c}
\toprule
 & V+T & \multicolumn{4}{c|}{v-distraction by domain} & v-distr & t-distr & nat:non \\
model & (vis) & photos & charts & art & rad. & (all) & (all) & ratio \\
\midrule
""" + body + "\n" + r"""\bottomrule
\end{tabular}
""")
    (ROOT / "paper/tables/leaderboard.tex").write_text(tex)
    print(f"leaderboard: {len(out)} models")


def assembled():
    out = []
    for f in glob.glob(str(DR / "eval/t8_assembled/*.jsonl")):
        k = Path(f).stem
        if k not in NAMES:
            continue
        ci, lab, vt, v, t, src = rows_of(f)
        isV = lab == "vision"; isT = lab == "text"
        vg = (v[isV] == ci[isV]).mean(); tg = (t[isT] == ci[isT]).mean()
        vd = vdistr(isV, ci, vt, v); ts = isT & (t == ci); td = (ts & (vt != ci)).sum() / max(ts.sum(), 1)
        out.append((k, vg, tg, vd, td, vd - td))
    out.sort(key=lambda r: -(r[1] + r[2]) / 2)
    def arrow(a):
        return "v$>$t" if a > 0.005 else ("t$>$v" if a < -0.005 else "$\\approx$")
    body = "\n".join(
        f"{disp(k)} & {vg:.3f} & {tg:.3f} & ${tg-vg:+.3f}$ & {vd:.3f} & {td:.3f} & ${asym:+.3f}$ \\;({arrow(asym)}) \\\\"
        for k, vg, tg, vd, td, asym in out)
    tex = (r"""\begin{tabular}{l|cc|c|cc|c}
\toprule
model & V ground & T ground & gap (T$-$V) & v-distr & t-distr & asymmetry (v$-$t) \\
\midrule
""" + body + "\n" + r"""\bottomrule
\end{tabular}
""")
    (ROOT / "paper/tables/assembled.tex").write_text(tex)
    print(f"assembled: {len(out)} models")


if __name__ == "__main__":
    leaderboard()
    assembled()
