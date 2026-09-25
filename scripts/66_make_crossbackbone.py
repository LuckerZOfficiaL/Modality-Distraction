"""Step 66: generate the 8-model x 2-pool mechanism tables (tab:crossbackbone[_assembled]) from
real results -- reproducible, mirrors scripts/62. Sources:
  T2  : recomputed from cf-stores (cheap norms; same formula as scripts/53)
  X2  : data/diagnostics/x2/{key}[_assembled].json   (peak raw-residual AUROC, from scripts/46)
  X4  : data/diagnostics/logit_lens/{key}[_assembled].json (commit layer + probs, from scripts/45)
  X1  : data/diagnostics/grounding_strength.json     (D_M nat:non v-distraction ratio; D_M only)

    python scripts/66_make_crossbackbone.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"; TAB = ROOT / "paper" / "tables"
KEYS = ["qwen", "Qwen_Qwen2.5-VL-7B-Instruct", "Qwen_Qwen2-VL-2B-Instruct", "OpenGVLab_InternVL3-8B-hf",
        "llava-hf_llava-1.5-7b-hf", "llava-hf_llava-onevision-qwen2-7b-ov-hf", "llavanext"]
NAME = {"qwen": "Qwen2.5-VL-3B", "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen_Qwen2-VL-2B-Instruct": "Qwen2-VL-2B", "OpenGVLab_InternVL3-8B-hf": "InternVL3-8B",
        "llava-hf_llava-1.5-7b-hf": "LLaVA-1.5-7B", "llava-hf_llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B",
        "llavanext": "LLaVA-NeXT-8B"}
SUB = {"dm": "dm_multidomain_v1_cf", "assembled": "merged_aokvqa_racehigh_cf"}


def t2(key, sub):
    st = DR / "activations" / key / sub
    acts = np.load(st / "activations.npy", mmap_mode="r"); ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    isV, isT = lab == "vision", lab == "text"

    def relp(mask, cond):
        vt = acts[mask, 0].astype(np.float32); o = acts[mask, cond].astype(np.float32)
        return (np.linalg.norm(vt - o, axis=2) / (np.linalg.norm(vt, axis=2) + 1e-6)).mean(0)

    vp, tp = relp(isV, 1).max(), relp(isT, 2).max()

    def acc(mask, cond):
        return float((ll[mask, cond].argmax(1) == ci[mask]).mean())
    return {"ratio": float(tp / max(vp, 1e-6)), "dvis": acc(isV, 0) - acc(isV, 1),
            "dtext": acc(isT, 0) - acc(isT, 2)}


def x2(key, tag):
    return json.loads((DR / "diagnostics/x2" / f"{key}{tag}.json").read_text())


def x4(key, tag):
    j = json.loads((DR / "diagnostics/logit_lens" / f"{key}{tag}.json").read_text())
    nl = len(j["robust_p_correct"]); c = j["commit_layer"]
    return {"depth": 100 * c / (nl - 1), "rob": j["robust_p_correct"][c], "dwrong": j["distr_p_final"][c]}


def x1_natnon():
    pts = json.loads((DR / "diagnostics/grounding_strength.json").read_text())
    out = {}
    for k in KEYS:
        dd = {p["domain"]: p["distr"] for p in pts
              if p["model"] == k and p["dataset"] == "dm" and p["modality"] == "vision"}
        if dd.get("dci", 0) > 0:
            out[k] = float(np.mean([dd[x] for x in ("vistext", "semart", "roco") if x in dd]) / dd["dci"])
    return out


def order_by_grounding():  # most-grounded first (mean D_M V-only vision accuracy), like the leaderboard
    pts = json.loads((DR / "diagnostics/grounding_strength.json").read_text())
    g = {k: np.mean([p["ground"] for p in pts if p["model"] == k and p["dataset"] == "dm" and p["modality"] == "vision"])
         for k in KEYS}
    return sorted(KEYS, key=lambda k: -g.get(k, 0))


DM_HEAD = r"""\begin{tabular}{l|c|cc|c|cc}
\toprule
 & Domain & \multicolumn{2}{c|}{Perturbation} & Decodability & \multicolumn{2}{c}{Commit (logit-lens)} \\
model & nat:non & ratio & vision harm & AUROC & depth & robust / distr.\ wrong \\
\midrule
"""
ASM_HEAD = r"""\begin{tabular}{l|ccc|c|cc}
\toprule
 & \multicolumn{3}{c|}{Perturbation} & Decodability & \multicolumn{2}{c}{Commit (logit-lens)} \\
model & ratio & vision harm & text harm & AUROC & depth & robust / distr.\ wrong \\
\midrule
"""
FOOT = "\\bottomrule\n\\end{tabular}\n"


def main():
    order = order_by_grounding()
    nn = x1_natnon()
    dm_rows, asm_rows = [], []
    for k in order:
        td, ta = t2(k, SUB["dm"]), t2(k, SUB["assembled"])
        xd, xa = x2(k, ""), x2(k, "_assembled")
        cd, ca = x4(k, ""), x4(k, "_assembled")
        dm_rows.append(f"{NAME[k]} & ${nn[k]:.2f}\\times$ & ${td['ratio']:.2f}\\times$ & ${td['dvis']:+.3f}$ & "
                       f"{xd['peak_auroc']:.2f} (L{xd['peak_layer']}) & {cd['depth']:.0f}\\% & "
                       f"{cd['rob']:.2f} / {cd['dwrong']:.2f} \\\\")
        asm_rows.append(f"{NAME[k]} & ${ta['ratio']:.2f}\\times$ & ${ta['dvis']:+.3f}$ & ${ta['dtext']:+.3f}$ & "
                        f"{xa['peak_auroc']:.2f} (L{xa['peak_layer']}) & {ca['depth']:.0f}\\% & "
                        f"{ca['rob']:.2f} / {ca['dwrong']:.2f} \\\\")
    (TAB / "crossbackbone.tex").write_text(DM_HEAD + "\n".join(dm_rows) + "\n" + FOOT)
    (TAB / "crossbackbone_assembled.tex").write_text(ASM_HEAD + "\n".join(asm_rows) + "\n" + FOOT)
    print("wrote crossbackbone.tex + crossbackbone_assembled.tex")
    print("order:", [NAME[k] for k in order])


if __name__ == "__main__":
    main()
