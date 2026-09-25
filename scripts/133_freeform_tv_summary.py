"""Does the robustness task vector, trained on MULTIPLE-CHOICE items, reduce distraction in
FREE-FORM generation?

Compares, per backbone, on the 420 certified vision test items:
  free-form v-distraction of the base model      (deterministic, one run, as in the paper)
  free-form v-distraction with the task vector   (w=0.5, mean over the paper's 4 adapter seeds)
  the same contrast under MCQ, read from data/eval/t8/ (base vs the same 4 seeds)

Conventions follow the paper exactly: own-set conditioning (each arm is scored on the items IT
solves from the image alone), first-wins dedup, w=0.5 default, 4-seed mean.

    python scripts/133_freeform_tv_summary.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("sc", ROOT / "scripts/131_freeform_score.py")
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

COHORT = [("qwen2.5-vl-7b", "qwen7b", "Qwen2.5-VL-7B"),
          ("internvl3-8b", "internvl", "InternVL3-8B"),
          ("qwen2.5-vl-3b", "qwen3b", "Qwen2.5-VL-3B"),
          ("llava-onevision-7b", "llavaov", "LLaVA-OV-7B"),
          ("llavanext", "next", "LLaVA-NeXT-8B"),
          ("qwen2-vl-2b", "qwen2b", "Qwen2-VL-2B"),
          ("llava-1.5-7b", "llava15", "LLaVA-1.5-7B")]
SEEDS = (0, 1, 2, 3)
_ENC = None


def enc():
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _ENC


def load(stem):
    fp = ROOT / f"data/eval/freeform/{stem}.jsonl"
    if not fp.exists():
        return None
    seen = {}
    for l in fp.read_text().splitlines():
        if l.strip():
            r = json.loads(l); seen.setdefault(r["candidate_id"], r)
    return list(seen.values()) if len(seen) >= 420 else None


def project(rows):
    """Turn each free-form answer into an induced choice over the item's own options."""
    for r in rows:
        O = enc().encode(r["options"], normalize_embeddings=True)
        onum = [sc.to_num(o) for o in r["options"]]
        num_item = all(x is not None for x in onum)
        for cond in ("gen_v", "gen_vt"):
            a = r[cond] or ""
            if sc.REFUSE.search(a):
                r["p_" + cond] = -1; continue
            an = sc.to_num(a)
            if num_item and an is not None:
                r["p_" + cond] = int(np.argmin([abs(an - x) / max(abs(x), 1e-9) for x in onum]))
            else:
                e = enc().encode([a], normalize_embeddings=True)[0]
                s = O @ e; j = int(s.argmax())
                r["p_" + cond] = j if float(s[j]) >= 0.25 else -1
    return rows


def vdistr_ff(rows):
    s = [r for r in rows if r["p_gen_v"] == r["correct_index"]]
    return float(np.mean([r["p_gen_vt"] != r["correct_index"] for r in s])), len(s)


def vdistr_mcq(tag, suffix, ids):
    fp = ROOT / f"data/eval/t8/{tag}{suffix}.jsonl"
    if not fp.exists():
        return None, 0
    seen = {}
    for l in fp.read_text().splitlines():
        if l.strip():
            m = json.loads(l)
            if m["candidate_id"] in ids:
                seen.setdefault(m["candidate_id"], m)
    s = [m for m in seen.values() if m["pred_v"] == m["correct_index"]]
    if not s:
        return None, 0
    return float(np.mean([m["pred_vt"] != m["correct_index"] for m in s])), len(s)


def main():
    print(f"{'Backbone':16} {'FF base':>8} {'FF TV':>7} {'FF red':>8} | "
          f"{'MCQ base':>9} {'MCQ TV':>8} {'MCQ red':>8}  seeds")
    print("-" * 78)
    out, ffb, fft, mcb, mct = {}, [], [], [], []
    for stem, tag, disp in COHORT:
        base = load(stem)
        if base is None:
            print(f"{disp:16} base incomplete"); continue
        base = project(base)
        ids = {r["candidate_id"] for r in base}
        b_ff, _ = vdistr_ff(base)

        tv_vals = []
        for s in SEEDS:
            rows = load(f"{tag}_tv_w0.5_s{s}")
            if rows is not None:
                tv_vals.append(vdistr_ff(project(rows))[0])
        if not tv_vals:
            print(f"{disp:16} {b_ff:8.3f} {'--':>7} {'--':>8} | (no task-vector cells yet)")
            continue
        t_ff = float(np.mean(tv_vals))

        b_mc, _ = vdistr_mcq(tag, "_w0.0", ids)
        mc_seeds = [v for v, _ in (vdistr_mcq(tag, sfx, ids) for sfx in
                                   ("_w0.5", "_s1_w0.5", "_s2_w0.5", "_s3_w0.5")) if v is not None]
        t_mc = float(np.mean(mc_seeds)) if mc_seeds else float("nan")

        r_ff = 100 * (1 - t_ff / b_ff) if b_ff else float("nan")
        r_mc = 100 * (1 - t_mc / b_mc) if b_mc else float("nan")
        print(f"{disp:16} {b_ff:8.3f} {t_ff:7.3f} {r_ff:7.0f}% | {b_mc:9.3f} {t_mc:8.3f} "
              f"{r_mc:7.0f}%  {len(tv_vals)}/4")
        ffb.append(b_ff); fft.append(t_ff); mcb.append(b_mc); mct.append(t_mc)
        out[disp] = {"ff_base": b_ff, "ff_tv": t_ff, "ff_red_pct": r_ff, "ff_seed_vals": tv_vals,
                     "mcq_base": b_mc, "mcq_tv": t_mc, "mcq_red_pct": r_mc, "n_seeds": len(tv_vals)}

    if out:
        print("-" * 78)
        mf = 100 * (1 - np.mean(fft) / np.mean(ffb)); mm = 100 * (1 - np.mean(mct) / np.mean(mcb))
        print(f"{'Mean':16} {np.mean(ffb):8.3f} {np.mean(fft):7.3f} {mf:7.0f}% | "
              f"{np.mean(mcb):9.3f} {np.mean(mct):8.3f} {mm:7.0f}%")
        print(f"{'':16} free-form falls on {sum(1 for v in out.values() if v['ff_tv'] < v['ff_base'])}"
              f"/{len(out)} backbones")
        json.dump(out, open(ROOT / "data/diagnostics/freeform_tv_summary.json", "w"), indent=1)
        print("\nwrote data/diagnostics/freeform_tv_summary.json")


if __name__ == "__main__":
    main()
