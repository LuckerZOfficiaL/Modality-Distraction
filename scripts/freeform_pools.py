"""Free-form vs MCQ, base vs robustness task vector, on ALL THREE reporting pools.

Pools and their MCQ counterparts (same filters the paper uses):
  test       data/eval/t8/{tag}_w0.0.jsonl                  , seeds {tag}{,_s1,_s2,_s3}_w0.5
  human      data/eval/t8/hm3_{tag}_w0.0.jsonl              , seeds hm3_{tag}_s{i}_w0.5
  assembled  data/eval/t8_assembled/{tag}_w0.0.jsonl        , seeds {tag}{,_s1,_s2,_s3}_w0.5
             (held-out split only, minus caption-certification violations)

Own-set conditioning throughout, first-wins dedup, w=0.5, 4-seed mean on the intervention and a
single deterministic run on the base, exactly as in the paper.

    python scripts/freeform_pools.py [--pools test human assembled]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("sc", ROOT / "scripts/freeform_score.py")
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

COHORT = [("qwen2.5-vl-7b", "qwen7b", "Qwen2.5-VL-7B"), ("internvl3-8b", "internvl", "InternVL3-8B"),
          ("qwen2.5-vl-3b", "qwen3b", "Qwen2.5-VL-3B"), ("llava-onevision-7b", "llavaov", "LLaVA-OV-7B"),
          ("llavanext", "next", "LLaVA-NeXT-8B"), ("qwen2-vl-2b", "qwen2b", "Qwen2-VL-2B"),
          ("llava-1.5-7b", "llava15", "LLaVA-1.5-7B")]
HELD = set(json.loads((ROOT / "data/dm/merged_aokvqa_racehigh/"
                       "assembled_dev_heldout_split.json").read_text())["heldout"])
VIOL = set(json.loads((ROOT / "data/diagnostics/"
                       "assembled_caption_certification.json").read_text())["violations"])
_ENC = None


def enc():
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _ENC


def ff_keys(pool, stem, tag, side):
    """(base key, [seed keys]) under data/eval/freeform/ for this pool and label side.

    The vision side of the test pool predates the pool/label naming scheme, so it keeps its
    original file names; everything else is {prefix}_{tag}_...{_T for text}.
    """
    if pool == "test" and side == "vision":
        return stem, [f"{tag}_tv_w0.5_s{i}" for i in range(4)]
    pre = {"test": "tst", "human": "hm3", "assembled": "asm"}[pool]
    sfx = "_T" if side == "text" else ""
    return f"{pre}_{tag}_base{sfx}", [f"{pre}_{tag}_tv_w0.5_s{i}{sfx}" for i in range(4)]


def mcq_files(pool, tag):
    if pool == "test":
        return (f"eval/t8/{tag}_w0.0.jsonl",
                [f"eval/t8/{tag}{s}_w0.5.jsonl" for s in ("", "_s1", "_s2", "_s3")])
    if pool == "human":
        return (f"eval/t8/hm3_{tag}_w0.0.jsonl",
                [f"eval/t8/hm3_{tag}_s{i}_w0.5.jsonl" for i in range(4)])
    return (f"eval/t8_assembled/{tag}_w0.0.jsonl",
            [f"eval/t8_assembled/{tag}{s}_w0.5.jsonl" for s in ("", "_s1", "_s2", "_s3")])


def load_ff(key):
    fp = ROOT / f"data/eval/freeform/{key}.jsonl"
    if not fp.exists():
        return None
    seen = {}
    for l in fp.read_text().splitlines():
        if l.strip():
            r = json.loads(l); seen.setdefault(r["candidate_id"], r)
    return list(seen.values())


CONDS = ("gen_v", "gen_t", "gen_vt")


def project(rows):
    """Induce a choice for every answer. One batched encode for the whole file: the per-row
    version needed hours on the assembled pool, which has ~2,400 items per cell."""
    need, texts = [], []                       # only non-numeric cases reach the encoder
    for r in rows:
        onum = [sc.to_num(o) for o in r["options"]]
        r["_onum"] = onum
        r["_num_item"] = all(x is not None for x in onum)
        for cond in CONDS:
            a = r[cond] or ""
            if sc.REFUSE.search(a):
                r["p_" + cond] = -1; continue
            an = sc.to_num(a)
            if r["_num_item"] and an is not None:
                r["p_" + cond] = int(np.argmin([abs(an - x) / max(abs(x), 1e-9) for x in onum]))
            else:
                need.append((r, cond, len(texts))); texts.append(a)
                r["_optoff"] = None
    if not need:
        return rows
    opt_off = {}
    for r, _, _ in need:
        if id(r) not in opt_off:
            opt_off[id(r)] = len(texts); texts.extend(r["options"])
    E = enc().encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    for r, cond, idx in need:
        o = opt_off[id(r)]
        O = E[o:o + len(r["options"])]
        sims = O @ E[idx]
        j = int(sims.argmax())
        r["p_" + cond] = j if float(sims[j]) >= 0.25 else -1
    return rows


def vd_ff(rows, side):
    """v-distraction conditions on V-only correct, t-distraction on T-only correct."""
    cond = "p_gen_v" if side == "vision" else "p_gen_t"
    s = [r for r in rows if r[cond] == r["correct_index"]]
    return float(np.mean([r["p_gen_vt"] != r["correct_index"] for r in s])) if s else float("nan")


def vd_mcq(rel, pool, ids, side="vision"):
    fp = ROOT / "data" / rel
    if not fp.exists():
        return None
    seen = {}
    for l in fp.read_text().splitlines():
        if l.strip():
            m = json.loads(l)
            if m["candidate_id"] in ids and m["candidate_id"] not in seen:
                if pool == "assembled" and (m["candidate_id"] not in HELD
                                            or m["candidate_id"] in VIOL):
                    continue
                seen[m["candidate_id"]] = m
    key = "pred_v" if side == "vision" else "pred_t"
    s = [m for m in seen.values() if m["label"] == side and m[key] == m["correct_index"]]
    return float(np.mean([m["pred_vt"] != m["correct_index"] for m in s])) if s else None


def run_pool(pool, out):
    for side, metric in (("vision", "v-distraction"), ("text", "t-distraction")):
        print(f"\n===== pool: {pool}   |   {metric} ({side} items) =====")
        print(f"{'Backbone':16} {'n':>5} {'FF base':>8} {'FF TV':>7} {'FF red':>7} | "
              f"{'MCQ base':>9} {'MCQ TV':>7} {'MCQ red':>8}  cells")
        print("-" * 82)
        rows_ok = []
        for stem, tag, disp in COHORT:
            bkey, skeys = ff_keys(pool, stem, tag, side)
            base = load_ff(bkey)
            if base is None:
                print(f"{disp:16} (missing {bkey})"); continue
            n = len(base)
            b_ff = vd_ff(project(base), side)
            tv = [vd_ff(project(r), side) for r in (load_ff(k) for k in skeys) if r is not None]
            if not tv:
                print(f"{disp:16} {n:5d} {b_ff:8.3f} {'--':>7} {'--':>7} | (no TV cells)"); continue
            t_ff = float(np.mean(tv))
            ids = {r["candidate_id"] for r in base}
            bf, sf = mcq_files(pool, tag)
            b_mc = vd_mcq(bf, pool, ids, side)
            mcs = [v for v in (vd_mcq(f, pool, ids, side) for f in sf) if v is not None]
            t_mc = float(np.mean(mcs)) if mcs else None
            r_ff = 100 * (1 - t_ff / b_ff) if b_ff else float("nan")
            r_mc = 100 * (1 - t_mc / b_mc) if (b_mc and t_mc is not None) else float("nan")
            print(f"{disp:16} {n:5d} {b_ff:8.3f} {t_ff:7.3f} {r_ff:6.0f}% | "
                  f"{(b_mc if b_mc is not None else float('nan')):9.3f} "
                  f"{(t_mc if t_mc is not None else float('nan')):7.3f} {r_mc:7.0f}%  {len(tv)}/4")
            rows_ok.append((b_ff, t_ff, b_mc, t_mc))
            out.setdefault(pool, {}).setdefault(side, {})[disp] = {
                "n": n, "ff_base": b_ff, "ff_tv": t_ff, "ff_red_pct": r_ff, "ff_seeds": tv,
                "mcq_base": b_mc, "mcq_tv": t_mc, "mcq_red_pct": r_mc}
        if rows_ok:
            a = np.array([[x if x is not None else np.nan for x in r] for r in rows_ok], dtype=float)
            mf = 100 * (1 - np.nanmean(a[:, 1]) / np.nanmean(a[:, 0]))
            mm = 100 * (1 - np.nanmean(a[:, 3]) / np.nanmean(a[:, 2]))
            print("-" * 82)
            print(f"{'Mean':16} {'':5} {np.nanmean(a[:, 0]):8.3f} {np.nanmean(a[:, 1]):7.3f} "
                  f"{mf:6.0f}% | {np.nanmean(a[:, 2]):9.3f} {np.nanmean(a[:, 3]):7.3f} {mm:7.0f}%")
            print(f"{'':16} free-form falls on {int((a[:, 1] < a[:, 0]).sum())}/{len(rows_ok)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["test", "human", "assembled"])
    args = ap.parse_args()
    out = {}
    for p in args.pools:
        run_pool(p, out)
    json.dump(out, open(ROOT / "data/diagnostics/freeform_pools_summary.json", "w"), indent=1)
    print("\nwrote data/diagnostics/freeform_pools_summary.json")


if __name__ == "__main__":
    main()
