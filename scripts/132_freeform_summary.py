"""Consolidate the free-form pilot across the seven paper backbones.

Two outputs:
  1. the paired free-form vs MCQ table (same items, same model, only the answer format differs)
  2. the pooled provenance test: when the caption names one of the wrong options, does the
     free-form answer land on THAT option more often than chance among the distractors?

Everything is recomputed from data/eval/freeform/*.jsonl, never read from a printed number.

    python scripts/132_freeform_summary.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("sc", ROOT / "scripts/131_freeform_score.py")
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

# (free-form key, MCQ tag, display name) in the paper's row order
COHORT = [("qwen2.5-vl-7b", "qwen7b", "Qwen2.5-VL-7B"),
          ("internvl3-8b", "internvl", "InternVL3-8B"),
          ("qwen2.5-vl-3b", "qwen3b", "Qwen2.5-VL-3B"),
          ("llava-onevision-7b", "llavaov", "LLaVA-OV-7B"),
          ("llavanext", "next", "LLaVA-NeXT-8B"),
          ("qwen2-vl-2b", "qwen2b", "Qwen2-VL-2B"),
          ("llava-1.5-7b", "llava15", "LLaVA-1.5-7B")]


def load_ff(key):
    fp = ROOT / f"data/eval/freeform/{key}.jsonl"
    if not fp.exists():
        return None
    seen = {}
    for l in fp.read_text().splitlines():
        if l.strip():
            r = json.loads(l); seen.setdefault(r["candidate_id"], r)
    return list(seen.values())


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    rows_out, pooled = [], {"obs": 0, "exp": 0.0, "n": 0, "flips": 0}
    for key, tag, disp in COHORT:
        rows = load_ff(key)
        if not rows or len(rows) < 420:
            rows_out.append((disp, None, f"incomplete ({0 if not rows else len(rows)}/420)"))
            continue

        for r in rows:                                    # project each answer onto the options
            O = enc.encode(r["options"], normalize_embeddings=True)
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
                    e = enc.encode([a], normalize_embeddings=True)[0]
                    s = O @ e; j = int(s.argmax())
                    r["p_" + cond] = j if float(s[j]) >= 0.25 else -1

        mcq, seen = {}, set()
        for l in (ROOT / f"data/eval/t8/{tag}_w0.0.jsonl").read_text().splitlines():
            if l.strip():
                m = json.loads(l)
                if m["candidate_id"] not in seen:
                    seen.add(m["candidate_id"]); mcq[m["candidate_id"]] = m

        common = [r for r in rows if r["candidate_id"] in mcq
                  and r["p_gen_v"] == r["correct_index"]
                  and mcq[r["candidate_id"]]["pred_v"] == mcq[r["candidate_id"]]["correct_index"]]
        ff = [r["p_gen_vt"] != r["correct_index"] for r in common]
        mc = [mcq[r["candidate_id"]]["pred_vt"] != mcq[r["candidate_id"]]["correct_index"]
              for r in common]
        b = sum(1 for f, m in zip(ff, mc) if f and not m)
        c = sum(1 for f, m in zip(ff, mc) if m and not f)
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
        rows_out.append((disp, dict(n=len(common), ff=float(np.mean(ff)), mcq=float(np.mean(mc)),
                                    b=b, c=c, p=float(p)), None))

        # provenance, on this model's flipped items
        flips = [r for r in rows if r["p_gen_v"] == r["correct_index"]
                 and r["p_gen_vt"] != r["correct_index"] and r["p_gen_vt"] != -1]
        pooled["flips"] += len(flips)
        for r in flips:
            dis = [i for i in range(len(r["options"])) if i != r["correct_index"]]
            planted = [i for i in dis if sc.strict_match(r["caption"], r["options"][i])]
            if not planted:
                continue
            pooled["n"] += 1
            pooled["obs"] += int(r["p_gen_vt"] in planted)
            pooled["exp"] += len(planted) / len(dis)

    print(f"{'Backbone':16} {'n':>5} {'FF':>7} {'MCQ':>7} {'FF-only':>8} {'MCQ-only':>9} {'p':>8}")
    print("-" * 64)
    ok = []
    for disp, d, note in rows_out:
        if d is None:
            print(f"{disp:16} {note}"); continue
        ok.append(d)
        print(f"{disp:16} {d['n']:5d} {d['ff']:7.3f} {d['mcq']:7.3f} {d['b']:8d} {d['c']:9d} {d['p']:8.4f}")
    if ok:
        print("-" * 64)
        print(f"{'Mean':16} {'':5} {np.mean([d['ff'] for d in ok]):7.3f} "
              f"{np.mean([d['mcq'] for d in ok]):7.3f}   "
              f"({sum(d['ff'] > d['mcq'] for d in ok)}/{len(ok)} backbones higher free-form, "
              f"{sum(d['ff'] < d['mcq'] for d in ok)} lower)")
        # sign test over backbones on the paired difference
        pos = sum(d['ff'] > d['mcq'] for d in ok); neg = sum(d['ff'] < d['mcq'] for d in ok)
        if pos + neg:
            print(f"{'':16} sign test over backbones: p="
                  f"{binomtest(pos, pos + neg, 0.5, alternative='greater').pvalue:.4f}")

    if pooled["n"]:
        ch = pooled["exp"] / pooled["n"]
        pv = binomtest(pooled["obs"], pooled["n"], ch, alternative="greater").pvalue
        print(f"\nPROVENANCE (pooled over backbones): of {pooled['flips']} flipped items, "
              f"{pooled['n']} have a caption-planted distractor; the free-form answer lands on a "
              f"planted one {pooled['obs']}/{pooled['n']} = {pooled['obs'] / pooled['n']:.2f}, "
              f"chance {ch:.2f}, binomial p={pv:.4f}")

    json.dump({"paired": {d: v for (d, v, _) in rows_out if v}, "provenance": pooled},
              open(ROOT / "data/diagnostics/freeform_pilot_summary.json", "w"), indent=1)
    print("\nwrote data/diagnostics/freeform_pilot_summary.json")


if __name__ == "__main__":
    main()
