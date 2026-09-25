#!/usr/bin/env python3
"""Dev-side readout for mitigation-improvement screening (Tier 0/1/2).

Reports the three SELECTION-side axes only -- MoGround val, assembled dev half, capability rows
0-400 -- for any set of run keys, next to the in-draft incumbent (4 seeds). Never touches the
reporting side (test split / assembled held-out / capability rows 400-800).

    python scripts/tier_readout.py qwen2b_t2_vteach_w0.5 qwen2b_t2_vteach2_w0.5
    python scripts/tier_readout.py --all-tiers
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
BEN = ["mmstar", "mmbench", "seedbench", "scienceqa", "naturalbench"]
VAL = {json.loads(l)["candidate_id"]
       for l in (DR / "dm/multidomain_v1/splits/val.jsonl").read_text().splitlines() if l.strip()}
_S = json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())
ASM_DEV = set(_S["dev"])
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
TAGS7 = ["qwen7b", "internvl", "qwen3b", "llavaov", "next", "qwen2b", "llava15"]


def inc_keys(tag="qwen2b"):
    """The in-draft method's 4 adapter seeds at w=0.5 for one backbone (dev-side incumbent)."""
    return [f"{tag}_w0.5"] + [f"{tag}_s{s}_w0.5" for s in (1, 2, 3)]


INCUMBENT = inc_keys()


def vd(fp, keep, cert=False):
    """v-distraction = P(V+T wrong | V-only correct) on vision-grounded items."""
    seen = {}  # the pools carry duplicate candidate_ids; keep the first occurrence
    for line in Path(fp).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen.setdefault(r["candidate_id"], r)
    rows = [r for r in seen.values() if r["candidate_id"] in keep]
    if cert:
        rows = [r for r in rows if r["candidate_id"] not in VIOL]
    rows = [r for r in rows if r["label"] == "vision"]
    if not rows:
        return float("nan")
    ci = np.array([r["correct_index"] for r in rows])
    vt = np.array([r["pred_vt"] for r in rows]); v = np.array([r["pred_v"] for r in rows])
    ok = v == ci
    return float((ok & (vt != ci)).sum() / max(ok.sum(), 1))


def _cov(fp, ids):
    """Fraction of expected ids present -- an eval still in flight has a partial file."""
    if not Path(fp).exists():
        return 0.0
    got = {json.loads(l)["candidate_id"] for l in Path(fp).read_text().splitlines() if l.strip()}
    return len(got & ids) / len(ids)


def axes(key, min_cov=0.98):
    """(val v-distr, assembled-dev v-distr, dev capability); nan unless the eval is COMPLETE.

    A running eval appends rows, so scoring a partial file yields nonsense (a half-written
    assembled pass once read as v-distraction 0.25). Every axis is gated on id coverage.
    """
    fd = DR / f"eval/t8/{key}.jsonl"
    fa = DR / f"eval/t8_assembled/{key}_asmdev.jsonl"
    if not fa.exists():                       # incumbent rows live in the full assembled file
        fa = DR / f"eval/t8_assembled/{key}.jsonl"
    caps = [DR / f"diagnostics/capability/{b}_phase_{key}.json" for b in BEN]
    cv, ca = _cov(fd, VAL), _cov(fa, ASM_DEV)
    return (vd(fd, VAL) if cv >= min_cov else float("nan"),
            vd(fa, ASM_DEV, cert=True) if ca >= min_cov else float("nan"),
            float(np.mean([json.loads(c.read_text())["acc"] for c in caps])) if all(c.exists() for c in caps)
            else float("nan"),
            (cv, ca, sum(c.exists() for c in caps) / len(BEN)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--all-tiers", action="store_true", help="every t1_*/t2_* key found on disk")
    ap.add_argument("--confirm7", metavar="W", help="7-model confirmation at strength W: each "
                    "backbone's t2 cell vs ITS OWN 4-seed incumbent")
    a = ap.parse_args()
    keys = list(a.keys)
    if a.all_tiers:
        keys += sorted({p.stem for p in (DR / "eval/t8").glob("*_t[12]_*.jsonl")} - set(keys))
    if not keys and not a.confirm7:
        print("no keys given"); return 1

    if a.confirm7:
        W = a.confirm7
        print(f"7-MODEL DEV-SIDE CONFIRMATION at w={W} (val / assembled-dev / capability rows 0-400)")
        print(f"\n{'backbone':11}{'val t2':>9}{'val inc':>9}{'asm t2':>9}{'asm inc':>9}"
              f"{'cap t2':>9}{'cap inc':>9}   dominates?")
        print("-" * 92)
        n_dom = 0
        for tag in TAGS7:
            iv = np.array([axes(k)[:3] for k in inc_keys(tag)])
            im, isd = np.nanmean(iv, 0), np.nanstd(iv, 0)
            v, sc, c, cov = axes(f"{tag}_t2_vteach2_w{W}")
            if np.isnan([v, sc, c]).all():
                print(f"{tag:11}{'-- not evaluated --':>30}   coverage "
                      f"{cov[0]:.0%}/{cov[1]:.0%}/{cov[2]:.0%}"); continue
            dom = ((not np.isnan(v)) and v <= im[0] and (not np.isnan(sc)) and sc <= im[1]
                   and (not np.isnan(c)) and c >= im[2])
            n_dom += dom
            f = lambda x: "  partial" if np.isnan(x) else f"{x:>9.4f}"
            print(f"{tag:11}{f(v)}{im[0]:>9.4f}{f(sc)}{im[1]:>9.4f}{f(c)}{im[2]:>9.4f}   "
                  + ("YES" if dom else "no"))
        print(f"\ndominates on {n_dom}/{len(TAGS7)} backbones "
              f"(all three axes at least as good as that backbone's own 4-seed incumbent)")
        return 0

    inc = np.array([axes(k)[:3] for k in INCUMBENT])
    print("SELECTION SIDE ONLY (MoGround val / assembled dev half / capability rows 0-400)")
    print(f"\n{'run':30}{'val v-distr':>13}{'asm-dev':>10}{'cap':>9}   verdict vs incumbent")
    print("-" * 88)
    m, sd = np.nanmean(inc, 0), np.nanstd(inc, 0)
    print(f"{'INCUMBENT (in-draft, 4 seeds)':30}{m[0]:>13.4f}{m[1]:>10.4f}{m[2]:>9.4f}   "
          f"(sd {sd[0]:.4f}/{sd[1]:.4f}/{sd[2]:.4f})")
    for k in keys:
        v, s, c, cov = axes(k)
        if np.isnan([v, s, c]).all():
            print(f"{k:30}{'-- incomplete --':>27}   coverage val/asm/cap "
                  f"{cov[0]:.0%}/{cov[1]:.0%}/{cov[2]:.0%}"); continue
        # "wins" = better than incumbent mean by more than its own seed sd on that axis
        wins = [(not np.isnan(x)) and (x < m[i] - sd[i] if i < 2 else x > m[i] + sd[i])
                for i, x in enumerate((v, s, c))]
        loses = [(not np.isnan(x)) and (x > m[i] + sd[i] if i < 2 else x < m[i] - sd[i])
                 for i, x in enumerate((v, s, c))]
        tag = ("BEATS on " + "/".join(n for n, w in zip(("val", "asm", "cap"), wins) if w)) if any(wins) else "no win"
        if any(loses):
            tag += "; WORSE on " + "/".join(n for n, l in zip(("val", "asm", "cap"), loses) if l)
        cells = [f"{x:>13.4f}" if not np.isnan(x) else f"{'(partial)':>13}" for x in (v,)]
        cells += [f"{x:>10.4f}" if not np.isnan(x) else f"{'(partial)':>10}" for x in (s,)]
        cells += [f"{x:>9.4f}" if not np.isnan(x) else f"{'(partial)':>9}" for x in (c,)]
        print(f"{k:30}{''.join(cells)}   {tag}"
              + ("" if min(cov) >= 0.98 else f"  [coverage {cov[0]:.0%}/{cov[1]:.0%}/{cov[2]:.0%}]"))
    print("\nlower is better for the two distraction axes; higher is better for capability.")
    print("A 1-seed screen must clear the incumbent's 4-seed sd to count as a real move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
