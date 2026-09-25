r"""Self-audit: does OUR robustness task vector raise t-distraction?

Every load-bearing table in the paper reports the intervention's effect on v-distraction and on
general capability, but never on the text side. This script asks the symmetric question for all
seven backbones on the SELECTION side only (\dsname{} val split + assembled dev half), so nothing
here touches the reporting surfaces.

Per backbone and pool: t-distraction at w=0 (unsteered) and at the default w=0.5 (mean over four
adapter seeds), the change in percentage points, a paired item-level bootstrap 95% CI on that
change, and T-only accuracy at both strengths -- because an arm that erodes T-only accuracy shrinks
the conditioning set and so understates its own damage in the conditional rate.

    python scripts/tdistraction_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
RNG = np.random.default_rng(0)
SEEDS = ["", "_s1", "_s2", "_s3"]
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3b", "Qwen2.5-VL-3B"),
          ("llavaov", "LLaVA-OV-7B"), ("next", "LLaVA-NeXT-8B"), ("qwen2b", "Qwen2-VL-2B"),
          ("llava15", "LLaVA-1.5-7B")]

VAL = {json.loads(l)["candidate_id"]                      # selection split of MoGround
       for l in (DR / "dm/multidomain_v1/splits/val.jsonl").read_text().splitlines() if l.strip()}
_S = json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())
ASMDEV = set(_S["dev"])                                   # selection half of the assembled pool
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
POOLS = {"dm_val": ("eval/t8", VAL, False), "asm_dev": ("eval/t8_assembled", ASMDEV, True)}


def load(tag, pool):
    sub, keep, cert = POOLS[pool]
    p = DR / sub / f"{tag}.jsonl"
    if not p.exists():
        return None
    seen = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); seen.setdefault(r["candidate_id"], r)
    return {c: r for c, r in seen.items() if c in keep and not (cert and c in VIOL)}


def t_flips(m):
    """Per-item t-distraction indicator over items the model solves from text alone."""
    return {c: int(r["pred_vt"] != r["correct_index"])
            for c, r in m.items() if r["label"] == "text" and r["pred_t"] == r["correct_index"]}


def t_only_acc(m):
    it = [r for r in m.values() if r["label"] == "text"]
    return float(np.mean([r["pred_t"] == r["correct_index"] for r in it])) if it else float("nan")


def paired(a, b, B=10000):
    """CI on mean(b)-mean(a) over the COMMON conditioning set, plus its point estimate -- the
    marginal rates below are each on their own arm's conditioning set, so when an arm changes
    T-only accuracy the two differ and both are worth seeing."""
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan"), float("nan"), float("nan"), 0
    x = np.array([a[c] for c in common]); y = np.array([b[c] for c in common])
    idx = RNG.integers(0, len(common), (B, len(common)))
    d = y[idx].mean(1) - x[idx].mean(1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(y.mean() - x.mean()), len(common))


def main():
    out = {}
    for pool in POOLS:
        print(f"\n=== {pool} (selection side) : t-distraction, base vs w=0.5 ===")
        print(f"{'backbone':17}{'base':>8}{'w=0.5':>8}{'marg pp':>9}{'paired pp':>11}"
              f"{'95% CI (pp)':>22}{'T-only base':>13}{'T-only w.5':>12}{'n_T':>7}")
        for tag, disp in MODELS:
            b = load(f"{tag}_w0.0", pool)
            if b is None:
                print(f"{disp:17}  (missing base)"); continue
            fb, rows, tacc = t_flips(b), [], []
            fs_all = {}
            for i, s in enumerate(SEEDS):
                m = load(f"{tag}{s}_w0.5", pool)
                if m is None:
                    continue
                f = t_flips(m)
                rows.append(float(np.mean(list(f.values()))) if f else np.nan)
                tacc.append(t_only_acc(m))
                for c, v in f.items():
                    fs_all[(i, c)] = v
            if not rows:
                print(f"{disp:17}  (no steered cells)"); continue
            rb = float(np.mean(list(fb.values()))) if fb else np.nan
            rs = float(np.nanmean(rows))
            # pair each seed against the same base item, so the CI is on the within-item change
            fb_rep = {(i, c): fb[c] for (i, c) in fs_all if c in fb}
            lo, hi, pe, n = paired(fb_rep, {k: v for k, v in fs_all.items() if k in fb_rep})
            flag = "*" if (lo > 0 or hi < 0) else " "
            out[f"{tag}|{pool}"] = {"base": rb, "w0.5": rs, "delta_pp": 100 * (rs - rb),
                                    "ci_pp": [100 * lo, 100 * hi], "paired_delta_pp": 100 * pe,
                                    "sig": bool(lo > 0 or hi < 0),
                                    "t_only_base": t_only_acc(b), "t_only_w05": float(np.nanmean(tacc)),
                                    "n_T_solvable": len(fb)}
            print(f"{disp:17}{rb:>8.4f}{rs:>8.4f}{100*(rs-rb):>+9.2f}{100*pe:>+11.2f}"
                  f"{f'[{100*lo:+.2f}, {100*hi:+.2f}]{flag}':>22}"
                  f"{t_only_acc(b):>13.4f}{float(np.nanmean(tacc)):>12.4f}{len(fb):>7}")
        sel = [v for k, v in out.items() if k.endswith(pool)]
        if sel:
            print(f"{'cohort mean':17}{np.mean([v['base'] for v in sel]):>8.4f}"
                  f"{np.mean([v['w0.5'] for v in sel]):>8.4f}"
                  f"{np.mean([v['delta_pp'] for v in sel]):>+9.2f}"
                  f"{np.mean([v['paired_delta_pp'] for v in sel]):>+11.2f}{'':>22}{np.mean([v['t_only_base'] for v in sel]):>13.4f}"
                  f"{np.mean([v['t_only_w05'] for v in sel]):>12.4f}")
            print(f"  significant changes: "
                  f"{sum(v['sig'] for v in sel)}/{len(sel)} backbones"
                  f"  (of which increases: {sum(v['sig'] and v['paired_delta_pp'] > 0 for v in sel)})")
    p = DR / "diagnostics/tdistraction_audit.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
