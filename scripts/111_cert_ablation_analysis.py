r"""Certification ablation: does routing the training objective by the grounding certificate matter?

Compares the reported arm (certificate-routed) against the shuffled-label arm on the reporting
side only, at the paper's fixed default strength w=0.5. Same items, same captions, same objective
mix, same seeds, same evaluation -- only the per-item routing label differs.

  A1 certified   runs/t2_qwen3b_vteach2[_s{S}]   -> eval keys qwen3b_t2r[_s{S}]_w0.5[_asmheld]
  A4 shuffled    runs/cert_qwen3b_shuffled_s{S}  -> eval keys cert_qwen3b_shuffled_s{S}_w0.5[...]

Reports per-seed and seed-mean v-distraction on \dsname test and the audited assembled held-out
half, the absolute and relative gap, and a paired item-level bootstrap CI on the difference.

    python scripts/111_cert_ablation_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
RNG = np.random.default_rng(0)
TEST = {json.loads(l)["candidate_id"]
        for l in (DR / "dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}
_SPL = json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())
HELD = set(_SPL["heldout"])
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])


def load(key, pool):
    p = DR / ("eval/t8_assembled" if pool == "asm" else "eval/t8") / f"{key}.jsonl"
    if not p.exists():
        return None
    d = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); d.setdefault(r["candidate_id"], r)
    ids = HELD if pool == "asm" else TEST
    return {c: r for c, r in d.items()
            if r["label"] == "vision" and c in ids and not (pool == "asm" and c in VIOL)}


def flips(d):
    """per-item flip indicator over V-solvable items, keyed by candidate_id."""
    return {c: int(r["pred_vt"] != r["correct_index"])
            for c, r in d.items() if r["pred_v"] == r["correct_index"]}


def rate(f):
    return sum(f.values()) / len(f) if f else float("nan")


def paired_bootstrap(fa, fb, B=10000):
    """CI on rate(shuffled) - rate(certified) over the items both arms can be scored on."""
    common = sorted(set(fa) & set(fb))
    a = np.array([fa[c] for c in common]); b = np.array([fb[c] for c in common])
    n = len(common)
    idx = RNG.integers(0, n, (B, n))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), n


def main():
    SEEDS = [0, 1, 2, 3]
    for pool, label in (("dm", "\\dsname test"), ("asm", "assembled held-out")):
        print(f"\n=== {label} (v-distraction, w=0.5) ===")
        print(f"{'seed':>5}{'certified':>12}{'shuffled':>11}{'abs gap':>10}{'rel':>9}")
        cert_f, shuf_f = {}, {}
        rows = []
        for s in SEEDS:
            ck = f"qwen3b_t2r_w0.5" if s == 0 else f"qwen3b_t2r_s{s}_w0.5"
            sk = f"cert_qwen3b_shuffled_s{s}_w0.5"
            if pool == "asm":
                ck += "_asmheld"; sk += "_asmheld"
            c, sh = load(ck, pool), load(sk, pool)
            if c is None or sh is None:
                print(f"{s:>5}   {'(missing)' if sh is None else '(missing certified)'}")
                continue
            fc, fs = flips(c), flips(sh)
            cert_f[s], shuf_f[s] = fc, fs
            rc, rs = rate(fc), rate(fs)
            rows.append((rc, rs))
            print(f"{s:>5}{rc:>12.4f}{rs:>11.4f}{rs-rc:>+10.4f}{(rs/rc-1)*100:>+8.0f}%")
        if not rows:
            print("  no completed cells yet"); continue
        mc = float(np.mean([r[0] for r in rows])); ms = float(np.mean([r[1] for r in rows]))
        print(f"{'mean':>5}{mc:>12.4f}{ms:>11.4f}{ms-mc:>+10.4f}{(ms/mc-1)*100:>+8.0f}%")
        # pool every seed's items for the paired test (each seed contributes its own flips)
        fa = {(s, c): v for s in cert_f for c, v in cert_f[s].items()}
        fb = {(s, c): v for s in shuf_f for c, v in shuf_f[s].items()}
        lo, hi, n = paired_bootstrap(fa, fb)
        excl = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  paired item bootstrap on (shuffled - certified): "
              f"[{lo:+.4f}, {hi:+.4f}] over {n} model-item pairs -> {excl}")


if __name__ == "__main__":
    main()
