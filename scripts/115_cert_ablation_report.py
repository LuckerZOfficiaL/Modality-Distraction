r"""Certification ablation, reporting-side grid for the paper (Qwen2.5-VL-3B).

Compares the paper's arm (objective routed by the per-item grounding certificate) against a
certificate-free arm built without any certificate at all: adversarial captions are written for
every training item rather than only the certified vision-grounded ones, and the objective is
routed uniformly instead of by grounding label. Everything else -- items, LoRA config, KL anchor,
seeds, strength, evaluation -- is identical.

Two recipe families exist; pass --t2r to score the second.
  paper (default, the recipe \S7 reports and the appendix describes)
    certified   runs/ft_kl2.0[_s{S}]              -> keys qwen3b[_s{S}]_w{W} / hm3_qwen3b_s{S}_w{W}
    cert-free   runs/certp_qwen3b_uniform_s{S}    -> keys certp_qwen3b_uniform_s{S}_w{W}[_asmheld]
  t2r (Tier-2 variant, adds a V-only self-distillation term; NOT the reported recipe)
    certified   runs/t2_qwen3b_vteach2[_s{S}]     -> keys qwen3b_t2r[_s{S}]_w{W}[_asmheld]
    cert-free   runs/cert_qwen3b_uniform_s{S}     -> keys cert_qwen3b_uniform_s{S}_w{W}[_asmheld]

Emits data/diagnostics/cert_ablation_report.json with per-pool v-distraction, t-distraction and
net V+T accuracy for both arms (seed means over 4 seeds), paired item-level bootstrap CIs on the
net-accuracy difference, the capability comparison, the strength sweep, and the routing exposure
counts that explain the mechanism.

    python scripts/115_cert_ablation_report.py [--t2r]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
RNG = np.random.default_rng(0)
RECIPE = "paper"          # set from argv in __main__; "t2r" selects the Tier-2 variant grid
SEEDS = [0, 1, 2, 3]
BEN = ["mmstar", "mmbench", "seedbench", "scienceqa"]  # naturalbench excluded: adversarial

TEST = {json.loads(l)["candidate_id"]
        for l in (DR / "dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}
HELD = set(json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())["heldout"])
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])

# pool -> (eval subdir, id filter, drop assembled irrelevance-audit violations)
POOLS = {
    "dm_test": ("eval/t8", TEST, False),
    "human": ("eval/t8", None, False),
    "assembled": ("eval/t8_assembled", HELD, True),
}


def key(arm, pool, seed, w, recipe="paper"):
    """Eval key for one (arm, pool, seed, strength) cell.

    recipe='paper': the arm §7 reports (runs/ft_kl2.0*, KL-anchor 2.0, no V-teacher term); its
    certificate-free counterpart is runs/certp_*.  recipe='t2r': the Tier-2 variant that adds a
    V-only self-distillation term (runs/t2_*_vteach2*) and its counterpart runs/cert_*_uniform_*.
    Both are internally matched; only 'paper' is the recipe the draft describes.
    """
    asm = pool == "assembled"
    if arm == "certified":
        if recipe == "t2r":
            base = f"qwen3b_t2r_w{w}" if seed == 0 else f"qwen3b_t2r_s{seed}_w{w}"
            base += "_asmheld" if asm else ""
        else:
            base = f"qwen3b_w{w}" if seed == 0 else f"qwen3b_s{seed}_w{w}"
        hum = f"hm3_qwen3b_s{seed}_w{w}"
    elif arm == "certfree":
        stem = ("cert" if recipe == "t2r" else "certp") + f"_qwen3b_uniform_s{seed}_w{w}"
        base = stem + ("_asmheld" if asm else "")
        hum = f"hm3_{stem}"
    else:                                     # unsteered reference, shared by both recipes
        base, hum = f"qwen3b_w{w}", f"hm3_qwen3b_w{w}"
    return hum if pool == "human" else base


def load(k, pool):
    sub, keep, cert = POOLS[pool]
    p = DR / sub / f"{k}.jsonl"
    if not p.exists():
        return None
    seen = {}                                 # the pools carry duplicate candidate_ids
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line); seen.setdefault(r["candidate_id"], r)
    return {c: r for c, r in seen.items()
            if (keep is None or c in keep) and not (cert and c in VIOL)}


def flips(m, side):
    """Per-item distraction indicator over items solvable from `side` alone."""
    p = "pred_v" if side == "vision" else "pred_t"
    return {c: int(r["pred_vt"] != r["correct_index"])
            for c, r in m.items() if r["label"] == side and r[p] == r["correct_index"]}


def solo_acc(m, side):
    """Single-modality accuracy over the pool's items of that grounding label: the conditioning
    set of the distraction rate. If an arm erodes it, the conditional rate understates the damage."""
    p = "pred_v" if side == "vision" else "pred_t"
    it = [r for r in m.values() if r["label"] == side]
    return float(np.mean([r[p] == r["correct_index"] for r in it])) if it else float("nan")


def correct(m):
    """Per-item V+T correctness indicator over the whole pool."""
    return {c: int(r["pred_vt"] == r["correct_index"]) for c, r in m.items()}


def rate(d):
    return float(np.mean(list(d.values()))) if d else float("nan")


def paired_bootstrap(a, b, B=10000):
    """95% CI on mean(b) - mean(a) over the (seed, item) pairs both arms are scored on."""
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan"), float("nan"), 0
    x = np.array([a[c] for c in common]); y = np.array([b[c] for c in common])
    idx = RNG.integers(0, len(common), (B, len(common)))
    d = y[idx].mean(1) - x[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(common)


def arm_pool(arm, pool, w="0.5", recipe="paper"):
    """Seed-mean rates plus the pooled (seed, item) indicators for the paired test."""
    if arm == "base":
        w = "0.0"                             # the unsteered reference is a w=0 cell, not w=0.5
    vf, tf, ac, per = {}, {}, {}, {"v": [], "t": [], "acc": [], "vacc": [], "tacc": []}
    for s in SEEDS:
        m = load(key(arm, pool, s, w, recipe), pool)
        if m is None:
            continue
        fv, ft, fa = flips(m, "vision"), flips(m, "text"), correct(m)
        per["v"].append(rate(fv)); per["t"].append(rate(ft)); per["acc"].append(rate(fa))
        per["vacc"].append(solo_acc(m, "vision")); per["tacc"].append(solo_acc(m, "text"))
        for src, dst in ((fv, vf), (ft, tf), (fa, ac)):
            for c, v in src.items():
                dst[(s, c)] = v
        if arm == "base":                     # unsteered: one file, not a seed grid
            break
    if not per["v"]:
        return None
    return {"v": float(np.mean(per["v"])), "t": float(np.mean(per["t"])),
            "acc": float(np.mean(per["acc"])), "v_only_acc": float(np.mean(per["vacc"])),
            "t_only_acc": float(np.mean(per["tacc"])), "n_seeds": len(per["v"]),
            "_vf": vf, "_tf": tf, "_ac": ac}


def cap(k):
    fs = [DR / f"diagnostics/capability/{b}_{k}_rep.json" for b in BEN]
    return float(np.mean([json.loads(f.read_text())["acc"] for f in fs])) if all(f.exists() for f in fs) else np.nan


def capability(arm):
    """Held-out capability rows (400-800), seed mean, and the pooled per-row indicators."""
    accs = []
    for s in SEEDS:
        if arm == "certified":
            stem = "qwen3b_t2r" if RECIPE == "t2r" else "qwen3b"
            tag = f"phase_{stem}_w0.5" if s == 0 else f"phase_{stem}_s{s}_w0.5"
        else:
            pre = "cert" if RECIPE == "t2r" else "certp"
            tag = f"phase_{pre}_qwen3b_uniform_s{s}_w0.5"
        c = cap(tag)
        if not np.isnan(c):
            accs.append(c)
    return float(np.mean(accs)) if accs else float("nan"), len(accs)


def routing_exposure():
    """How many train items each arm carries an adversarial caption for, by grounding label."""
    out = {}
    for arm, fn in (("certified", "train_adv.jsonl"), ("certfree", "train_adv_all.jsonl")):
        p = DR / "dm/multidomain_v1/splits" / fn
        if not p.exists():
            out[arm] = None; continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        have = [r for r in rows if r.get("adv_caption")]
        out[arm] = {"vision": sum(1 for r in have if r["label"] == "vision"),
                    "text": sum(1 for r in have if r["label"] == "text"),
                    "total_items": len(rows)}
    return out


def main(recipe="paper"):
    rep = {"model": "Qwen2.5-VL-3B", "recipe": recipe, "w": 0.5, "seeds": SEEDS, "pools": {}}
    for pool in POOLS:
        cells = {a: arm_pool(a, pool, recipe=recipe) for a in ("base", "certified", "certfree")}
        if cells["certified"] is None or cells["certfree"] is None:
            print(f"[skip] {pool}: incomplete"); continue
        e = {}
        for a, c in cells.items():
            if c:
                e[a] = {"v_distraction": c["v"], "t_distraction": c["t"], "net_acc": c["acc"],
                        "v_only_acc": c["v_only_acc"], "t_only_acc": c["t_only_acc"],
                        "n_seeds": c["n_seeds"]}
        for stat, fld in (("d_net_acc", "_ac"), ("d_v_distraction", "_vf"), ("d_t_distraction", "_tf")):
            lo, hi, n = paired_bootstrap(cells["certified"][fld], cells["certfree"][fld])
            e[stat] = {"lo": lo, "hi": hi, "n_pairs": n, "excludes_zero": bool(lo > 0 or hi < 0)}
        rep["pools"][pool] = e

    cc, ncc = capability("certified"); cf, ncf = capability("certfree")
    rep["capability"] = {"certified": cc, "certfree": cf, "delta_pp": 100 * (cf - cc),
                         "n_seeds": [ncc, ncf]}

    rep["wsweep"] = {}
    for w in ("0.25", "0.5", "0.75"):
        row = {}
        for a in ("certified", "certfree"):
            c = arm_pool(a, "dm_test", w, recipe)
            if c:
                row[a] = {"v_distraction": c["v"], "t_distraction": c["t"], "net_acc": c["acc"],
                          "n_seeds": c["n_seeds"]}
        rep["wsweep"][w] = row

    rep["routing_exposure"] = routing_exposure()

    out = DR / f"diagnostics/cert_ablation_report{'_t2r' if recipe == 't2r' else ''}.json"
    out.write_text(json.dumps(rep, indent=2))

    # ---- console view -------------------------------------------------------
    name = {"dm_test": "MoGround test", "human": "MoGround-Human", "assembled": "assembled held-out"}
    print(f"\n{'pool':22}{'arm':11}{'v-distr':>9}{'t-distr':>9}{'net acc':>9}   d(net) 95% CI")
    for pool, e in rep["pools"].items():
        for a in ("base", "certified", "certfree"):
            if a not in e:
                continue
            ci = ""
            if a == "certfree":
                d = e["d_net_acc"]
                ci = f"   [{100*d['lo']:+.2f}, {100*d['hi']:+.2f}]pp {'*' if d['excludes_zero'] else '(ns)'}"
            print(f"{name[pool]:22}{a:11}{e[a]['v_distraction']:>9.4f}"
                  f"{e[a]['t_distraction']:>9.4f}{e[a]['net_acc']:>9.4f}{ci}")
        print()
    c = rep["capability"]
    print(f"capability (held-out rows): certified {c['certified']:.4f}  "
          f"cert-free {c['certfree']:.4f}  delta {c['delta_pp']:+.2f}pp")
    print("\nstrength sweep on MoGround test (v-distr / t-distr / net acc):")
    for w, row in rep["wsweep"].items():
        for a, v in row.items():
            print(f"  w={w:5}{a:11}{v['v_distraction']:>9.4f}{v['t_distraction']:>9.4f}{v['net_acc']:>9.4f}")
    print(f"\nrouting exposure (train items given an adversarial caption): {rep['routing_exposure']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import sys
    RECIPE = "t2r" if "--t2r" in sys.argv else "paper"
    main(RECIPE)
