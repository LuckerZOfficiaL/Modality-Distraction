r"""Can a learned gate make the h_V patch deployable?

The h_V patch removes ~all v-distraction but is marked "oracle" in the baseline table because it
must be applied only to the right items: applied to text-grounded items it destroys the text side.
This script asks whether a *learned* gate can supply that decision at inference time, testing the
two gates one would actually try:

  gate D  predicted DISTRACTION  (the item is V-solvable and the caption will flip it)
  gate V  predicted VISION-GROUNDING (the item's answer is in the image; patch is then harmless)

Both are cross-fitted logistic probes on the stored h_VT residual at the intervention layer, so
every item is scored by a model that never saw it. Only the GATE is simulated; the patch's effect
is taken from the measured runs (scripts/87, 88):

  fires on a distracted item -> recovered with prob recovery_real
  fires on a robust V item   -> stays correct with prob vpass_preserved
  fires on a text item       -> correct with prob (patched text acc), from the universal-patch run

so the numbers below are what the measured intervention would deliver under each realisable gate.
Compared against: oracle gate (fires exactly on distracted items), always-patch, and a
matched-rate random gate.

Offline, CPU only.

    python scripts/103_gated_patch_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
MODELS = [("qwen", "Qwen2.5-VL-3B", 28), ("llavanext", "LLaVA-NeXT-8B", 18)]
SUB = "dm_multidomain_v1_cf"
TEST = {json.loads(l)["candidate_id"] for l in
        (DR / "dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}
FOLDS, SEED = 5, 0


def crossfit(X, y, seed=SEED):
    """Out-of-fold P(y=1) so no item is scored by a probe that saw it."""
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.05).fit((X[tr] - mu) / sd, y[tr])
        p[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return p


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


POOLS = [("test", "\\dsname test", "dm_multidomain_v1_cf", "test", "test"),
         ("asmheld", "assembled held-out", "merged_aokvqa_racehigh_cf", "asmheld", "asm")]


def main():
    VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
    HELD = set(json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())["heldout"])
    rows = []
    for key, disp, L in MODELS:
        for pool, pdisp, sub, dtag, utag in POOLS:
            st = DR / "activations" / key / sub
            if not (st / "letter_logits.npy").exists():
                print(f"  [skip] {key}/{pool}: no store"); continue
            idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
            ll = np.load(st / "letter_logits.npy").astype(np.float64)
            acts = np.load(st / "activations.npy", mmap_mode="r")
            ids = {r["candidate_id"] for r in idx}
            keepset = TEST if pool == "test" else (HELD & ids) - VIOL
            keep = [i for i, r in enumerate(idx) if r["candidate_id"] in keepset]

            gold = np.array([idx[i]["correct_index"] for i in keep])
            lab = np.array([idx[i]["label"] for i in keep])
            vt_ok = ll[keep, 0].argmax(1) == gold
            v_ok = ll[keep, 1].argmax(1) == gold
            X = np.asarray(acts[keep, 0, L], dtype=np.float32)

            is_v = lab == "vision"; is_t = ~is_v
            v_solv = is_v & v_ok
            distracted = v_solv & ~vt_ok
            robust_v = v_solv & vt_ok
            if distracted.sum() < 5:
                print(f"  [skip] {key}/{pool}: only {distracted.sum()} distracted"); continue

            pa = json.loads((DR / f"diagnostics/dense_control_{key}_L{L}_{dtag}_patch.json").read_text())
            up = json.loads((DR / f"diagnostics/universal_patch_{key}_L{L}_{utag}.json").read_text())
            r_d, p_v = pa["recovery_real"], pa["vpass_preserved"]
            p_t, t_base = up["patched"]["text acc"], up["unpatched"]["text acc"]
            base_vd = distracted.sum() / v_solv.sum()

            def evaluate(fire):
                vd = ((~fire & distracted).sum() + (fire & distracted).sum() * (1 - r_d)
                      + (fire & robust_v).sum() * (1 - p_v)) / v_solv.sum()
                t_acc = (((fire & is_t).sum() * p_t + (~fire & is_t).sum() * t_base)
                         / max(is_t.sum(), 1))
                return float(vd), float(100 * (t_acc - t_base)), float(fire.mean())

            pD = crossfit(X, distracted.astype(int))
            pV = crossfit(X, is_v.astype(int))
            rng = np.random.default_rng(SEED)
            cells = {"oracle gate": evaluate(distracted) + (float("nan"),),
                     "always patch": evaluate(np.ones(len(keep), bool)) + (float("nan"),)}
            for name, sc, y in (("gate D (predicted distraction)", pD, distracted),
                                ("gate V (predicted vision-grounding)", pV, is_v)):
                k = int(y.sum())
                fire = sc >= np.sort(sc)[::-1][k - 1]
                cells[name] = evaluate(fire) + (auroc(y.astype(int), sc),)
                rnd = np.zeros(len(keep), bool)
                rnd[rng.choice(len(keep), int(fire.sum()), replace=False)] = True
                cells[f"  matched-rate random ({name.split()[1]})"] = evaluate(rnd) + (float("nan"),)

            print(f"\n===== {disp} / {pdisp} (L{L})  base v-distr {base_vd:.3f} "
                  f"({distracted.sum()}/{v_solv.sum()}), text acc {t_base:.3f}")
            print(f"  measured patch: recovery {r_d:.2f}, V-pass kept {p_v:.2f}, patched text {p_t:.2f}")
            print(f"  {'policy':38}{'v-distr':>9}{'rel':>8}{'d text':>9}{'fires':>8}{'AUROC':>8}")
            for name, (vd, dt, fr, au) in cells.items():
                a = "" if au != au else f"{au:.3f}"
                print(f"  {name:38}{vd:>9.3f}{100*(vd/base_vd-1):>+7.0f}%{dt:>+9.1f}{fr:>8.2f}{a:>8}")
                rows.append(dict(model=key, pool=pool, policy=name, v_distraction=vd,
                                 rel_change=100 * (vd / base_vd - 1), d_text_pp=dt,
                                 fire_rate=fr, auroc=(None if au != au else au), base=float(base_vd)))
    (DR / "diagnostics/gated_patch_probe.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {DR / 'diagnostics/gated_patch_probe.json'}")


if __name__ == "__main__":
    main()
