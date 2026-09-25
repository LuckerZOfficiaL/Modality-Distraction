r"""Does a gate trained on MoGround transfer to the assembled pool?

scripts/gated_patch_probe.py cross-fits its gates *within* each pool, which cannot tell a transferable signal from a
pool-specific artifact. Here the gates are trained on the \dsname{} TRAIN split only and evaluated
on the assembled held-out half, which shares no items, no source datasets and no caption
distribution with the training pool.

  gate D  predicted DISTRACTION      (fires where the caption is expected to flip the answer)
  gate V  predicted VISION-GROUNDING (fires where the answer is in the image, making the patch safe)

As in scripts/gated_patch_probe.py only the gate is hypothetical: the patch's effect is taken from the measured runs.
Reported against the oracle gate, the always-patch policy and a matched-rate random gate.

Offline, CPU only.

    python scripts/gate_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
MODELS = [("qwen", "Qwen2.5-VL-3B", 28), ("llavanext", "LLaVA-NeXT-8B", 18)]
TRAIN = {json.loads(l)["candidate_id"] for l in
         (DR / "dm/multidomain_v1/splits/train.jsonl").read_text().splitlines() if l.strip()}
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
HELD = set(json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())["heldout"])


def load(model, sub, L, keepset):
    st = DR / "activations" / model / sub
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ll = np.load(st / "letter_logits.npy").astype(np.float64)
    acts = np.load(st / "activations.npy", mmap_mode="r")
    keep = [i for i, r in enumerate(idx) if r["candidate_id"] in keepset]
    gold = np.array([idx[i]["correct_index"] for i in keep])
    lab = np.array([idx[i]["label"] for i in keep])
    return (np.asarray(acts[keep, 0, L], dtype=np.float32),
            lab == "vision",
            ll[keep, 1].argmax(1) == gold,          # V-only correct
            ll[keep, 0].argmax(1) == gold)          # V+T correct


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=0.05).fit((X - mu) / sd, y)
    return lambda Z: clf.predict_proba((Z - mu) / sd)[:, 1]


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    TEST = {json.loads(l)["candidate_id"] for l in
            (DR / "dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}
    EVAL = [("test", "\\dsname test", "dm_multidomain_v1_cf", TEST, "test", "test"),
            ("asmheld", "assembled held-out", "merged_aokvqa_racehigh_cf", HELD - VIOL, "asmheld", "asm")]
    rows = []
    for key, disp, L in MODELS:
      Xtr, vtr, vok_tr, vtok_tr = load(key, "dm_multidomain_v1_cf", L, TRAIN)
      for pool, pdisp, sub, keepset, dtag, utag in EVAL:
          Xte, vte, vok_te, vtok_te = load(key, sub, L, keepset)
          dtr = vtr & vok_tr & ~vtok_tr                       # distracted, train pool
          v_solv = vte & vok_te
          distracted = v_solv & ~vtok_te
          robust_v = v_solv & vtok_te
          is_t = ~vte

          pa = json.loads((DR / f"diagnostics/dense_control_{key}_L{L}_{dtag}_patch.json").read_text())
          up = json.loads((DR / f"diagnostics/universal_patch_{key}_L{L}_{utag}.json").read_text())
          r_d, p_v = pa["recovery_real"], pa["vpass_preserved"]
          p_t, t_base = up["patched"]["text acc"], up["unpatched"]["text acc"]
          base = distracted.sum() / v_solv.sum()

          def ev(fire):
              vd = ((~fire & distracted).sum() + (fire & distracted).sum() * (1 - r_d)
                    + (fire & robust_v).sum() * (1 - p_v)) / v_solv.sum()
              t = ((fire & is_t).sum() * p_t + (~fire & is_t).sum() * t_base) / max(is_t.sum(), 1)
              return float(vd), float(100 * (t - t_base)), float(fire.mean())

          sV, sD = fit(Xtr, vtr.astype(int))(Xte), fit(Xtr, dtr.astype(int))(Xte)
          rng = np.random.default_rng(0)
          cells = {"oracle gate": ev(distracted) + (float("nan"),),
                   "always patch": ev(np.ones(len(vte), bool)) + (float("nan"),)}
          for nm, sc, y in (("gate D (trained on \\dsname)", sD, distracted),
                            ("gate V (trained on \\dsname)", sV, vte)):
              k = max(int(y.sum()), 1)
              fire = sc >= np.sort(sc)[::-1][k - 1]
              cells[nm] = ev(fire) + (auroc(y.astype(int), sc),)
              rnd = np.zeros(len(vte), bool)
              rnd[rng.choice(len(vte), int(fire.sum()), replace=False)] = True
              cells[f"  matched-rate random ({nm.split()[1]})"] = ev(rnd) + (float("nan"),)

          print(f"\n===== {disp} (L{L}): train on \\dsname{{}} train, evaluate on {pdisp}")
          print(f"  base v-distr {base:.3f} ({distracted.sum()}/{v_solv.sum()}), text acc {t_base:.3f}; "
                f"measured patch recovery {r_d:.2f}, V-pass kept {p_v:.2f}, patched text {p_t:.2f}")
          print(f"  {'policy':38}{'v-distr':>9}{'rel':>8}{'d text':>9}{'fires':>8}{'AUROC':>8}")
          for nm, (vd, dt, fr, au) in cells.items():
              a = "" if au != au else f"{au:.3f}"
              print(f"  {nm:38}{vd:>9.3f}{100*(vd/base-1):>+7.0f}%{dt:>+9.1f}{fr:>8.2f}{a:>8}")
              # net end-to-end change in overall accuracy, all counts from this same pool
              net = 100 * ((base - vd) * int(v_solv.sum()) + dt / 100 * int(is_t.sum())) \
                  / (int(v_solv.sum()) + int(is_t.sum()))
              rows.append(dict(model=key, pool=pool, policy=nm, v_distraction=vd,
                               rel_change=100 * (vd / base - 1), d_text_pp=dt, fire_rate=fr,
                               auroc=(None if au != au else au), base=float(base),
                               n_v_solvable=int(v_solv.sum()), n_text=int(is_t.sum()),
                               net_overall_pp=float(net)))
    (DR / "diagnostics/gate_transfer.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {DR / 'diagnostics/gate_transfer.json'}")


if __name__ == "__main__":
    main()
