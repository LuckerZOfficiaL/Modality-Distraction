r"""M3ID-style text-contrastive decoding baseline (Favero et al., CVPR 2024) on cached logits.

Decoding rule, faithful to the paper's Eq. 3-4 / Algorithm 1 at t=1 (single-token MCQ):

    l* = l_c + 1[ max_k softmax(l_c)_k < alpha ] * mu * (l_c - l_u)

with l_c = image-conditioned (VT) letter logits, l_u = unconditioned (T-only) letter logits,
mu = (1-gamma_1)/gamma_1 the collapsed forgetting coefficient, and the paper's confidence gate:
no correction when the conditioned model is already confident. Adaptations, both documented in
the paper draft if this row is reported: the gate is computed on the softmax over the four answer
letters (full-vocab logits are not cached; in forced-choice MCQ the next token is a letter), and
per-condition log-partition constants cancel in the letter argmax, so raw cached letter logits
are exact for prediction.

Scoring path: all rates (including the mu=0 base) come from the cf-store letter-argmax, so the
intervention is measured against its own base within one measurement channel. For Qwen this
channel agrees with the canonical behavioral runs at 99.3-100%; for LLaVA-NeXT the canonical
path is generate-and-parse and agrees 80-92%, which is why deltas, not absolute rates, are the
comparable quantity.

(mu, alpha) are swept on the SELECTION side only (MoGround val + assembled dev half); the chosen
pair is then evaluated once on the reporting side (MoGround test + assembled held-out, audited).
Selection criterion mirrors the task vector's: mean v-distraction reduction across the two
selection surfaces, subject to net accuracy not dropping by more than 0.5pp on either surface
(the gate exists so the method is not allowed to buy v-distraction with text collapse).

    python scripts/128_m3id_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
MUS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
ALPHAS = [1.0, 0.9, 0.8, 0.6, 0.4]          # 1.0 = gate never blocks (correction always applied)
MODELS = [("Qwen_Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B"),
          ("OpenGVLab_InternVL3-8B-hf", "InternVL3-8B"),
          ("qwen", "Qwen2.5-VL-3B"),
          ("llava-hf_llava-onevision-qwen2-7b-ov-hf", "LLaVA-OV-7B"),
          ("llavanext", "LLaVA-NeXT-8B"),
          ("Qwen_Qwen2-VL-2B-Instruct", "Qwen2-VL-2B"),
          ("llava-hf_llava-1.5-7b-hf", "LLaVA-1.5-7B")]

VAL = {json.loads(l)["candidate_id"] for l in (DR / "dm/multidomain_v1/splits/val.jsonl").read_text().splitlines() if l.strip()}
TEST = {json.loads(l)["candidate_id"] for l in (DR / "dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}
_S = json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())
DEV, HELD = set(_S["dev"]), set(_S["heldout"])
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])


def load_store(model, store):
    d = DR / f"activations/{model}/{store}"
    ll = np.load(d / "letter_logits.npy").astype(np.float64)
    ix = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines() if l.strip()]
    return ll, ix


def rates(ll, ix, keep, mu, alpha):
    """Own-set v/t-distraction + net accuracy under the M3ID rule with parameters (mu, alpha)."""
    sel = [i for i, r in enumerate(ix) if r["candidate_id"] in keep]
    lc, lv, lu = ll[sel, 0], ll[sel, 1], ll[sel, 2]
    gold = np.array([ix[i]["correct_index"] for i in sel])
    lab = np.array([ix[i]["label"] for i in sel])
    p_c = np.exp(lc - lc.max(1, keepdims=True))
    p_c /= p_c.sum(1, keepdims=True)
    gate = (p_c.max(1) < alpha)                        # correction applied only where not confident
    lstar = lc + gate[:, None] * mu * (lc - lu)
    pred = lstar.argmax(1)
    pv, pt = lv.argmax(1), lu.argmax(1)
    V, T = lab == "vision", lab == "text"
    sv, st = V & (pv == gold), T & (pt == gold)
    return {"vd": float(((pred != gold) & sv).sum() / max(sv.sum(), 1)),
            "td": float(((pred != gold) & st).sum() / max(st.sum(), 1)),
            "acc": float((pred == gold).mean()),
            # text accuracy on ALL text-grounded items: the quantity tab:baseline's "Delta Text"
            # column reports for every other intervention (accuracy, not the conditional rate)
            "text_acc": float((pred[T] == gold[T]).mean()) if T.any() else float("nan"),
            "gate_applied_frac": float(gate.mean())}


def main():
    out = {}
    for model, disp in MODELS:
        dm_ll, dm_ix = load_store(model, "dm_multidomain_v1_cf")
        as_ll, as_ix = load_store(model, "merged_aokvqa_racehigh_cf")
        askeep_dev = {c for c in DEV if c not in VIOL}
        askeep_hld = {c for c in HELD if c not in VIOL}

        base_sel = {"dm": rates(dm_ll, dm_ix, VAL, 0.0, 1.0),
                    "asm": rates(as_ll, as_ix, askeep_dev, 0.0, 1.0)}
        # ---- selection sweep --------------------------------------------------------------
        best, best_score = None, -1e9
        grid = []
        for mu in MUS:
            for alpha in ALPHAS:
                if mu == 0.0 and alpha != 1.0:
                    continue                            # mu=0 is the base regardless of alpha
                rd = rates(dm_ll, dm_ix, VAL, mu, alpha)
                ra = rates(as_ll, as_ix, askeep_dev, mu, alpha)
                red = np.mean([1 - rd["vd"] / max(base_sel["dm"]["vd"], 1e-9),
                               1 - ra["vd"] / max(base_sel["asm"]["vd"], 1e-9)])
                acc_ok = (rd["acc"] >= base_sel["dm"]["acc"] - 0.005
                          and ra["acc"] >= base_sel["asm"]["acc"] - 0.005)
                grid.append({"mu": mu, "alpha": alpha, "red_sel": float(red),
                             "dm": rd, "asm": ra, "acc_ok": bool(acc_ok)})
                if acc_ok and mu > 0 and red > best_score:
                    best, best_score = (mu, alpha), red
        # ---- report side at the chosen point ---------------------------------------------
        rep = {}
        if best:
            mu, alpha = best
            rep = {"dm_test": {"base": rates(dm_ll, dm_ix, TEST, 0.0, 1.0),
                               "m3id": rates(dm_ll, dm_ix, TEST, mu, alpha)},
                   "asm_held": {"base": rates(as_ll, as_ix, askeep_hld, 0.0, 1.0),
                                "m3id": rates(as_ll, as_ix, askeep_hld, mu, alpha)}}
        out[model] = {"display": disp, "selected": best, "selection_score": best_score,
                      "grid": grid, "report": rep}

        print(f"\n=== {disp} ===")
        print(f"{'mu':>5}{'alpha':>7}{'dm vd':>8}{'dm acc':>8}{'asm vd':>8}{'asm acc':>9}"
              f"{'sel red':>9}{'acc_ok':>8}")
        for g in grid:
            tag = " <-- selected" if best and (g["mu"], g["alpha"]) == best else ""
            print(f"{g['mu']:>5}{g['alpha']:>7}{g['dm']['vd']:>8.3f}{g['dm']['acc']:>8.3f}"
                  f"{g['asm']['vd']:>8.3f}{g['asm']['acc']:>9.3f}{g['red_sel']:>9.2f}"
                  f"{str(g['acc_ok']):>8}{tag}")
        if best:
            for surf, r in rep.items():
                b, m = r["base"], r["m3id"]
                print(f"  REPORT {surf}: vd {b['vd']:.3f} -> {m['vd']:.3f}"
                      f" ({100*(1-m['vd']/max(b['vd'],1e-9)):+.0f}%)   td {b['td']:.3f} -> {m['td']:.3f}"
                      f"   acc {b['acc']:.3f} -> {m['acc']:.3f}")
        else:
            print("  no (mu, alpha) passed the selection-side accuracy guard")
    p = DR / "diagnostics/m3id_baseline.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
