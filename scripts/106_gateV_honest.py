r"""Gate V, evaluated under the paper's own selection/reporting discipline.

scripts/105 fired the gate at an arbitrary rate; a threshold sweep showed the verdict depends on it.
Here the threshold is CHOSEN ON THE SELECTION SIDE and then applied unchanged to the reporting side,
exactly as the task-vector strength is.

  probe      logistic on h_VT at the intervention layer, fitted on \dsname TRAIN only
  threshold  argmax of net overall accuracy on the SELECTION side
             (\dsname val + assembled dev half, pooled), never on reporting data
  report     \dsname test and the assembled held-out half, at that fixed threshold

As before only the gate is simulated: the patch's effect comes from the measured runs. The task
vector is shown alongside on the identical pools for reference.

Offline, CPU only.

    python scripts/106_gateV_honest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
MODELS = [("qwen", "qwen3b", "Qwen2.5-VL-3B", 28), ("llavanext", "next", "LLaVA-NeXT-8B", 18)]
SPL = lambda s: {json.loads(l)["candidate_id"] for l in
                 (DR / f"dm/multidomain_v1/splits/{s}.jsonl").read_text().splitlines() if l.strip()}
TRAIN, VAL, TEST = SPL("train"), SPL("val"), SPL("test")
ASM = json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())
VIOL = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
DEV, HELD = set(ASM["dev"]) - VIOL, set(ASM["heldout"]) - VIOL
GRID = np.concatenate([np.linspace(0.05, 0.95, 19), 1 - np.logspace(-1, -5, 25)])


def load(model, sub, L, keepset):
    st = DR / "activations" / model / sub
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ll = np.load(st / "letter_logits.npy").astype(np.float64)
    acts = np.load(st / "activations.npy", mmap_mode="r")
    k = [i for i, r in enumerate(idx) if r["candidate_id"] in keepset]
    g = np.array([idx[i]["correct_index"] for i in k])
    lab = np.array([idx[i]["label"] for i in k])
    return (np.asarray(acts[k, 0, L], dtype=np.float32), lab == "vision",
            ll[k, 1].argmax(1) == g, ll[k, 0].argmax(1) == g)


def effects(model, L, dtag, utag):
    pa = json.loads((DR / f"diagnostics/dense_control_{model}_L{L}_{dtag}_patch.json").read_text())
    up = json.loads((DR / f"diagnostics/universal_patch_{model}_L{L}_{utag}.json").read_text())
    return pa["recovery_real"], pa["vpass_preserved"], up["patched"]["text acc"], up["unpatched"]["text acc"]


def policy(fire, is_v, v_ok, vt_ok, eff):
    r_d, p_v, p_t, t_base = eff
    vs = is_v & v_ok; dis = vs & ~vt_ok; rob = vs & vt_ok; ist = ~is_v
    base = dis.sum() / max(vs.sum(), 1)
    vd = ((~fire & dis).sum() + (fire & dis).sum() * (1 - r_d)
          + (fire & rob).sum() * (1 - p_v)) / max(vs.sum(), 1)
    fixed = (base - vd) * vs.sum()
    broken = (fire & ist).sum() * (t_base - p_t)
    net = 100 * (fixed - broken) / (vs.sum() + ist.sum())
    d_text = -100 * broken / max(ist.sum(), 1)
    return dict(v_distraction=float(vd), base=float(base), rel=float(100 * (vd / base - 1)),
                d_text_pp=float(d_text), net_overall_pp=float(net), fire_rate=float(fire.mean()))


def tv_net(tag, sub, keep, cert):
    def ov(fp):
        seen = {}
        for l in Path(fp).read_text().splitlines():
            if l.strip():
                r = json.loads(l); seen.setdefault(r["candidate_id"], r)
        rows = [r for r in seen.values() if r["candidate_id"] in keep and (not cert or r["candidate_id"] not in VIOL)]
        ci = np.array([r["correct_index"] for r in rows]); vt = np.array([r["pred_vt"] for r in rows])
        lab = np.array([r["label"] for r in rows]); v = np.array([r["pred_v"] for r in rows])
        vs = (lab == "vision") & (v == ci)
        return float((vt == ci).mean()), float((vs & (vt != ci)).sum() / vs.sum())
    b, bd = ov(DR / f"eval/{sub}/{tag}_w0.0.jsonl")
    a = [ov(DR / f"eval/{sub}/{tag}{s}_w0.5.jsonl") for s in ("", "_s1", "_s2", "_s3")]
    return 100 * (np.mean([x[0] for x in a]) - b), 100 * (np.mean([x[1] for x in a]) / bd - 1)


def main():
    out = []
    for key, tag, disp, L in MODELS:
        Xtr, vtr, _, _ = load(key, "dm_multidomain_v1_cf", L, TRAIN)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.05).fit((Xtr - mu) / sd, vtr.astype(int))
        P = lambda X: clf.predict_proba((X - mu) / sd)[:, 1]

        sel = {"val": load(key, "dm_multidomain_v1_cf", L, VAL) + (effects(key, L, "val", "test"),),
               "asmdev": load(key, "merged_aokvqa_racehigh_cf", L, DEV) + (effects(key, L, "asmdev", "asm"),)}
        rep = {"test": load(key, "dm_multidomain_v1_cf", L, TEST) + (effects(key, L, "test", "test"),),
               "asmheld": load(key, "merged_aokvqa_racehigh_cf", L, HELD) + (effects(key, L, "asmheld", "asm"),)}

        # ---- choose the threshold on the SELECTION side (pooled net overall accuracy) ----
        best_t, best_net = None, -1e9
        for t in GRID:
            tot_gain, tot_n = 0.0, 0
            for X, is_v, v_ok, vt_ok, eff in sel.values():
                r = policy(P(X) >= t, is_v, v_ok, vt_ok, eff)
                n = int((is_v & v_ok).sum() + (~is_v).sum())
                tot_gain += r["net_overall_pp"] * n; tot_n += n
            if tot_gain / tot_n > best_net:
                best_net, best_t = tot_gain / tot_n, float(t)
        print(f"\n===== {disp} (L{L})   threshold chosen on selection side: {best_t:.5f} "
              f"(selection-side net {best_net:+.2f}pp)")
        print(f"  {'reporting pool':22}{'v-distr':>9}{'rel':>7}{'d text':>9}{'net':>9}{'fires':>8}"
              f"   | task vector: rel / net")
        for pool, (X, is_v, v_ok, vt_ok, eff) in rep.items():
            r = policy(P(X) >= best_t, is_v, v_ok, vt_ok, eff)
            sub, keep, cert = (("t8", TEST, False) if pool == "test" else ("t8_assembled", HELD, True))
            tvn, tvr = tv_net(tag, sub, keep, cert)
            print(f"  {pool:22}{r['v_distraction']:>9.3f}{r['rel']:>+6.0f}%{r['d_text_pp']:>+9.1f}"
                  f"{r['net_overall_pp']:>+9.2f}{r['fire_rate']:>8.2f}   | {tvr:+.0f}% / {tvn:+.2f}pp")
            out.append(dict(model=key, pool=pool, threshold=best_t, **r,
                            tv_rel=float(tvr), tv_net_pp=float(tvn)))
    (DR / "diagnostics/gateV_honest.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {DR / 'diagnostics/gateV_honest.json'}")


if __name__ == "__main__":
    main()
