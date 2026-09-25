"""Horse race: does anything predict cell distraction as well as grounding strength?

Unit = cell (backbone x domain x modality), the same 70 cells as the grounding-strength relation.
Outcome = the cell's conditional distraction rate. Horses, all computed from the released stores
(letter logits under the three conditions) or from public model cards:

  grounding    single-modality accuracy in the grounded condition (the paper's predictor)
  size         log10 parameter count of the backbone
  entropy      mean softmax entropy over the 4 answer letters, grounded condition
  confidence   mean max-softmax probability, grounded condition
  ece          expected calibration error (10 bins), grounded condition
  off_acc      accuracy of the OFF modality alone (strength of the distractor channel)
  vt_acc       accuracy with both modalities (overall task difficulty)
  qlen/clen    mean question / context length in words (surface covariates)

Three comparisons:
  (1) univariate Pearson/Spearman r across the 70 cells, with model-family-clustered
      bootstrap 95% CIs;
  (2) leave-one-model-out prediction: fit distraction ~ predictor on six backbones' cells,
      predict the held-out backbone's cells; report MAE and held-out R^2 per predictor;
  (3) all-predictor standardized OLS + drop-one delta-R^2 (dominance-lite).

Offline, CPU only.

    python scripts/predictor_horserace.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
DOMS = ["dci", "vistext", "semart", "roco"]
DM_SUB, ASM_SUB = "dm_multidomain_v1_cf", "merged_aokvqa_racehigh_cf"
MIN_N = 30
SIZES = {  # billions of parameters (HF model cards)
    "qwen": 3.75, "Qwen_Qwen2.5-VL-7B-Instruct": 8.29, "Qwen_Qwen2-VL-2B-Instruct": 2.21,
    "OpenGVLab_InternVL3-8B-hf": 8.08, "llava-hf_llava-1.5-7b-hf": 7.06,
    "llavanext": 8.36, "llava-hf_llava-onevision-qwen2-7b-ov-hf": 8.03,
}
FAMILY = {"qwen": "qwen", "Qwen_Qwen2.5-VL-7B-Instruct": "qwen", "Qwen_Qwen2-VL-2B-Instruct": "qwen",
          "OpenGVLab_InternVL3-8B-hf": "internvl", "llava-hf_llava-1.5-7b-hf": "llava",
          "llavanext": "llava", "llava-hf_llava-onevision-qwen2-7b-ov-hf": "llava"}


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def item_meta():
    meta = {}
    dmdir = Path(yaml.safe_load((ROOT / "configs/pilot_multidomain_v1.yaml").read_text())["paths"]["dm_dir"])
    files = [dmdir / "splits" / f"{sp}.jsonl" for sp in ("train", "val", "test")]
    files.append(DR / "dm/merged_aokvqa_racehigh/dm_all.jsonl")
    for f in files:
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            meta[r["candidate_id"]] = dict(
                qlen=len(str(r["question"]).split()),
                clen=len(str(r.get("caption_for_filter") or "").split()),
                source=r.get("source") or r.get("source_text_qa") or "aokvqa")
    return meta


def build_cells():
    meta = item_meta()
    viol = set(json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
    rows = []
    for pool, sub in (("dm", DM_SUB), ("assembled", ASM_SUB)):
        for mdir in sorted((DR / "activations").iterdir()):
            m = mdir.name
            if m in EXCLUDED or not (mdir / sub / "letter_logits.npy").exists():
                continue
            ll = np.load(mdir / sub / "letter_logits.npy").astype(np.float64)
            idx = [json.loads(l) for l in (mdir / sub / "index.jsonl").read_text().splitlines() if l.strip()]
            buckets = {}
            for i, r in enumerate(idx):
                cid = r["candidate_id"]
                if (pool == "assembled" and cid in viol) or cid not in meta:
                    continue
                lab = r["label"]
                dom = meta[cid]["source"] if pool == "dm" else ("aokvqa" if lab == "vision" else "race_high")
                if pool == "dm" and dom not in DOMS:
                    continue
                buckets.setdefault((dom, lab), []).append((i, int(r["correct_index"]), cid))
            for (dom, lab), v in buckets.items():
                if len(v) < MIN_N:
                    continue
                ii = np.array([x[0] for x in v]); gold = np.array([x[1] for x in v])
                gc, oc = (1, 2) if lab == "vision" else (2, 1)
                pg = softmax(ll[ii, gc]); pv = softmax(ll[ii, 0])
                pred_g = ll[ii, gc].argmax(1); pred_o = ll[ii, oc].argmax(1); pred_vt = ll[ii, 0].argmax(1)
                ok_g = pred_g == gold
                if ok_g.sum() < MIN_N:
                    continue
                conf = pg.max(1)
                ent = -(pg * np.log(np.clip(pg, 1e-12, 1))).sum(1)
                # 10-bin ECE, grounded condition
                bins = np.clip((conf * 10).astype(int), 0, 9)
                ece = sum(abs(ok_g[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                          for b in range(10) if (bins == b).any())
                rows.append(dict(
                    model=m, family=FAMILY[m], domain=dom, modality=lab, pool=pool,
                    n=len(v), n_solv=int(ok_g.sum()),
                    distraction=float((pred_vt[ok_g] != gold[ok_g]).mean()),
                    grounding=float(ok_g.mean()),
                    size=float(np.log10(SIZES[m] * 1e9)),
                    entropy=float(ent.mean()), confidence=float(conf.mean()), ece=float(ece),
                    off_acc=float((pred_o == gold).mean()),
                    vt_acc=float((pred_vt == gold).mean()),
                    qlen=float(np.mean([meta[x[2]]["qlen"] for x in v])),
                    clen=float(np.mean([meta[x[2]]["clen"] for x in v]))))
    return rows


PRED = ["grounding", "size", "entropy", "confidence", "ece", "off_acc", "vt_acc", "qlen", "clen"]


def main():
    rows = build_cells()
    d = np.array([r["distraction"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    mdl = np.array([r["model"] for r in rows])
    X = {k: np.array([r[k] for r in rows]) for k in PRED}
    print(f"cells: {len(rows)} (models {len(set(mdl))}, families {len(set(fam))})")

    from scipy import stats
    rng = np.random.default_rng(0)
    fams = sorted(set(fam))
    print(f"\n(1) univariate across {len(rows)} cells  [family-clustered bootstrap 95% CI]")
    print(f"    {'predictor':12}{'pearson r':>11}{'CI':>18}{'spearman':>10}")
    uni = {}
    for k in PRED:
        r = float(np.corrcoef(X[k], d)[0, 1]); rho = float(stats.spearmanr(X[k], d).statistic)
        bs = []
        for _ in range(4000):
            pick = rng.choice(len(fams), len(fams), replace=True)
            sel = np.concatenate([np.where(fam == fams[j])[0] for j in pick])
            if X[k][sel].std() > 1e-9 and d[sel].std() > 1e-9:
                bs.append(np.corrcoef(X[k][sel], d[sel])[0, 1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        uni[k] = dict(r=r, ci=[float(lo), float(hi)], rho=rho)
        print(f"    {k:12}{r:>+11.3f}   [{lo:+.2f},{hi:+.2f}]{rho:>+10.3f}")

    print(f"\n(2) leave-one-model-out prediction (single-predictor linear fit)")
    print(f"    {'predictor':12}{'LOMO MAE':>10}{'LOMO R^2':>10}")
    lomo = {}
    for k in PRED:
        errs, preds_all, obs_all = [], [], []
        for m in sorted(set(mdl)):
            tr = mdl != m; te = mdl == m
            if X[k][tr].std() < 1e-9:
                continue
            b, a = np.polyfit(X[k][tr], d[tr], 1)
            pr = a + b * X[k][te]
            errs += list(np.abs(pr - d[te])); preds_all += list(pr); obs_all += list(d[te])
        obs_all = np.array(obs_all); preds_all = np.array(preds_all)
        r2 = 1 - ((preds_all - obs_all) ** 2).sum() / ((obs_all - obs_all.mean()) ** 2).sum()
        lomo[k] = dict(mae=float(np.mean(errs)), r2=float(r2))
        print(f"    {k:12}{np.mean(errs):>10.4f}{r2:>+10.3f}")

    print(f"\n(3) all-predictor standardized OLS + drop-one delta-R^2")
    Z = np.column_stack([(X[k] - X[k].mean()) / X[k].std() for k in PRED])
    y = (d - d.mean()) / d.std()
    A = np.column_stack([np.ones(len(y)), Z])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    r2_full = 1 - (resid ** 2).sum() / (y ** 2).sum()
    print(f"    full R^2 = {r2_full:.3f}")
    print(f"    {'predictor':12}{'std beta':>10}{'dR^2 drop-one':>15}")
    drop = {}
    for j, k in enumerate(PRED):
        keep = [i for i in range(len(PRED)) if i != j]
        A2 = np.column_stack([np.ones(len(y)), Z[:, keep]])
        b2 = np.linalg.lstsq(A2, y, rcond=None)[0]
        r2_wo = 1 - ((y - A2 @ b2) ** 2).sum() / (y ** 2).sum()
        drop[k] = dict(beta=float(beta[j + 1]), dr2=float(r2_full - r2_wo))
        print(f"    {k:12}{beta[j+1]:>+10.3f}{r2_full - r2_wo:>15.4f}")

    out = DR / "diagnostics/predictor_horserace.json"
    out.write_text(json.dumps(dict(n_cells=len(rows), univariate=uni, lomo=lomo,
                                   ols=dict(r2_full=float(r2_full), drop_one=drop)), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
