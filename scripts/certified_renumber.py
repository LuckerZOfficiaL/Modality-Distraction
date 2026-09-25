"""Option A: make the CERTIFIED assembled pool the paper's primary basis.

Drops the 239 caption-violation vision items (data/diagnostics/assembled_caption_certification.json,
scripts pilot+full Sonnet audit) from every assembled-pool statistic:
  1. regenerates data/diagnostics/grounding_strength.json (80 cells, 8 paper models; assembled
     vision cells on certified items only; backup of full-pool version kept alongside)
  2. reprints every number the draft quotes that depends on the assembled pool:
     law r (all/deconf/vision), LOCO R2, LOMO MAE range, margin-model assembled stats
     (corr(delta,m), margin->flip AUROC), Table-3 per-model rows, asym~gap r.

    python scripts/certified_renumber.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
MODELS = ["qwen", "Qwen_Qwen2.5-VL-7B-Instruct", "Qwen_Qwen2-VL-2B-Instruct",
          "llava-hf_llava-onevision-qwen2-7b-ov-hf", "OpenGVLab_InternVL3-8B-hf"]

_s = importlib.util.spec_from_file_location("s54", ROOT / "scripts/grounding_strength.py")
s54 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s54)  # type: ignore

VIOL = set(json.load(open(DR / "diagnostics/assembled_caption_certification.json"))["violations"])


def cells_certified(path):
    """assembled cells with violator vision items dropped (text side untouched)."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip() and json.loads(l)["candidate_id"] not in VIOL]
    tmp = Path("/tmp/claude-1000/-home-ubuntu-thesis/ad3df912-4d6c-45ba-bff9-dd1d05eb60dc/scratchpad/_cert_tmp.jsonl")
    tmp.write_text("\n".join(json.dumps(r) for r in rows))
    return s54.cells_from(tmp, by_source=False)


def main() -> None:
    # 1. regenerate grounding_strength.json
    src = DR / "diagnostics/grounding_strength.json"
    bak = DR / "diagnostics/grounding_strength_fullpool.json"
    if not bak.exists():
        shutil.copy(src, bak)
    pts = []
    for m in MODELS:
        for c in s54.cells_from(DR / f"eval/t8/{m}.jsonl", by_source=True):
            c.update(model=m, dataset="dm"); pts.append(c)
        for c in cells_certified(DR / f"eval/t8_assembled/{m}.jsonl"):
            c.update(model=m, dataset="assembled"); pts.append(c)
    assert len(pts) == 80, f"expected 80 cells, got {len(pts)}"
    src.write_text(json.dumps(pts, indent=2))
    print(f"regenerated {src} (80 cells, assembled=certified; backup {bak.name})")

    # 2. law numbers
    g = np.array([p["ground"] for p in pts]); d = np.array([p["distr"] for p in pts])
    dm_text = np.array([(p["dataset"] == "dm") and (p["modality"] == "text") for p in pts])
    vis = np.array([p["modality"] == "vision" for p in pts])
    print(f"\nlaw r ALL       = {np.corrcoef(g, d)[0,1]:+.4f}")
    print(f"law r deconf    = {np.corrcoef(g[~dm_text], d[~dm_text])[0,1]:+.4f}")
    print(f"law r vision    = {np.corrcoef(g[vis], d[vis])[0,1]:+.4f}")
    preds = np.zeros(len(pts))
    for i in range(len(pts)):
        msk = np.arange(len(pts)) != i; b = np.polyfit(g[msk], d[msk], 1); preds[i] = b[0]*g[i] + b[1]
    print(f"LOCO R2         = {1 - ((preds-d)**2).sum()/((d-d.mean())**2).sum():.3f}")
    model = np.array([p["model"] for p in pts]); maes = []
    for M in MODELS:
        tr = model != M; bb = np.polyfit(g[tr], d[tr], 1)
        maes.append(float(np.abs((bb[0]*g[model == M] + bb[1]) - d[model == M]).mean()))
    print(f"LOMO MAE range  = {min(maes):.3f} - {max(maes):.3f}")

    # 3. margin-model assembled stats on certified items
    from sklearn.metrics import roc_auc_score
    mvs, deltas, flips = [], [], []
    for m in MODELS:
        st = DR / "activations" / m / "merged_aokvqa_racehigh_cf"
        ll = np.load(st / "letter_logits.npy")
        idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
        for i, r in enumerate(idx):
            if r["label"] != "vision" or r["candidate_id"] in VIOL:
                continue
            ci = int(r["correct_index"])
            lv = ll[i, 1].astype(np.float64); lvt = ll[i, 0].astype(np.float64)
            if lv.argmax() != ci:
                continue
            mv = lv[ci] - np.delete(lv, ci).max(); mvt = lvt[ci] - np.delete(lvt, ci).max()
            mvs.append(mv); deltas.append(mvt - mv); flips.append(int(mvt < 0))
    mvs, deltas, flips = map(np.array, (mvs, deltas, flips))
    print(f"\nmargin assembled (certified): corr(delta,m) = {np.corrcoef(deltas, mvs)[0,1]:+.3f}   "
          f"margin->flip AUROC = {roc_auc_score(flips, -mvs):.3f}   (n={len(mvs)})")

    # 4. Table-3 rows (certified)
    print(f"\n{'model':44} {'Vg':>6} {'Tg':>6} {'gap':>7} {'vd':>6} {'td':>6} {'asym':>7}")
    for m in MODELS:
        rows = [json.loads(l) for l in (DR / f"eval/t8_assembled/{m}.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["candidate_id"] not in VIOL]
        ci = np.array([r["correct_index"] for r in rows]); lab = np.array([r["label"] for r in rows])
        vt = np.array([r["pred_vt"] for r in rows]); v = np.array([r["pred_v"] for r in rows])
        t = np.array([r["pred_t"] for r in rows])
        isV, isT = lab == "vision", lab == "text"
        vg = (v[isV] == ci[isV]).mean(); tg = (t[isT] == ci[isT]).mean()
        vs = isV & (v == ci); ts = isT & (t == ci)
        vd = (vs & (vt != ci)).sum() / max(vs.sum(), 1); td = (ts & (vt != ci)).sum() / max(ts.sum(), 1)
        print(f"{m:44} {vg:>6.3f} {tg:>6.3f} {tg-vg:>+7.3f} {vd:>6.3f} {td:>6.3f} {vd-td:>+7.3f}")


if __name__ == "__main__":
    main()
