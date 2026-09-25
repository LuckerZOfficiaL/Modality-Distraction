"""Logit-margin analysis of the misleading-context cell (data/eval/t8_adv).

For each vision item we have 4-way logits under three conditions. The correct-vs-lie
margin m = logit[correct] - logit[adv_target] tells us how firmly the model holds the
truth over the asserted lie.

  m_v    clean image-only margin        (image evidence for truth)
  m_vt   image + misleading caption
  dm     m_v - m_vt                      (margin the caption erodes)
  m_t    caption-alone margin           (how hard the text pushes the lie; expect << 0)

Additivity probe: if vt were a plain sum of channels, m_vt ~= m_v + (m_t - 0).
resid = m_vt - (m_v + m_t) is the image's nonlinear "defense" (>0 = image protects
beyond additive). Also correlate clean margin m_v with robustness (lower drop).
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

files = sorted(glob.glob("data/eval/t8_adv/*.jsonl"))
hdr = f"{'model':30}| {'m_v':>7} {'m_vt':>7} {'dm':>7} {'m_t':>7} | {'resid':>7} | {'drop':>6}"
print(hdr); print("-" * len(hdr))
agg = []
for fp in files:
    model = Path(fp).stem
    R = [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]
    R = [r for r in R if "logits_vt" in r]
    if not R:
        print(f"{model:30}| (no logits)"); continue
    ci = np.array([r["correct_index"] for r in R]); at = np.array([r["adv_target_index"] for r in R])
    Lvt = np.array([r["logits_vt"] for r in R]); Lv = np.array([r["logits_v"] for r in R])
    Lt = np.array([r["logits_t"] for r in R])
    idx = np.arange(len(R))
    m_v  = Lv[idx, ci]  - Lv[idx, at]
    m_vt = Lvt[idx, ci] - Lvt[idx, at]
    m_t  = Lt[idx, ci]  - Lt[idx, at]
    resid = m_vt - (m_v + m_t)
    drop = (np.array([r["pred_v"] for r in R]) == ci).mean() - (np.array([r["pred_vt"] for r in R]) == ci).mean()
    print(f"{model:30}| {m_v.mean():>7.2f} {m_vt.mean():>7.2f} {(m_v-m_vt).mean():>7.2f} "
          f"{m_t.mean():>7.2f} | {resid.mean():>7.2f} | {drop:>+6.3f}")
    agg.append((m_v.mean(), resid.mean(), drop))
print("-" * len(hdr))
A = np.array(agg)
if len(A) > 2:
    print(f"corr(clean margin m_v, drop)     = {np.corrcoef(A[:,0], A[:,2])[0,1]:+.3f}  "
          "(negative => firmer clean belief resists the lie)")
    print(f"corr(image defense resid, drop)  = {np.corrcoef(A[:,1], A[:,2])[0,1]:+.3f}  "
          "(negative => stronger nonlinear image defense resists the lie)")
