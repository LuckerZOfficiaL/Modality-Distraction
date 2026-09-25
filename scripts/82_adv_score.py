"""Score the misleading-context cell (scripts/81 -> data/eval/t8_adv).

All rows are TEST vision items given a fluent same-scene caption asserting a WRONG
option (adv_target_index). Three conditions: pred_v (image only, clean reference),
pred_vt (image + misleading caption), pred_t (caption alone).

Reports per backbone:
  acc_v   clean image-only accuracy
  acc_vt  accuracy with the misleading caption present
  drop    acc_v - acc_vt                              (interference from misleading text)
  cap_vt  P(pred_vt == adv_target)                    (actively fooled INTO the lie)
  cap_v   P(pred_v  == adv_target)                    (baseline pull, should be ~chance)
  flip    P(pred_vt wrong | pred_v correct)           (conditional on solvable-clean)
  ->adv   P(pred_vt == adv_target | pred_v correct)   (of those, how many land on the lie)
  cap_t   P(pred_t  == adv_target)                    (caption's standalone persuasiveness)
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

files = sorted(glob.glob("data/eval/t8_adv/*.jsonl"))
hdr = f"{'model':30}| {'n':>3} {'acc_v':>6} {'acc_vt':>6} {'drop':>6} | {'cap_v':>6} {'cap_vt':>6} | {'flip':>6} {'->adv':>6} | {'cap_t':>6}"
print(hdr); print("-" * len(hdr))
rowsout = []
for fp in files:
    model = Path(fp).stem
    R = [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]
    ci  = np.array([r["correct_index"]   for r in R])
    at  = np.array([r["adv_target_index"] for r in R])
    vt  = np.array([r["pred_vt"] for r in R]); v = np.array([r["pred_v"] for r in R])
    t   = np.array([r["pred_t"]  for r in R])
    n = len(R)
    acc_v  = (v == ci).mean(); acc_vt = (vt == ci).mean()
    cap_v  = (v == at).mean(); cap_vt = (vt == at).mean()
    cap_t  = (t == at).mean()
    solv = v == ci
    flip = ((vt != ci) & solv).sum() / max(solv.sum(), 1)
    toadv = ((vt == at) & solv).sum() / max(solv.sum(), 1)
    print(f"{model:30}| {n:>3} {acc_v:>6.3f} {acc_vt:>6.3f} {acc_v-acc_vt:>+6.3f} "
          f"| {cap_v:>6.3f} {cap_vt:>6.3f} | {flip:>6.3f} {toadv:>6.3f} | {cap_t:>6.3f}")
    rowsout.append((model, acc_v-acc_vt, cap_vt-cap_v))
print("-" * len(hdr))
drops = np.array([r[1] for r in rowsout]); caps = np.array([r[2] for r in rowsout])
print(f"{'MEAN across backbones':30}| {'':>3} {'':>6} {'':>6} {drops.mean():>+6.3f} "
      f"| {'':>6} {caps.mean():>+6.3f} (cap_vt-cap_v)")
