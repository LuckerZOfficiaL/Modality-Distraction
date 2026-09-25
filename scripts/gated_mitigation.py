"""P2: turn the law into a mitigation — selective modality routing from stored 3-condition logits.

Deployment framing: the model receives (image, context text, question) and does not know which
modality is answer-bearing. Unconditional policies fail one side (always-V kills text-grounded
items, always-VT eats distraction). A per-item GATE decides, from label-free confidence signals,
which condition's answer to trust. All policies are computed from the stored letter logits
(conditions: 0=VT, 1=V, 2=T) — no new inference.

Policies (label-free at decision time):
  vt / v / t     unconditional single-condition baselines
  maxconf        answer of the condition with the largest top1-top2 logit gap
  gated(tau)     trust VT when its gap >= tau, else fall back to the more confident of V / T;
                 tau chosen by 2-fold cross-fitting over items (report test-fold accuracy)
  random(tau)    same fallback rate as gated but flags random items (matched-rate null)
  oracle         upper bound: correct if any condition is correct

    python scripts/gated_mitigation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
POOLS = {"dm": "dm_multidomain_v1_cf", "assembled": "merged_aokvqa_racehigh_cf"}
RNG = np.random.default_rng(0)


def gaps_and_preds(ll):
    s = np.sort(ll, axis=2)
    gap = s[:, :, -1] - s[:, :, -2]          # [n, 3] top1-top2 per condition
    pred = ll.argmax(2)                       # [n, 3]
    return gap, pred


def policy_acc(pred, gap, ci, mode, tau=None, flag_mask=None):
    n = len(ci)
    if mode in ("vt", "v", "t"):
        k = {"vt": 0, "v": 1, "t": 2}[mode]
        return (pred[:, k] == ci).mean()
    if mode == "maxconf":
        k = gap.argmax(1)
        return (pred[np.arange(n), k] == ci).mean()
    if mode == "gated":
        flag = gap[:, 0] < tau if flag_mask is None else flag_mask
        fb = np.where(gap[:, 1] >= gap[:, 2], pred[:, 1], pred[:, 2])   # more confident of V/T
        ans = np.where(flag, fb, pred[:, 0])
        return (ans == ci).mean()
    if mode == "oracle":
        return ((pred == ci[:, None]).any(1)).mean()
    raise ValueError(mode)


def crossfit_gated(pred, gap, ci):
    """2-fold: pick tau on one half (grid = deciles of gap_VT), evaluate on the other."""
    n = len(ci)
    perm = RNG.permutation(n)
    folds = [perm[: n // 2], perm[n // 2:]]
    accs, rates, taus = [], [], []
    for a, b in [(0, 1), (1, 0)]:
        tr, te = folds[a], folds[b]
        grid = np.quantile(gap[tr, 0], np.linspace(0.05, 0.95, 19))
        best = max(grid, key=lambda t: policy_acc(pred[tr], gap[tr], ci[tr], "gated", tau=t))
        accs.append(policy_acc(pred[te], gap[te], ci[te], "gated", tau=best))
        rates.append(float((gap[te, 0] < best).mean()))
        taus.append(float(best))
    return float(np.mean(accs)), float(np.mean(rates)), taus


def matched_random(pred, gap, ci, rate, reps=200):
    n = len(ci)
    fb = np.where(gap[:, 1] >= gap[:, 2], pred[:, 1], pred[:, 2])
    accs = []
    for _ in range(reps):
        flag = RNG.random(n) < rate
        ans = np.where(flag, fb, pred[:, 0])
        accs.append((ans == ci).mean())
    return float(np.mean(accs))


def main() -> None:
    models = sorted(p.name for p in (DR / "activations").iterdir()
                    if all((p / sub / "letter_logits.npy").exists() for sub in POOLS.values())
                    and p.name not in EXCLUDED)
    print(f"backbones: {len(models)}")
    OUT = {}
    for pool, sub in POOLS.items():
        print(f"\n================ pool={pool} ================")
        hdr = (f"{'model':42}| {'VT':>6} {'V':>6} {'T':>6} {'maxcf':>6} | "
               f"{'gated':>6} {'rate':>5} {'rand':>6} | {'oracle':>6}")
        print(hdr); print("-" * len(hdr))
        agg = []
        for mo in models:
            st = DR / "activations" / mo / sub
            ll = np.load(st / "letter_logits.npy").astype(np.float64)
            idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
            ci = np.array([r["correct_index"] for r in idx])
            gap, pred = gaps_and_preds(ll)
            base = {m: policy_acc(pred, gap, ci, m) for m in ("vt", "v", "t", "maxconf", "oracle")}
            g_acc, g_rate, _ = crossfit_gated(pred, gap, ci)
            r_acc = matched_random(pred, gap, ci, g_rate)
            print(f"{mo:42}| {base['vt']:>6.3f} {base['v']:>6.3f} {base['t']:>6.3f} "
                  f"{base['maxconf']:>6.3f} | {g_acc:>6.3f} {g_rate:>5.2f} {r_acc:>6.3f} | {base['oracle']:>6.3f}")
            agg.append((base["vt"], base["maxconf"], g_acc, r_acc, base["oracle"]))
        A = np.array(agg)
        print("-" * len(hdr))
        print(f"{'MEAN':42}| {A[:,0].mean():>6.3f} {'':>6} {'':>6} {A[:,1].mean():>6.3f} | "
              f"{A[:,2].mean():>6.3f} {'':>5} {A[:,3].mean():>6.3f} | {A[:,4].mean():>6.3f}")
        print(f"  gated - VT: {(A[:,2]-A[:,0]).mean():+.3f} (mean)   "
              f"gated - random: {(A[:,2]-A[:,3]).mean():+.3f}   "
              f"recovered fraction of oracle headroom: {((A[:,2]-A[:,0])/np.maximum(A[:,4]-A[:,0],1e-9)).mean():.2f}")
        OUT[pool] = {"vt_mean": float(A[:, 0].mean()), "oracle_mean": float(A[:, 4].mean()),
                     "headroom": float((A[:, 4] - A[:, 0]).mean()),
                     "gated_minus_vt": float((A[:, 2] - A[:, 0]).mean()),
                     "gated_minus_random": float((A[:, 2] - A[:, 3]).mean())}
    out = DR / "diagnostics" / "gated_mitigation.json"
    out.write_text(json.dumps(OUT, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
