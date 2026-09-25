"""X2: is distractability predictable from the RAW representation (baseline for X3)?

X3 showed the SAE *code* of h_VT predicts distractability (L28 AUROC 0.711). The
question X2 answers: does the SAE concentrate a signal that is otherwise diffuse, or
is distractability already trivially decodable from the raw residual? We probe the raw
h_V and h_VT residuals per layer (same items, same CV + permutation null as X3) and
compare AUROC to the SAE-code probe.

Offline: cf-store residuals + letter_logits.

    python scripts/46_distraction_raw_probe.py --layers 13,20,28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--layers", default="13,20,28")
    ap.add_argument("--n-perms", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    st = Path(cfg["paths"]["data_root"]) / "activations" / args.model / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    v_ok = ll[:, 1].argmax(1) == ci; vt_ok = ll[:, 0].argmax(1) == ci
    sel = np.where((lab == "vision") & v_ok)[0]
    y = (~vt_ok[sel]).astype(int)
    n_layers = acts.shape[2]
    # Layer spec accepts absolute indices ("13,20,28", the cohort's convention -- unchanged) OR
    # depth FRACTIONS ("0.46,0.71,0.96"), detected by the presence of a decimal point.
    # Why fractions matter: the absolute default was tuned for 28-layer models, where 13/20/28 sit
    # the probe never sees the second half of the network, and AUROC was still rising at
    # the deepest probed layer. Absolute indices are NOT comparable across models of different
    # depth; report which convention was used alongside any cross-model comparison.
    toks = [t.strip() for t in args.layers.split(",") if t.strip()]
    if any("." in t for t in toks):
        layers = sorted({min(max(int(round(float(t) * (n_layers - 1))), 0), n_layers - 1) for t in toks})
        print(f"[layers] depth-relative spec {toks} -> absolute {layers} of {n_layers} layers")
    else:
        layers = sorted({min(int(t), n_layers - 1) for t in toks})  # clamp (28-layer models)
    # never probe a layer whose stored activations are corrupt (fp16 saturation on huge-activation
    # models); silently returning a probe fit on inf/NaN would be worse than dropping the layer.
    bad = [L for L in layers if not np.isfinite(np.asarray(acts[: min(64, acts.shape[0]), :, L])).all()]
    if bad:
        print(f"[layers] WARNING dropping non-finite layers {bad} (saturated store; re-collect with --act-dtype float32)")
        layers = [L for L in layers if L not in bad]
        if not layers:
            raise SystemExit("all requested layers are non-finite; re-collect the cf-store in float32")
    print(f"vision V-solvable n={len(sel)}  distracted={int(y.sum())}  layers={layers}\n")

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.3,
                                                   class_weight="balanced", solver="liblinear"))
    rng = np.random.default_rng(0)
    print(f"{'layer':>6}{'cond':>6}{'CV-AUROC':>10}{'perm-null':>11}{'Δ':>8}   (vs X3 SAE-code probe)")
    results = []
    for L in layers:
        for cond, nm in [(0, "h_VT"), (1, "h_V")]:
            X = acts[sel, cond, L].astype(np.float32)
            real = cross_val_score(clf(), X, y, cv=skf, scoring="roc_auc", n_jobs=-1).mean()
            null = np.mean([cross_val_score(clf(), X, rng.permutation(y), cv=skf,
                                            scoring="roc_auc", n_jobs=-1).mean()
                            for _ in range(args.n_perms)])
            print(f"{L:>6}{nm:>6}{real:>10.3f}{null:>11.3f}{real-null:>+8.3f}")
            results.append({"layer": L, "cond": nm, "auroc": float(real), "null": float(null)})

    peak = max(results, key=lambda r: r["auroc"])
    tag = "_assembled" if ("aokvqa" in args.cf_subdir or "merged" in args.cf_subdir) else ""
    out = Path(cfg["paths"]["data_root"]) / "diagnostics" / "x2" / f"{args.model}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "cf_subdir": args.cf_subdir,
        "n_solvable": int(len(sel)), "n_distracted": int(y.sum()),
        "peak_auroc": peak["auroc"], "peak_layer": peak["layer"], "peak_cond": peak["cond"],
        "peak_null": peak["null"], "all": results}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
