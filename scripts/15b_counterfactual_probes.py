"""Step 15b: layer-wise modality separability under the counterfactual framing.

Trains L2-logistic probes per layer on the counterfactual activations and
reports val accuracy. Three probe families:

  (A) h_VT(V) vs h_VT(T)        -- original discriminative probe (re-fit on
                                    the dedup'd counterfactual pool for parity).
  (B) h_V vs h_T                -- within-condition: can we tell whether the
                                    activation came from image-only or
                                    caption-only input? Expected to saturate
                                    early; tests input-modality encoding.
  (C) (h_VT-h_V on V-items)     -- causal routing: discriminate the within-
      vs (h_VT-h_T on T-items)     item activation CHANGE caused by adding
                                    the off-modality input. This is the
                                    closest layer-wise probe of "where does
                                    modality routing live."

Each probe is fit on train_pass split and evaluated on val_pass. Output:
  data/probes/qwen/multidomain_v1_counterfactual/per_layer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_probe(X_train, y_train, X_val, y_val):
    """L2-logistic, single fit; return val acc + coef + classes."""
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)
    val_acc = float(clf.score(X_val, y_val))
    train_acc = float(clf.score(X_train, y_train))
    return train_acc, val_acc, clf.coef_, list(map(int, clf.classes_))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir",
                    default="data/activations/qwen/dm_multidomain_v1_counterfactual")
    ap.add_argument("--out-dir",
                    default="data/probes/qwen/multidomain_v1_counterfactual")
    ap.add_argument("--layers", default="all",
                    help="comma-separated, or 'all' for every layer")
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acts = np.load(act_dir / "activations.npy")   # (N, 3, n_layers, d), conditions [VT, V, T]
    index = [json.loads(l) for l in (act_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    N, n_cond, n_layers, d = acts.shape
    assert len(index) == N
    print(f"loaded {N} rows, {n_cond} conditions, {n_layers} layers, d={d}")

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    # split masks. fit on train_pass, eval on val_pass. label per row.
    is_train = np.array([r["split"] == "train_pass" for r in index])
    is_val   = np.array([r["split"] == "val_pass"   for r in index])
    labels   = np.array([0 if r["label"] == "vision" else 1 for r in index])  # 0=V, 1=T
    print(f"train_pass rows: {is_train.sum()}  (V={(is_train & (labels==0)).sum()} "
          f"T={(is_train & (labels==1)).sum()})")
    print(f"val_pass   rows: {is_val.sum()}  (V={(is_val & (labels==0)).sum()} "
          f"T={(is_val & (labels==1)).sum()})")

    # Condition indices
    CI_VT, CI_V, CI_T = 0, 1, 2

    results = {"A_discriminative_VT": [], "B_V_vs_T": [], "C_causal_delta": []}

    for L in layers:
        # --- (A) original-style: h_VT(V) vs h_VT(T) ---
        Xtr_A = acts[is_train, CI_VT, L].astype(np.float32)
        Xva_A = acts[is_val,   CI_VT, L].astype(np.float32)
        ytr   = labels[is_train]; yva = labels[is_val]
        tr_A, va_A, _, _ = fit_probe(Xtr_A, ytr, Xva_A, yva)
        results["A_discriminative_VT"].append({"layer": L, "train_acc": tr_A, "val_acc": va_A})

        # --- (B) within-condition: h_V vs h_T (stacked across labels) ---
        # row contributes h_V (class 0=image-only) AND h_T (class 1=caption-only).
        # train fit on train_pass rows; val on val_pass rows.
        Xtr_B = np.concatenate([acts[is_train, CI_V, L], acts[is_train, CI_T, L]],
                               axis=0).astype(np.float32)
        ytr_B = np.concatenate([np.zeros(is_train.sum(), dtype=np.int64),
                                np.ones(is_train.sum(), dtype=np.int64)])
        Xva_B = np.concatenate([acts[is_val,   CI_V, L], acts[is_val,   CI_T, L]],
                               axis=0).astype(np.float32)
        yva_B = np.concatenate([np.zeros(is_val.sum(), dtype=np.int64),
                                np.ones(is_val.sum(), dtype=np.int64)])
        tr_B, va_B, _, _ = fit_probe(Xtr_B, ytr_B, Xva_B, yva_B)
        results["B_V_vs_T"].append({"layer": L, "train_acc": tr_B, "val_acc": va_B})

        # --- (C) causal-routing delta probe ---
        # for V-items: delta = h_VT - h_V  (text was added)
        # for T-items: delta = h_VT - h_T  (image was added)
        # class 0 = "text was added", class 1 = "image was added"
        v_train_mask = is_train & (labels == 0)
        t_train_mask = is_train & (labels == 1)
        v_val_mask   = is_val   & (labels == 0)
        t_val_mask   = is_val   & (labels == 1)
        dVT_minus_dV_train = (acts[v_train_mask, CI_VT, L].astype(np.float32)
                              - acts[v_train_mask, CI_V, L].astype(np.float32))
        dVT_minus_dT_train = (acts[t_train_mask, CI_VT, L].astype(np.float32)
                              - acts[t_train_mask, CI_T, L].astype(np.float32))
        Xtr_C = np.concatenate([dVT_minus_dV_train, dVT_minus_dT_train], axis=0)
        ytr_C = np.concatenate([np.zeros(dVT_minus_dV_train.shape[0], dtype=np.int64),
                                np.ones(dVT_minus_dT_train.shape[0], dtype=np.int64)])
        dVT_minus_dV_val = (acts[v_val_mask, CI_VT, L].astype(np.float32)
                            - acts[v_val_mask, CI_V, L].astype(np.float32))
        dVT_minus_dT_val = (acts[t_val_mask, CI_VT, L].astype(np.float32)
                            - acts[t_val_mask, CI_T, L].astype(np.float32))
        Xva_C = np.concatenate([dVT_minus_dV_val, dVT_minus_dT_val], axis=0)
        yva_C = np.concatenate([np.zeros(dVT_minus_dV_val.shape[0], dtype=np.int64),
                                np.ones(dVT_minus_dT_val.shape[0], dtype=np.int64)])
        tr_C, va_C, _, _ = fit_probe(Xtr_C, ytr_C, Xva_C, yva_C)
        results["C_causal_delta"].append({"layer": L, "train_acc": tr_C, "val_acc": va_C})

        print(f"  L{L:>2d}  A_VT={va_A:.3f}  B_V_vs_T={va_B:.3f}  C_causal_delta={va_C:.3f}")

    (out_dir / "per_layer.json").write_text(json.dumps(results, indent=2))

    # quick best layer per probe family
    print("\n=== Best layer per probe family (val_acc) ===")
    for k, lst in results.items():
        best = max(lst, key=lambda r: r["val_acc"])
        print(f"  {k:<24s}  best L{best['layer']:>2d}  val_acc={best['val_acc']:.4f}")

    # plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for k, lst in results.items():
            plt.plot([r["layer"] for r in lst], [r["val_acc"] for r in lst],
                     marker="o", label=k)
        plt.xlabel("layer")
        plt.ylabel("val accuracy")
        plt.title("Modality separability per layer — three probe families")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "per_layer.png", dpi=140)
        print(f"\nwrote {out_dir/'per_layer.png'}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
