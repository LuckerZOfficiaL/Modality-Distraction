"""X4: logit-lens answer trajectory on distracted vs robust vision items.

Near-offline: apply the model's final norm + lm_head to the STORED per-layer h_VT
residuals (no forward pass) and read, at each layer, the option-letter distribution.
Track P(correct) and P(model's final V+T answer) across all 36 layers. On distracted
items (V-only correct, V+T wrong) this shows WHEN the caption-induced wrong answer
overtakes the correct one --- a per-item view of the commit depth localized causally
at L26-28. Robust items (V+T correct) are the control: P(correct) should stay on top.

Loads the model on CPU only to borrow `norm` + `lm_head` (compute is tiny matmuls).

    python scripts/45_logit_lens_distraction.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s65", _ROOT / "65_collect_cf_anymodel.py")
s65 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s65)  # type: ignore


def find_final_norm(model):
    """Locate the final pre-head norm across VLM families (for the logit-lens unembed)."""
    for path in ("model.language_model.norm", "model.model.language_model.norm", "model.model.norm",
                 "language_model.model.norm", "language_model.norm", "model.norm"):
        obj = model
        for a in path.split("."):
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "weight"):
            return obj
    raise RuntimeError("could not locate final norm for logit-lens (add this family's path to find_final_norm)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_cf")
    ap.add_argument("--model", default="qwen", help="HF path or REGISTRY key (see scripts/61); cf-store dir = key")
    ap.add_argument("--key", default=None, help="cf-store key (default sanitized --model; use to match collection, e.g. qwen)")
    ap.add_argument("--out", default=None, help="default data/diagnostics/logit_lens/{key}[_<cf>].json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep", action="store_true", help="keep HF checkpoint (default delete after, hf/qwen)")
    ap.add_argument("--force", action="store_true", help="recompute even if the output json exists")
    args = ap.parse_args()

    import re
    cfg = yaml.safe_load(Path(args.config).read_text())
    key = args.key or re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    tag = "_assembled" if ("aokvqa" in args.cf_subdir or "merged" in args.cf_subdir) else ""
    out = Path(args.out or f"data/diagnostics/logit_lens/{key}{tag}.json")
    if out.exists() and not args.force:   # resumable: skip BEFORE downloading/loading the model
        print(f"[skip] {out} exists (use --force to recompute)"); return
    st = Path(cfg["paths"]["data_root"]) / "activations" / key / args.cf_subdir
    acts = np.load(st / "activations.npy", mmap_mode="r")   # (N,3,Lyr,d) [VT,V,T]
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    vt_pred = ll[:, 0].argmax(1); v_ok = ll[:, 1].argmax(1) == ci
    n_layers = acts.shape[2]

    # reuse the exact per-family loader the cf-store was collected with (so lm_head/norm match the residuals)
    spec = s65.resolve(args.model)
    model, _prep, letter_ids, del_id = s65.setup(spec, args.device)
    model.eval()
    lm_head = model.get_output_embeddings()       # final unembed (HF-standard across families)
    norm = find_final_norm(model)                 # final norm before head
    letter_ids = letter_ids.cpu()
    dev = lm_head.weight.device

    def trajectory(sel_ix):
        """mean P(correct) and P(final V+T pred) over layers, restricted to option letters."""
        pc = np.zeros(n_layers); pf = np.zeros(n_layers)
        with torch.no_grad():
            for L in range(n_layers):
                h = torch.tensor(acts[sel_ix, 0, L].astype(np.float32)).to(dev, lm_head.weight.dtype)
                logits = lm_head(norm(h)).float().cpu()                    # (n,vocab)
                for r, i in enumerate(sel_ix):
                    k = len(idx[i].get("options", [])) or 4
                    optlog = logits[r, letter_ids[:k]]
                    p = torch.softmax(optlog, -1)
                    pc[L] += p[ci[i]].item()
                    pf[L] += p[vt_pred[i]].item()
        return pc / len(sel_ix), pf / len(sel_ix)

    # need options length per row -> join from splits
    dm = Path(cfg["paths"]["dm_dir"]); meta = {}
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); meta[r["candidate_id"]] = r
    for i, r in enumerate(idx):
        idx[i]["options"] = meta.get(r["candidate_id"], {}).get("options", [0, 1, 2, 3])

    distr = np.where((lab == "vision") & v_ok & (vt_pred != ci))[0]
    robust = np.where((lab == "vision") & v_ok & (vt_pred == ci))[0]
    print(f"distracted n={len(distr)}  robust n={len(robust)}")
    pc_d, pf_d = trajectory(distr)
    pc_r, pf_r = trajectory(robust)

    # commit layer = first layer where the answer EMERGES from the chance regime
    # (a decisive option prob), not the near-uniform early-layer noise.
    # commit = first layer where the ROBUST answer crystallizes (P(correct) clears
    # the chance regime); using pf_d here mis-fires (distracted-wrong can be high early).
    emerge_thr = 0.50
    commit = next((L for L in range(n_layers) if pc_r[L] > emerge_thr), int(np.argmax(pc_r)))
    print("\n=== logit-lens trajectory (P over option letters, by layer) ===")
    print(f"{'L':>3} | {'distr P(correct)':>16} {'distr P(final-wrong)':>20} | {'robust P(correct)':>17}")
    for L in range(n_layers):
        mark = "  <- answer crystallizes" if L == commit else ""
        print(f"{L:>3} | {pc_d[L]:>16.3f} {pf_d[L]:>20.3f} | {pc_r[L]:>17.3f}{mark}")
    print(f"\nAnswer crystallizes at L{commit} (chance before). At commit: distracted commits to the "
          f"WRONG caption-answer (P {pf_d[commit]:.2f} vs correct {pc_d[commit]:.2f}); "
          f"robust commits correct (P {pc_r[commit]:.2f}).")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_distracted": len(distr), "n_robust": len(robust),
        "commit_layer": commit,
        "distr_p_correct": pc_d.tolist(), "distr_p_final": pf_d.tolist(),
        "robust_p_correct": pc_r.tolist()}, indent=2))
    print(f"-> {out}")
    if del_id and not args.keep:
        del model; torch.cuda.empty_cache()
        try:
            s65.s61.delete_hf_cache(del_id)   # delete_hf_cache lives in scripts/61 (s65 imports it as s61)
        except Exception as e:
            print(f"[warn] could not delete HF cache for {del_id}: {e}")


if __name__ == "__main__":
    main()
