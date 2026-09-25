"""P6b: dense positive control for the SAE-ablation negative result (tab:ablation).

Same protocol as scripts/causal_ablation.py (same distracted items, same layer, same last-token edit point on the
V+T forward, same matched-null discipline), but the edit is DENSE instead of sparse:

  --edit probe    rank-1 projection-out of the distraction-probe direction u
                      h' = h - (h.u) u
                  (u = logistic probe on h_VT@L, distracted vs V/pass, trained in-script;
                   null = random unit direction, same rank-1 edit)
  --edit patch    per-item counterfactual patch: replace the last-token residual by that item's
                  stored V-only residual, h' = h_V@L (upper bound: full dense replacement;
                  null = another random distracted item's h_V, matched replacement)

If dense recovers where the sparse top-K did not (scripts/causal_ablation.py), the negative upgrades to
"controllable at this site, but not through the SAE basis".

  CUDA_VISIBLE_DEVICES=0 python scripts/dense_control.py --model qwen --layer 28 --edit probe
  CUDA_VISIBLE_DEVICES=0 python scripts/dense_control.py --model qwen --layer 28 --edit patch
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from moground.steering import get_letter_token_ids

_spec = importlib.util.spec_from_file_location("_c6", Path(__file__).parent / "collect_dm_counterfactual_activations.py")
_c6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c6)  # type: ignore
_s56 = importlib.util.spec_from_file_location("_s56", Path(__file__).parent / "causal_ablation.py")
s56 = importlib.util.module_from_spec(_s56); _s56.loader.exec_module(s56)  # type: ignore
build_messages = _c6.build_messages
MAXPX = 1024 * 1024


@torch.inference_mode()
def predict_dense(model, processor, row, layer, edit_vec, mode, letter_ids, device):
    """V+T forward with a dense last-token edit at `layer`.
    mode="proj": h' = h - (h.u)u with u=edit_vec (unit). mode="patch": h' = edit_vec."""
    msgs = build_messages(row, "VT", MAXPX)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    if imgs is not None and len(imgs) == 0:
        imgs = None
    inp = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(device)
    li = int(inp.attention_mask[0].sum().item() - 1)
    handle = None
    if edit_vec is not None:
        layers = model.model.language_model.layers
        def hook(mod, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            x = h[0, li].float()
            if mode == "proj":
                x = x - (x @ edit_vec) * edit_vec
            else:  # patch
                x = edit_vec
            h[0, li] = x.to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        handle = layers[layer].register_forward_hook(hook)
    try:
        logits = model(**inp).logits[0, li]
    finally:
        if handle:
            handle.remove()
    k = len(row["options"])
    return int(logits[letter_ids[:k]].argmax().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual_full")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--edit", choices=["probe", "patch"], default="probe")
    ap.add_argument("--n-pass", type=int, default=150)
    ap.add_argument("--keep-ids", default=None, help="json with an id list to restrict to")
    ap.add_argument("--keep-key", default="heldout")
    ap.add_argument("--rows-file", default=None,
                    help="explicit item jsonl (e.g. the assembled pool); overrides --splits")
    ap.add_argument("--splits", default="train,val,test",
                    help="restrict the item pool to these MoGround splits (comma-sep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    st = dr / "activations" / args.model / args.cf_subdir
    ll = np.load(st / "letter_logits.npy")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in idx]); lab = np.array([r["label"] for r in idx])
    v_ok = ll[:, 1].argmax(1) == ci; vt_ok = ll[:, 0].argmax(1) == ci
    tag = (("_asm" + {"heldout": "held", "dev": "dev"}.get(args.keep_key, args.keep_key)) if (args.rows_file and args.keep_ids) else "_asm" if args.rows_file else ("" if args.splits == "train,val,test"
           else "_" + args.splits.replace(",", "")))
    rowmap = {}
    if args.rows_file:                        # assembled pool (or any explicit item file)
        for l in Path(args.rows_file).read_text().splitlines():
            if l.strip():
                r = json.loads(l); rowmap[r["candidate_id"]] = r
    else:
        for sp in args.splits.split(","):      # default all; use val,test for cross-method comparison
            p = dm / "splits" / f"{sp}.jsonl"
            if p.exists():
                for l in p.read_text().splitlines():
                    if l.strip():
                        r = json.loads(l); rowmap[r["candidate_id"]] = r
    if args.keep_ids:                          # e.g. assembled held-out half, certified only
        keep = set(json.loads(Path(args.keep_ids).read_text())[args.keep_key])
        viol = set(json.loads((dr / "diagnostics/assembled_caption_certification.json").read_text())["violations"])
        rowmap = {k: v for k, v in rowmap.items() if k in keep and k not in viol}

    distracted = [i for i in range(len(idx)) if lab[i] == "vision" and v_ok[i] and not vt_ok[i]
                  and idx[i]["candidate_id"] in rowmap]
    vpass = [i for i in range(len(idx)) if lab[i] == "vision" and v_ok[i] and vt_ok[i]
             and idx[i]["candidate_id"] in rowmap]
    rng = np.random.default_rng(args.seed); rng.shuffle(vpass); vpass = vpass[:args.n_pass]
    print(f"{args.model} L{args.layer} edit={args.edit}: distracted={len(distracted)} V/pass={len(vpass)}")

    acts = np.load(st / "activations.npy", mmap_mode="r")
    dev = args.device

    def unit(v):
        v = v / (v.norm() + 1e-8)
        return v.to(dev)

    if args.edit == "probe":
        from sklearn.linear_model import LogisticRegression
        Xd = np.asarray(acts[distracted, 0, args.layer], dtype=np.float32)
        Xr = np.asarray(acts[vpass, 0, args.layer], dtype=np.float32)
        X = np.concatenate([Xd, Xr]); y = np.array([1] * len(Xd) + [0] * len(Xr))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.05).fit((X - mu) / sd, y)
        u = unit(torch.tensor(clf.coef_[0] / sd, dtype=torch.float32))
        nu = unit(torch.tensor(rng.standard_normal(u.shape[0]), dtype=torch.float32))
        edits = {"real": (u, "proj"), "null": (nu, "proj")}
    else:  # patch: per-item h_V; null = shuffled other-item h_V
        perm = rng.permutation(len(distracted))
        edits = {"real": ("patch_self", "patch"), "null": ("patch_shuf", "patch")}

    model, processor = s56.load_model(args.model, dev); model.eval()
    letter_ids = get_letter_token_ids(processor, dev, n_letters=6)

    def run(items, which, kind):
        rec = 0
        for j, i in enumerate(tqdm(items, desc=f"{args.edit}/{which}", leave=False)):
            if kind == "proj":
                ev = edits[which][0]
            else:
                src = i if which == "real" or items is not distracted else i
                if which == "real":
                    ev = torch.tensor(np.asarray(acts[i, 1, args.layer], dtype=np.float32), device=dev)
                else:
                    ev = torch.tensor(np.asarray(acts[distracted[perm[j % len(perm)]], 1, args.layer],
                                                 dtype=np.float32), device=dev)
            pred = predict_dense(model, processor, rowmap[idx[i]["candidate_id"]], args.layer,
                                 ev, kind, letter_ids, dev)
            rec += int(pred == ci[i])
        return rec / max(len(items), 1)

    kind = edits["real"][1]
    rr = run(distracted, "real", kind)
    rn = run(distracted, "null", kind)
    vp = run(vpass, "real", kind)
    print(f"\n=== dense control @ L{args.layer} ({args.model}, edit={args.edit}) ===")
    print(f"recovery real={rr:.3f}  null={rn:.3f}  Δ={rr-rn:+.3f}   V/pass preserved={vp:.3f}")
    out = dr / "diagnostics" / f"dense_control_{args.model}_L{args.layer}{tag}_{args.edit}.json"
    out.write_text(json.dumps({"model": args.model, "layer": args.layer, "edit": args.edit,
                               "n_distracted": len(distracted), "recovery_real": rr,
                               "recovery_null": rn, "vpass_preserved": vp}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
