"""#2 Causal test: are the X3 distraction-susceptibility features CAUSAL or merely
DIAGNOSTIC? On distracted items (V-only correct, V+T wrong), ablate the
distraction-associated SAE features (the ones that fire MORE on distracted items) from
the V+T residual at the commit layer and measure recovery (flip back to correct).
Compare against a matched-random-alive-feature null, and track V/pass collateral.

Honest framing (per our steering history): we EXPECT this may be ~null vs the random
null -- a clean negative ("diagnostic, not a clean lever") is the likely, publishable
outcome. Do not report recovery without the matched null.

Ablation removes the decoded contribution of the chosen features:
    h' = h - z[feats] @ W_dec[feats]      (TopK code z at the last token)

Runs on Qwen now (diversified SAE + X3 features exist); for LLaVA backbones pass their
--model / --sae-variant / --layer once the diversified SAEs are trained and X3 (scripts/sae_distraction_features.py)
has been run to produce layer{L}_top.json.

    CUDA_VISIBLE_DEVICES=N python scripts/causal_ablation.py --model qwen \
        --cf-subdir dm_multidomain_v1_counterfactual_full --sae-variant diversified_v1 --layer 28
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

from moground.sae import TopKSAE, SAEConfig
from moground.steering import get_letter_token_ids

_spec = importlib.util.spec_from_file_location("_c6", Path(__file__).parent / "collect_dm_counterfactual_activations.py")
_c6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c6)  # type: ignore
build_messages = _c6.build_messages
MAXPX = 1024 * 1024


def load_model(name, device):
    if name == "qwen":
        from moground.models import load_qwen_vl
        return load_qwen_vl(device=device)
    if name == "llavanext":
        from moground.models_llavanext import load_llavanext
        return load_llavanext(device=device)


@torch.inference_mode()
def predict(model, processor, row, layer, sae, feats, letter_ids, device, n_layers):
    """V+T forward; if feats given, ablate their decoded contribution at last token of `layer`."""
    msgs = build_messages(row, "VT", MAXPX)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    if imgs is not None and len(imgs) == 0:
        imgs = None
    inp = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(device)
    li = int(inp.attention_mask[0].sum().item() - 1)
    handle = None
    if feats is not None and len(feats):
        layers = model.model.language_model.layers
        def hook(mod, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            x = h[0, li].float()
            z = sae(x.unsqueeze(0))["z"][0]                      # (d_sae,)
            contrib = z[feats] @ sae.W_dec[feats]                # (d_model,)
            h[0, li] = (x - contrib).to(h.dtype)
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
    ap.add_argument("--sae-variant", default="diversified_v1")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n-pass", type=int, default=150, help="V/pass sample for collateral")
    ap.add_argument("--n-feats-list", default="6,20,50", help="feature-count sweep (top-K by Δfr)")
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
    print(f"{args.model}: distracted={len(distracted)}  V/pass sample={len(vpass)}")

    ck = torch.load(dr / "sae" / args.model / args.sae_variant / f"layer{args.layer:02d}" / "sae.pt",
                    map_location=args.device, weights_only=False)
    sae = TopKSAE(SAEConfig(d_model=ck["d_model"], d_sae=ck["d_sae"], k=ck["k"]))
    sae.load_state_dict(ck["state_dict"]); sae.to(args.device).eval()

    # Rank distraction features internally by Δfr = fr(distracted) - fr(robust) on h_VT@L,
    # so the sweep can take any top-K (self-contained; not limited to scripts/sae_distraction_features.py's saved top).
    acts = np.load(st / "activations.npy", mmap_mode="r")
    with torch.no_grad():
        zd = sae(torch.tensor(acts[distracted, 0, args.layer].astype(np.float32), device=args.device))["z"]
        zr = sae(torch.tensor(acts[vpass, 0, args.layer].astype(np.float32), device=args.device))["z"]
    fr_d = (zd > 0).float().mean(0); fr_r = (zr > 0).float().mean(0)
    alive = ((zd > 0).any(0) | (zr > 0).any(0))
    dfr = torch.where(alive, fr_d - fr_r, torch.full_like(fr_d, -1e9))
    ranked = torch.argsort(dfr, descending=True)            # distraction features, most-distraction-first
    alive_idx = torch.where(alive)[0].cpu().numpy()
    print(f"alive features={len(alive_idx)}; top Δfr={dfr[ranked[0]]:.3f} (feat {int(ranked[0])})")

    model, processor = load_model(args.model, args.device); model.eval()
    n_layers = len(model.model.language_model.layers)
    letter_ids = get_letter_token_ids(processor, args.device, n_letters=6)

    def run(items, feats):
        rec = 0
        for i in tqdm(items, desc=f"K={len(feats)}", leave=False):
            pred = predict(model, processor, rowmap[idx[i]["candidate_id"]], args.layer, sae, feats,
                           letter_ids, args.device, n_layers)
            rec += int(pred == ci[i])
        return rec / max(len(items), 1)

    print(f"\n=== causal-ablation feature-count sweep @ L{args.layer} ({args.model}) ===")
    print(f"{'K':>5}{'recov_real':>12}{'recov_null':>12}{'Δ':>9}{'Vpass_keep':>12}")
    results = []
    for K in [int(x) for x in args.n_feats_list.split(",")]:
        distr_feats = ranked[:K]
        null_feats = torch.tensor(rng.choice(alive_idx, size=min(K, len(alive_idx)), replace=False),
                                  device=args.device)
        rr = run(distracted, distr_feats); rn = run(distracted, null_feats)
        vp = run(vpass, distr_feats)
        results.append({"K": K, "recovery_real": rr, "recovery_null": rn, "vpass_preserved": vp})
        print(f"{K:>5}{rr:>12.3f}{rn:>12.3f}{rr-rn:>+9.3f}{vp:>12.3f}")
    out = dr / "diagnostics" / f"causal_ablation_{args.model}_L{args.layer}{tag}_sweep.json"
    out.write_text(json.dumps({"model": args.model, "layer": args.layer,
        "n_distracted": len(distracted), "n_alive": int(len(alive_idx)), "sweep": results}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
