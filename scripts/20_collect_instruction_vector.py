"""Step 20: build an instruction-contrastive steering vector.

On D_M train_pass, inject an explicit modality directive into the prompt and
collect the last-token residual at the target layers, then take the average
activation difference between the "trust the image" and "trust the text"
conditions. The resulting vector encodes the *instruction* to rely on one
modality over the other; it is applied downstream exactly like v_V_causal
(additive, bcast+normmatch) via scripts/19 --method additive --steering-vectors.

We run BOTH directives on EVERY item, which lets us emit two vectors:

  crossitem (the requested baseline): mean over V-grounded items of h(+IMG)
      minus mean over T-grounded items of h(+TXT). Conflates instruction with
      question-type, like the discriminative contrastive direction.

  peritem (better-motivated control): mean over ALL items of [h(+IMG) - h(+TXT)].
      Same item, same question, only the directive differs -> isolates the
      instruction direction, not reducible to question-encoding (cf. the
      counterfactual h_V - h_VT construction).

Both are saved with keys v_V_L{L} so they drop straight into the additive sweep.
Sign convention: +alpha pushes toward vision (IMG pole minus TXT pole).

    CUDA_VISIBLE_DEVICES=0 python scripts/20_collect_instruction_vector.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from sae_steering.canonical_eval import load_canonical_pool
from sae_steering.models import load_qwen_vl
from sae_steering.steering import CANONICAL_CF_SUBDIR, build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--layers", default="13,20,23,28,31")
    ap.add_argument("--out-dir", default="data/steering/canonical")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dir-img", default="The answer is in the image.")
    ap.add_argument("--dir-txt", default="The answer is in the text.")
    ap.add_argument("--cf-subdir", default=CANONICAL_CF_SUBDIR)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    dm_dir = Path(cfg["paths"]["dm_dir"])
    dist = Path("data/dm/multidomain_v1/distraction_pool.jsonl")
    layers = [int(x) for x in args.layers.split(",")]

    pool = load_canonical_pool(data_root, dm_dir, dist, pool="train_pass",
                               cf_subdir=args.cf_subdir)
    print(f"train_pass pool: {len(pool)} rows")

    model, processor = load_qwen_vl(device=args.device)
    d_model = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else 2048

    # capture last-token residual at each target layer in a single forward
    cap: dict[int, torch.Tensor] = {}
    state = {"last_idx": None}

    def mk(L):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[L] = h[:, state["last_idx"], :].detach().float().cpu()
        return hook

    handles = [model.model.language_model.layers[L].register_forward_hook(mk(L)) for L in layers]

    @torch.inference_mode()
    def acts(row, directive):
        r = dict(row); r["directive"] = directive
        msgs = build_messages(r)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inputs = processor(text=[text], images=ii, videos=vi, padding=True,
                           return_tensors="pt").to(model.device)
        state["last_idx"] = int(inputs.attention_mask.sum(dim=1).item() - 1)
        model(**inputs)
        return {L: cap[L][0].clone() for L in layers}

    sum_img_V = {L: torch.zeros(d_model) for L in layers}
    sum_txt_T = {L: torch.zeros(d_model) for L in layers}
    sum_img_all = {L: torch.zeros(d_model) for L in layers}
    sum_txt_all = {L: torch.zeros(d_model) for L in layers}
    n_V = n_T = n_all = 0

    for row in tqdm(pool, desc="instruction acts"):
        a_img = acts(row, args.dir_img)
        a_txt = acts(row, args.dir_txt)
        n_all += 1
        for L in layers:
            sum_img_all[L] += a_img[L]
            sum_txt_all[L] += a_txt[L]
        if row["label"] == "vision":
            n_V += 1
            for L in layers:
                sum_img_V[L] += a_img[L]
        else:
            n_T += 1
            for L in layers:
                sum_txt_T[L] += a_txt[L]

    for h in handles:
        h.remove()

    def unit(v):
        v = v.numpy().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    cross, per = {}, {}
    for L in layers:
        cross[f"v_V_L{L}"] = unit(sum_img_V[L] / max(n_V, 1) - sum_txt_T[L] / max(n_T, 1))
        per[f"v_V_L{L}"] = unit((sum_img_all[L] - sum_txt_all[L]) / max(n_all, 1))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "v_instruction_crossitem_vectors.npz", **cross)
    np.savez(out / "v_instruction_peritem_vectors.npz", **per)
    print(f"n_V={n_V} n_T={n_T} n_all={n_all}; saved crossitem + peritem npz to {out}")
    # quick sanity: cosine(crossitem, peritem) per layer
    for L in layers:
        c, p = cross[f"v_V_L{L}"], per[f"v_V_L{L}"]
        print(f"  L{L}: cos(cross, per) = {float(np.dot(c, p)):.3f}")


if __name__ == "__main__":
    main()
