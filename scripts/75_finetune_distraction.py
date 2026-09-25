"""Distraction-aware LoRA finetuning (Qwen2.5-VL-3B / LLaVA-NeXT).

Goal: reduce v-distraction (an irrelevant caption flipping a correct vision answer) WITHOUT
regressing text-use, by finetuning on a routing mix:
  - vision-grounded items: answer from the image;
  - text-grounded items:   answer from the caption (this is the preservation signal that stops
    the model from simply learning to ignore text).

Methods:
  --method dropout  (Phase 0 baseline): on vision items, drop the caption with prob --p-drop.
  --method hardneg  (Phase 1):          on vision items, replace the caption with an adversarial one
                                        (field `adv_caption`; built by a separate Sonnet pass).
  --method grounded (E1, localization): show each item ONLY its grounded modality (vision -> image
                                        alone, text -> caption alone). Raises single-modality
                                        accuracy without ever training the V+T condition, so
                                        distraction stays an untrained readout. Qwen family only
                                        (the llava/hf tokenizers always attach the image).

Attention-only LoRA on the language model; loss = CE on the answer letter (prompt masked).
Trains on the D_M TRAIN split only (eval on test is leakage-free). Saves the adapter to --out.

    CUDA_VISIBLE_DEVICES=0 python scripts/75_finetune_distraction.py --method dropout --out runs/ft_dropout_qwen
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
LETTERS = ["A", "B", "C", "D"]
MAXPX = 1024 * 1024

# qwen-family backbones share the c6 message builder + process_vision_info; only the checkpoint/class
# differ. Enables the within-family grounding sweep (2B / 3B / 7B) for the correctability phase diagram.
QWEN_SPECS = {
    "qwen":           ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl"),
    "qwen2.5-vl-7b":  ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
    "qwen2-vl-2b":    ("Qwen/Qwen2-VL-2B-Instruct",   "qwen2_vl"),
    "qwen3-vl-30b":   ("Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen3_vl_moe"),  # MoE: 30B total / 3B active
}


def load_qwen_family(model_key, device):
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration)
    model_id, cls_name = QWEN_SPECS[model_key]
    cls = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen2_vl": Qwen2VLForConditionalGeneration,
           "qwen3_vl_moe": Qwen3VLMoeForConditionalGeneration}[cls_name]
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                low_cpu_mem_usage=True).to(device)
    return model, AutoProcessor.from_pretrained(model_id)


# generic HF-native VLMs (AutoModelForImageTextToText): the remaining leaderboard backbones.
# Train-time messages are built byte-identical to the scripts/61 make_hf eval format.
HF_SPECS = {
    "internvl3-8b":       "OpenGVLab/InternVL3-8B-hf",
    "internvl3-14b":      "OpenGVLab/InternVL3-14B-hf",
    "llava-onevision-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-1.5-7b":       "llava-hf/llava-1.5-7b-hf",
    "mistral-small-3.1-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",   # 3rd large model (4th family)
}


def load_hf_family(model_key, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_id = HF_SPECS[model_key]
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device)
    return model, AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def route(row, method, p_drop, mix_truthful, rng, label=None):
    """Decide (kind, caption_text) for one item. Backbone-agnostic routing logic.

    kind is "V" (image only, caption dropped) or "VT" (image + caption). caption_text is the string to
    show when kind=="VT" (original truthful caption, or the adversarial one), else None.

    mix_truthful (hardneg only): on a vision item that has an adv_caption, with this prob show the
    TRUTHFUL original caption (consistent with the image) instead of the adversarial one. Both target
    the correct answer, so the model sees the same vision-question sometimes with a lying caption (must
    override) and sometimes with an agreeing caption (fine) -> conditional routing on image-consistency
    rather than a blanket "distrust the caption on vision questions".
    """
    cap = row.get("caption_for_filter")
    if method == "sft":
        return "VT", cap        # vanilla SFT: every item as-is, true caption, no routing at all
    # `label` overrides the item's certificate (certification ablation); None = use the certificate
    lab = label if label is not None else row["label"]
    if method == "grounded":
        # E1 (grounding_strength_localization): show each item ONLY its grounded modality, so the
        # objective raises single-modality accuracy and NEVER touches the V+T condition that the
        # distraction metric reads out. Filter the train file to one label to get a one-sided vector.
        return ("V", None) if lab == "vision" else ("T", cap)
    if lab == "vision":
        if method == "dropout" and rng.random() < p_drop:
            return "V", None                                        # drop caption
        if method == "hardneg" and row.get("adv_caption"):
            if rng.random() < mix_truthful:
                return "VT", cap                                    # truthful, image-consistent caption
            return "VT", row["adv_caption"]                         # adversarial
        return "VT", cap
    return "VT", cap                                                # text items always keep caption


def make_tokenizer(family, proc, device, llmod=None, model_id=""):
    """Return (tok, replay_tok) for the given backbone family.

    tok(kind, caption_text, row, target) -> (inputs, labels) for a D_M MCQ item.
    replay_tok(cc3m_item) -> (inputs, labels) for a generic image-captioning item, used only by the
    KL-to-base replay anchor (D): it preserves the base model's general image-conditioned language
    behavior on out-of-D_M data, targeting the broad LM drift that hurts caption-use / text-only on
    weak backbones.

    Qwen path is byte-identical to the original (c6.build_messages + process_vision_info). LLaVA path
    uses the family's _build_messages + the HF chat template, appending the answer to the prompt so the
    prompt boundary (for label masking) does not depend on the template rendering an assistant turn.
    """
    if family == "qwen":
        from qwen_vl_utils import process_vision_info
        _c = importlib.util.spec_from_file_location(
            "_c6", ROOT / "06c_collect_dm_counterfactual_activations.py")
        c6 = importlib.util.module_from_spec(_c); _c.loader.exec_module(c6)  # type: ignore

        def _pack(user, assistant_text):
            full = user + [{"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]
            text_full = proc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            text_prompt = proc.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(full)
            inp = proc(text=[text_full], images=imgs, videos=vids, return_tensors="pt").to(device)
            plen = proc(text=[text_prompt], images=imgs, videos=vids,
                        return_tensors="pt")["input_ids"].shape[1]
            labels = inp["input_ids"].clone(); labels[:, :plen] = -100
            return inp, labels

        def tok(kind, cap, row, target):
            row2 = row if kind == "V" else {**row, "caption_for_filter": cap}
            return _pack(c6.build_messages(row2, kind, MAXPX), target)

        def replay_tok(item):
            user = [{"role": "user", "content": [
                {"type": "image", "image": item["image_path"], "max_pixels": MAXPX},
                {"type": "text", "text": "Describe the image."}]}]
            return _pack(user, item["caption"])
        return tok, replay_tok

    if family == "hf":  # generic HF VLM: mirror the scripts/61 make_hf eval prompt exactly
        def _build(row, cap):
            content = []
            if cap is not None:
                content.append({"type": "text", "text": f"Caption: {cap}"})
            content.append({"type": "image"})
            opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
            content.append({"type": "text",
                            "text": f"Question: {row['question']}\n{opts}\nReply with one letter (A-D)."})
            return [{"role": "user", "content": content}]

        # Must match the eval-side tokenization exactly (scripts/61 make_hf): a train/eval BOS
        # mismatch is worse than a consistent double-BOS.
        from moground.steering import special_token_kwargs
        stk = special_token_kwargs(model_id)   # {} for every pre-existing backbone

        def _pack_hf(user, image, assistant_text):
            prompt = proc.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
            inp = proc(text=prompt + assistant_text, images=image, return_tensors="pt", **stk).to(device)
            plen = proc(text=prompt, images=image, return_tensors="pt", **stk)["input_ids"].shape[1]
            labels = inp["input_ids"].clone(); labels[:, :plen] = -100
            return inp, labels

        def tok(kind, cap, row, target):
            img = Image.open(row["image_path"]).convert("RGB")   # image always present (V and VT)
            return _pack_hf(_build(row, cap if kind == "VT" else None), img, target)

        def replay_tok(item):
            user = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
            return _pack_hf(user, Image.open(item["image_path"]).convert("RGB"), item["caption"])
        return tok, replay_tok

    def _pack_llava(user, image, assistant_text):
        prompt = proc.apply_chat_template(user, add_generation_prompt=True)
        inp = proc(text=prompt + assistant_text, images=image, return_tensors="pt").to(device)
        plen = proc(text=prompt, images=image, return_tensors="pt")["input_ids"].shape[1]
        labels = inp["input_ids"].clone(); labels[:, :plen] = -100
        return inp, labels

    def tok(kind, cap, row, target):                                # llava family
        ip = row["image_path"]                                      # image always present (V and VT)
        user = llmod._build_messages(row["question"], row["options"], ip, cap if kind == "VT" else None)
        return _pack_llava(user, Image.open(ip).convert("RGB"), target)

    def replay_tok(item):
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
        return _pack_llava(user, Image.open(item["image_path"]).convert("RGB"), item["caption"])
    return tok, replay_tok


def resolve_targets(model, family, which, layer_min=0):
    leaf = {"attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
            "mlp": {"gate_proj", "up_proj", "down_proj"},
            "all": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}}[which]
    if layer_min:  # explicit per-layer module names, language model only, layers >= layer_min
        import re as _re
        names = []
        for n, _ in model.named_modules():
            if n.split(".")[-1] not in leaf:
                continue
            m = _re.search(r"\.layers\.(\d+)\.", n + ".")
            if m and int(m.group(1)) >= layer_min and ("language_model" in n or family == "qwen"):
                names.append(n)
        if not names:
            raise SystemExit(f"no LoRA targets at layers >= {layer_min}")
        return names
    if family == "qwen":
        return list(leaf)                                          # matches Qwen LM only (visual uses qkv)
    # scope to the language model: the CLIP vision tower shares q/k/v_proj names and must not be adapted
    names = [n for n, _ in model.named_modules()
             if n.split(".")[-1] in leaf and "language_model" in n]
    if not names:
        raise SystemExit(f"no LoRA targets found under language_model for {family}/{which}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--model", default="qwen",
                    choices=["qwen", "qwen2.5-vl-7b", "qwen2-vl-2b", "llavanext", 
                             "internvl3-8b", "llava-onevision-7b", "llava-1.5-7b",
                             
                             "internvl3-14b", "mistral-small-3.1-24b", "qwen3-vl-30b"],
                    help="backbone to finetune (qwen-family + LLaVA modules + generic HF VLMs)")
    ap.add_argument("--train", default=None, help="train jsonl override (e.g. train_adv.jsonl for hardneg)")
    ap.add_argument("--method", default="dropout", choices=["dropout", "hardneg", "sft", "grounded"])
    ap.add_argument("--label-mode", default="certified",
                    choices=["certified", "shuffled", "uniform"],
                    help="certification ablation. certified: route by the item's grounding "
                         "certificate (the paper's recipe). shuffled: permute the certificates "
                         "across the train split, holding the marginal V/T mix fixed (permutation "
                         "control). uniform: treat every item as vision-grounded, i.e. what one "
                         "would do without certificates at all.")
    ap.add_argument("--p-drop", type=float, default=0.5)
    ap.add_argument("--mix-truthful", type=float, default=0.0,
                    help="hardneg: prob a vision item shows its truthful caption instead of adv (routing)")
    ap.add_argument("--text-weight", type=int, default=1,
                    help="replicate each text item this many times per epoch (rebalance vs vision)")
    ap.add_argument("--kl-anchor", type=float, default=0.0,
                    help="on text items, add this * KL(base||FT) on the answer token(s) to preserve "
                         "the base model's text-with-image behavior (fights the caption-use regression)")
    ap.add_argument("--replay-kl", type=float, default=0.0,
                    help="(D) weight for a KL(base||FT) anchor on generic CC3M image-captioning replay "
                         "items, interleaved during training to preserve broad multimodal LM behavior")
    ap.add_argument("--replay-every", type=int, default=4, help="insert one replay item every N train steps")
    ap.add_argument("--replay-n", type=int, default=2000, help="CC3M replay pool size to sample from")
    ap.add_argument("--out", required=True, help="adapter output dir, e.g. runs/ft_dropout_qwen")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-target", default="attn", choices=["attn", "mlp", "all"],
                    help="which LM submodules the LoRA adapts")
    ap.add_argument("--layer-min", type=int, default=0,
                    help="train LoRA only on language-model layers with index >= this (0 = all); "
                         "the commit-layer localization test")
    ap.add_argument("--vteacher-kl", type=float, default=0.0,
                    help="(Tier 2) on vision items shown WITH a caption, add this * "
                         "KL(base_V-only || FT_V+T) on the answer tokens: distil the base model's "
                         "image-only answer distribution (the h_V patch that recovers 93.5%% of "
                         "flips at inference) into the weights. Teacher is the frozen base model "
                         "run on the SAME item without the caption.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all train items (smoke-test with e.g. 32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()); dm = Path(cfg["paths"]["dm_dir"])
    train_path = Path(args.train) if args.train else dm / "splits/train.jsonl"
    rows = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    if args.method == "hardneg":
        na = sum(bool(r["label"] == "vision" and r.get("adv_caption")) for r in rows)
        print(f"hardneg: {na} vision items carry adv_caption (rest fall through to plain VT)")
    if args.limit:
        rows = rows[:args.limit]
    rng = random.Random(args.seed)
    nV = sum(r["label"] == "vision" for r in rows)

    # --- certification ablation: fix each item's ROUTING label once, before training ------------
    # The training objective is routed per item by its grounding certificate. These modes vary only
    # that routing; captions, items, objective, KL anchor, seeds and eval are untouched.
    if args.label_mode == "certified":
        route_label = [r["label"] for r in rows]
    elif args.label_mode == "shuffled":
        route_label = [r["label"] for r in rows]
        random.Random(args.seed).shuffle(route_label)   # same V/T marginal, assignment destroyed
    else:                                               # uniform: no certificate available at all
        route_label = ["vision"] * len(rows)
    n_agree = sum(a == b for a, b in zip(route_label, [r["label"] for r in rows]))
    print(f"=== FT {args.method} [labels={args.label_mode}] on {len(rows)} train items "
          f"({nV} vision / {len(rows)-nV} text); routing label agrees with certificate on "
          f"{n_agree}/{len(rows)} ({n_agree/len(rows):.1%}) ===")

    family = ("qwen" if args.model in QWEN_SPECS else
              "hf" if args.model in HF_SPECS else "llava")
    llmod = None
    if family == "qwen":
        model, proc = load_qwen_family(args.model, args.device)
    elif family == "hf":
        model, proc = load_hf_family(args.model, args.device)
    else:
        llmod = importlib.import_module(f"moground.models_{args.model}")
        model, proc = getattr(llmod, f"load_{args.model}")(device=args.device)
    tok, replay_tok = make_tokenizer(family, proc, args.device, llmod,
                                     model_id=HF_SPECS.get(args.model, ""))

    replay_pool = []
    if args.replay_kl > 0:
        import os
        idx = Path(cfg["paths"]["data_root"]) / "pretraining/diversified_v1/index.jsonl"
        cand = [json.loads(l) for l in idx.read_text().splitlines() if l.strip()]
        cand = [r for r in cand if r.get("source") == "cc3m"]
        rng.shuffle(cand)
        for r in cand:
            if os.path.exists(r["image_path"]):
                replay_pool.append(r)
            if len(replay_pool) >= args.replay_n:
                break
        print(f"replay-kl {args.replay_kl}: {len(replay_pool)} CC3M items, 1 every {args.replay_every} steps")
    model.train(); model.config.use_cache = False
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    targets = resolve_targets(model, family, args.lora_target, args.layer_min)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # text-item upweighting: replicate each text index --text-weight times per epoch
    base_order = [ix for ix, r in enumerate(rows)
                  for _ in range(args.text_weight if route_label[ix] == "text" else 1)]
    if args.text_weight != 1:
        print(f"text-weight {args.text_weight}: epoch size {len(rows)} -> {len(base_order)}")

    step = 0
    for ep in range(args.epochs):
        order = list(base_order); rng.shuffle(order)
        running = 0.0
        for i, ix in enumerate(tqdm(order, desc=f"epoch {ep+1}/{args.epochs}")):
            row = rows[ix]
            kind, cap = route(row, args.method, args.p_drop, args.mix_truthful, rng,
                              label=route_label[ix])
            inp, labels = tok(kind, cap, row, LETTERS[int(row["correct_index"])])
            out = model(**inp, labels=labels)
            loss = out.loss
            if args.kl_anchor > 0 and route_label[ix] == "text":
                # anchor the FT answer distribution to the base model (adapter disabled) on the same
                # text-with-image input, so learning to route on vision items doesn't drift the
                # legitimate caption-use behavior. KL(base||FT) over the answer token positions.
                with torch.no_grad(), model.disable_adapter():
                    base_logits = model(**inp).logits
                m = labels[:, 1:] != -100
                if m.any():
                    lp_ft = F.log_softmax(out.logits[:, :-1, :][m], dim=-1)
                    p_base = F.softmax(base_logits[:, :-1, :][m].float(), dim=-1)
                    kl = (p_base * (torch.log(p_base + 1e-9) - lp_ft)).sum(-1).mean()
                    loss = loss + args.kl_anchor * kl
            if args.vteacher_kl > 0 and route_label[ix] == "vision" and kind == "VT":
                # teacher = frozen base model on the same item with NO caption (V-only);
                # student = FT model on the caption-bearing input. Distil V-only -> V+T.
                inp_v, labels_v = tok("V", None, row, LETTERS[int(row["correct_index"])])
                with torch.no_grad(), model.disable_adapter():
                    t_logits = model(**inp_v).logits
                mt = labels_v[:, 1:] != -100
                ms = labels[:, 1:] != -100
                if mt.any() and ms.any() and int(mt.sum()) == int(ms.sum()):
                    lp_s = F.log_softmax(out.logits[:, :-1, :][ms], dim=-1)
                    p_t = F.softmax(t_logits[:, :-1, :][mt].float(), dim=-1)
                    vkl = (p_t * (torch.log(p_t + 1e-9) - lp_s)).sum(-1).mean()
                    loss = loss + args.vteacher_kl * vkl
            loss = loss / args.grad_accum
            loss.backward(); running += loss.item()

            if replay_pool and step % args.replay_every == 0:
                ritem = replay_pool[rng.randrange(len(replay_pool))]
                try:
                    rinp, rlabels = replay_tok(ritem)
                except Exception:                                   # skip unreadable image, keep training
                    rinp = None
                if rinp is not None:
                    rlogits = model(**rinp).logits
                    with torch.no_grad(), model.disable_adapter():
                        rbase = model(**rinp).logits
                    rm = rlabels[:, 1:] != -100
                    if rm.any():
                        lp = F.log_softmax(rlogits[:, :-1, :][rm], dim=-1)
                        pb = F.softmax(rbase[:, :-1, :][rm].float(), dim=-1)
                        rkl = (pb * (torch.log(pb + 1e-9) - lp)).sum(-1).mean()
                        (args.replay_kl * rkl / args.grad_accum).backward()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if step % (args.grad_accum * 25) == 0:
                print(f"  step {step}: loss {running / (args.grad_accum*25):.4f}"); running = 0.0
        opt.step(); opt.zero_grad()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    (out / "ft_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"-> adapter saved to {out}")


if __name__ == "__main__":
    main()
