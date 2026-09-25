"""Step 6b (LLaVA-NeXT-LLaMA3-8B): collect last-input-token hidden states on
CC3M across selected LM layers, in a single pass, using natural-caption format.

For each (image, caption) pair: build a minimal user-turn chat with just the
image and the caption — no synthetic MCQ, no question, no instruction tail.

Resume-safe: writes one memmap per layer and a shared index.jsonl. Existing
rows are kept across re-runs by checking the index file. Each input row maps
1-1 to an activation row at the matching index.

Outputs (default --variant cc3m_1M):
  data/activations/llavanext/<variant>/layer<L>.npy
  data/activations/llavanext/<variant>/index.jsonl
  data/activations/llavanext/<variant>/meta.json

CLI:
  --layers 13 20 28
  --variant cc3m_1M
  --target N         (0 = all in cc3m index)
  --batch-size 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from sae_steering.activations import ActivationExtractor
from sae_steering.models_llavanext import load_llavanext


def build_prompt(image_path: str, caption: str, processor) -> tuple[str, Image.Image]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": caption},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    img = Image.open(image_path).convert("RGB")
    return prompt, img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, nargs="+", default=[13, 20, 28])
    ap.add_argument("--variant", default="cc3m_1M",
                    help="output subdir under data/activations/llavanext/")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--target", type=int, default=0, help="0 = all in cc3m index")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cc3m_dir = Path(cfg["paths"]["cc3m_dir"])
    out_dir = Path(cfg["paths"]["data_root"]) / "activations" / "llavanext" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in (cc3m_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    if args.target > 0:
        pairs = pairs[:args.target]
    n_total = len(pairs)
    print(f"cc3m pairs: {n_total}  layers={args.layers}  variant={args.variant}")

    layers = sorted(set(int(L) for L in args.layers))
    out_index_path = out_dir / "index.jsonl"
    act_paths = {L: out_dir / f"layer{L:02d}.npy" for L in layers}

    n_done = 0
    all_acts_ok = all(p.exists() and p.stat().st_size > 0 for p in act_paths.values())
    if not all_acts_ok and out_index_path.exists():
        print("WARN: some activation files missing/empty; truncating index and restarting")
        out_index_path.unlink()
        for p in act_paths.values():
            if p.exists(): p.unlink()
    if out_index_path.exists() and all_acts_ok:
        existing = [json.loads(l) for l in out_index_path.read_text().splitlines() if l.strip()]
        n_done = len(existing)
        if any(existing[i]["key"] != pairs[i]["key"] for i in range(min(len(existing), len(pairs)))):
            raise RuntimeError("existing index.jsonl does not align with cc3m index prefix")
        print(f"resuming: {n_done}/{n_total} already collected")

    if n_done >= n_total:
        print("nothing to do.")
        return

    model, processor = load_llavanext(device=args.device)
    d_model = model.config.text_config.hidden_size
    print(f"hidden_size={d_model}")

    arrs = {}
    for L, p in act_paths.items():
        if p.exists() and p.stat().st_size > 0:
            old = np.load(p, mmap_mode="r")
            if old.shape[0] >= n_total:
                arrs[L] = np.array(old)
            else:
                a = np.zeros((n_total, d_model), dtype=np.float16)
                a[:old.shape[0]] = old
                arrs[L] = a
        else:
            arrs[L] = np.zeros((n_total, d_model), dtype=np.float16)

    extractor = ActivationExtractor(model, layers)
    pbar = tqdm(total=n_total, initial=n_done, desc=f"cc3m llavanext {args.variant}")
    idx_f = out_index_path.open("a")
    try:
        i = n_done
        while i < n_total:
            batch = pairs[i:i + args.batch_size]
            try:
                prompts, images = zip(*[build_prompt(p["image_path"], p["caption"], processor) for p in batch])
                inputs = processor(text=list(prompts), images=list(images),
                                   padding=True, return_tensors="pt").to(model.device)
                with extractor, torch.inference_mode():
                    _ = model(**inputs)
                    last = extractor.last_token_states(inputs.attention_mask)
                last = last.to(torch.float16).cpu().numpy()
            except Exception as e:
                pbar.write(f"batch error at i={i}: {e}; trying singles")
                last = np.zeros((len(batch), len(layers), d_model), dtype=np.float16)
                for bi, p in enumerate(batch):
                    try:
                        pr, im = build_prompt(p["image_path"], p["caption"], processor)
                        inp = processor(text=[pr], images=[im], padding=True, return_tensors="pt").to(model.device)
                        with extractor, torch.inference_mode():
                            _ = model(**inp)
                            l1 = extractor.last_token_states(inp.attention_mask)
                        last[bi] = l1[0].to(torch.float16).cpu().numpy()
                    except Exception as ee:
                        pbar.write(f"  single fail key={p['key']}: {ee}; zeroing")
            for li, L in enumerate(layers):
                arrs[L][i:i + len(batch)] = last[:, li, :]
            for p in batch:
                idx_f.write(json.dumps({
                    "key": p["key"], "image_path": p["image_path"], "caption": p["caption"],
                }) + "\n")
            idx_f.flush()
            i += len(batch)
            pbar.update(len(batch))
            if i % (args.batch_size * 250) < args.batch_size:
                for L, p in act_paths.items():
                    np.save(p, arrs[L])
    finally:
        idx_f.close()
        pbar.close()
        for L, p in act_paths.items():
            np.save(p, arrs[L])
            print(f"saved -> {p}  shape=({n_total}, {d_model})")
        meta = {
            "dataset": "cc3m",
            "n_samples": n_total,
            "format": "natural_caption_chat",
            "layers": layers,
            "base_model": "llava-hf/llama3-llava-next-8b-hf",
            "d_model": d_model,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"meta -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
