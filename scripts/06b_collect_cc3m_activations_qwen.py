"""Step 6b: collect Qwen2.5-VL hidden states on CC3M (multi-layer, natural format).

For each (image, caption) pair: build a minimal user-turn chat with just the
image and the caption — no synthetic MCQ, no question, no instruction tail.
Forward, hook the chosen layers, extract the last-non-pad-input-token state
per layer in a single pass.

Resume-safe: writes one memmap per layer and a shared index.jsonl. Existing
rows are kept across re-runs by checking the index file. Each input row maps
1-1 to an activation row at the matching index.

Outputs (default --variant cc3m_1M):
  data/activations/qwen/cc3m_1M/layer<L>.npy   memmap, shape (N, d_model), float16
  data/activations/qwen/cc3m_1M/index.jsonl    one row: {"key", "image_path", "caption"}
  data/activations/qwen/cc3m_1M/meta.json      {"dataset","n_samples","format","layers",...}

CLI:
  --layers 7 11 20    decoder layers to hook in one pass
  --variant cc3m_1M   subdirectory name under data/activations/qwen/
  --target N          number of pairs to collect (defaults to all in cc3m index)
  --batch-size 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from moground.models import load_qwen_vl
from moground.activations import ActivationExtractor


def build_messages(image_path: str, caption: str,
                   max_pixels: int | None = None) -> list[dict]:
    img = {"type": "image", "image": image_path}
    if max_pixels is not None:
        img["max_pixels"] = max_pixels
    return [{
        "role": "user",
        "content": [img, {"type": "text", "text": caption}],
    }]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 11, 20])
    ap.add_argument("--variant", default="cc3m_1M",
                    help="output subdir under data/activations/qwen/")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=1024*1024,
                    help="cap each image's pixel count (Qwen processor auto-downscales). "
                         "Default ~1M. Drop to 512*512 for safer batching on high-res sources.")
    ap.add_argument("--target", type=int, default=0, help="0 = all in cc3m index")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cc3m_dir = Path(cfg["paths"]["cc3m_dir"])
    out_dir = Path(cfg["paths"]["data_root"]) / "activations" / "qwen" / args.variant
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

    model, processor = load_qwen_vl(device=args.device)
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
    pbar = tqdm(total=n_total, initial=n_done, desc=f"cc3m {args.variant}")
    idx_f = out_index_path.open("a")
    try:
        i = n_done
        while i < n_total:
            batch = pairs[i:i + args.batch_size]
            messages_batch = [build_messages(p["image_path"], p["caption"], max_pixels=args.max_pixels) for p in batch]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                     for m in messages_batch]
            try:
                image_inputs, video_inputs = process_vision_info(messages_batch)
                inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)
                with extractor, torch.inference_mode():
                    _ = model(**inputs)
                    last = extractor.last_token_states(inputs.attention_mask)  # [B, n_layers, D]
                last = last.to(torch.float16).cpu().numpy()
            except Exception as e:
                pbar.write(f"batch error at i={i}: {e}; trying singles")
                last = np.zeros((len(batch), len(layers), d_model), dtype=np.float16)
                for bi, p in enumerate(batch):
                    try:
                        msgs = [build_messages(p["image_path"], p["caption"], max_pixels=args.max_pixels)]
                        t = [processor.apply_chat_template(msgs[0], tokenize=False, add_generation_prompt=False)]
                        ii, vi = process_vision_info(msgs)
                        inp = processor(text=t, images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
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
            "base_model": cfg["model"]["name"],
            "d_model": d_model,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"meta -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
