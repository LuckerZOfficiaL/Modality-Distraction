"""Step 8: train a Top-K SAE on CC3M layer-20 activations.

Reads data/activations/cc3m/layer20.npy (N, d_model) float16 and trains a
Top-K SAE with width = width_mult * d_model and k from configs/pilot.yaml.

Outputs:
  data/sae/layer20/sae.pt          weights + config
  data/sae/layer20/metrics.json    final summary (loss, alive frac, mean L0)
  data/sae/layer20/log.csv         per-step log

Resume-safe: if sae.pt exists and matches the current config, the script
loads it and continues training from the saved step count.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sae_steering.sae import (
    SAEConfig,
    TopKSAE,
    reconstruction_loss,
    remove_decoder_parallel_grad,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--model", default="qwen", help="backbone subdir under data/activations/ and data/sae/")
    ap.add_argument("--variant", default="cc3m_1M",
                    help="activations subdir under data/activations/<model>/ and output subdir under data/sae/<model>/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    act_path = data_root / "activations" / args.model / args.variant / f"layer{args.layer:02d}.npy"
    out_dir = data_root / "sae" / args.model / args.variant / f"layer{args.layer:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_meta_path = act_path.parent / "meta.json"
    src_meta = json.loads(src_meta_path.read_text()) if src_meta_path.exists() else {}

    print(f"loading activations from {act_path}")
    acts_mm = np.load(act_path, mmap_mode="r")  # (N, d_model) float16
    n_total, d_model = acts_mm.shape
    d_sae = int(cfg["sae"]["width_mult"]) * d_model
    k = int(cfg["sae"]["k"])
    print(f"N={n_total}  d_model={d_model}  d_sae={d_sae}  k={k}")

    device = torch.device(args.device)
    sae_cfg = SAEConfig(d_model=d_model, d_sae=d_sae, k=k)
    sae = TopKSAE(sae_cfg).to(device).to(torch.float32)

    # init b_dec from a sample of activations (mean)
    sample_idx = np.random.default_rng(cfg["seed"]).choice(n_total, size=min(50_000, n_total), replace=False)
    sample = torch.from_numpy(np.array(acts_mm[sample_idx])).to(device).float()
    sae.init_b_dec(sample.mean(dim=0))
    print(f"b_dec initialized from {sample.shape[0]} samples")

    ckpt_path = out_dir / "sae.pt"
    log_path = out_dir / "log.csv"
    start_step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        if ck.get("d_sae") == d_sae and ck.get("k") == k and ck.get("d_model") == d_model:
            sae.load_state_dict(ck["state_dict"])
            start_step = int(ck.get("step", 0))
            print(f"resumed from step {start_step}")

    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))

    rng = np.random.default_rng(cfg["seed"] + 1)
    steps_per_epoch = max(1, n_total // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    log_f = log_path.open("a" if start_step > 0 else "w", newline="")
    log_w = csv.writer(log_f)
    if start_step == 0:
        log_w.writerow(["step", "loss", "alive_frac", "l0", "fvu"])

    # alive tracking: a latent is "alive" if it fired in the last 1k steps
    fired_recent = torch.zeros(d_sae, dtype=torch.bool, device=device)
    fired_window: list[torch.Tensor] = []

    pbar = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps)
    sae.train()
    for step in pbar:
        idx = rng.integers(0, n_total, size=args.batch_size)
        idx.sort()
        x = torch.from_numpy(np.array(acts_mm[idx])).to(device).float()

        out = sae(x)
        loss = reconstruction_loss(x, out["x_hat"])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        remove_decoder_parallel_grad(sae)
        opt.step()
        sae.normalize_decoder()

        # rolling alive tracker
        with torch.no_grad():
            fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
            fired[out["indices"].reshape(-1)] = True
            fired_window.append(fired)
            if len(fired_window) > 1000:
                fired_window.pop(0)

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                fvu = (loss / x.var(unbiased=False)).item()
                alive = torch.stack(fired_window).any(dim=0).float().mean().item()
                l0 = float(k)
                pbar.set_postfix(loss=loss.item(), alive=alive, fvu=fvu)
                log_w.writerow([step + 1, loss.item(), alive, l0, fvu])
                log_f.flush()

        if (step + 1) % args.ckpt_every == 0:
            torch.save({
                "state_dict": sae.state_dict(),
                "step": step + 1,
                "d_model": d_model,
                "d_sae": d_sae,
                "k": k,
                "layer": args.layer,
            }, ckpt_path)

    torch.save({
        "state_dict": sae.state_dict(),
        "step": total_steps,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": k,
        "layer": args.layer,
    }, ckpt_path)
    log_f.close()

    # final eval
    sae.eval()
    with torch.no_grad():
        eval_idx = rng.integers(0, n_total, size=min(20_000, n_total))
        eval_idx.sort()
        x = torch.from_numpy(np.array(acts_mm[eval_idx])).to(device).float()
        out = sae(x)
        final_loss = reconstruction_loss(x, out["x_hat"]).item()
        fvu = final_loss / x.var(unbiased=False).item()
        fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
        fired[out["indices"].reshape(-1)] = True
        alive = fired.float().mean().item()

    summary = {
        "layer": args.layer,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": k,
        "n_train": n_total,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": final_loss,
        "final_fvu": fvu,
        "final_alive_frac_eval_batch": alive,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    meta = {
        "base_model": cfg["model"]["name"],
        "layer": args.layer,
        "variant": args.variant,
        "pretraining": {
            "dataset": src_meta.get("dataset", "cc3m"),
            "n_samples": src_meta.get("n_samples", n_total),
            "format": src_meta.get("format", "natural_caption_chat"),
        },
        "sae": {"sparsity": "topk", "k": k, "width_mult": int(cfg["sae"]["width_mult"]),
                "d_model": d_model, "d_sae": d_sae},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                     "weight_decay": args.weight_decay},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
