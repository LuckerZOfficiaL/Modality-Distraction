r"""Assemble the SAE checkpoint release: data/sae_release/ -> models/<ns>/moground-saes.

Ships the seven checkpoints the paper's table reports (four Qwen2.5-VL-3B layers, three
LLaVA-NeXT-8B layers), each with its training config, the metrics the table quotes, and a loader.

    python scripts/137_build_sae_release.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/sae_release"

# (backbone dir, layer, display name, d_model) -- exactly the rows of tab:sae_details
CKPTS = [
    ("qwen", 13, "Qwen2.5-VL-3B", 2048), ("qwen", 20, "Qwen2.5-VL-3B", 2048),
    ("qwen", 28, "Qwen2.5-VL-3B", 2048), ("qwen", 31, "Qwen2.5-VL-3B", 2048),
    ("llavanext", 12, "LLaVA-NeXT-8B", 4096), ("llavanext", 18, "LLaVA-NeXT-8B", 4096),
    ("llavanext", 25, "LLaVA-NeXT-8B", 4096),
]
PAPER_LAYER = {"qwen": 28, "llavanext": 18}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows, total = [], 0
    for tag, layer, disp, dm in CKPTS:
        src = ROOT / f"data/sae/{tag}/diversified_v1/layer{layer}"
        dst = OUT / f"{tag}/layer{layer}"
        dst.mkdir(parents=True)
        for f in ("sae.pt", "meta.json", "metrics.json"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        met = json.loads((src / "metrics.json").read_text()) if (src / "metrics.json").exists() else {}
        n = (dst / "sae.pt").stat().st_size
        total += n
        rows.append(dict(backbone=disp, tag=tag, layer=layer, d_model=dm, d_sae=8 * dm,
                         path=f"{tag}/layer{layer}/sae.pt", bytes=n,
                         sha256=hashlib.sha256((dst / "sae.pt").read_bytes()).hexdigest(),
                         fvu=(round(met["final_fvu"], 3) if "final_fvu" in met else None),
                         alive_frac=(round(met["final_alive_frac_eval_batch"], 2)
                                     if "final_alive_frac_eval_batch" in met else None),
                         k=met.get("k"), n_train=met.get("n_train"),
                         used_in_paper=(layer == PAPER_LAYER[tag])))
        print(f"  {tag}/layer{layer}  {n / 1e6:7.0f} MB")
    (OUT / "MANIFEST.json").write_text(json.dumps(rows, indent=2))
    (OUT / "load_sae.py").write_text(LOADER)
    (OUT / "README.md").write_text(card(rows))
    print(f"\n{len(rows)} checkpoints, {total / 1e9:.1f} GB -> {OUT}")


LOADER = r'''"""Load a released MoGround SAE checkpoint.

    from load_sae import load
    sae = load("qwen/layer28/sae.pt")
    out = sae(residual)        # {"x_hat", "z", "indices", "pre"}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """z = ReLU(TopK(W_enc (x - b_dec) + b_enc)),  x_hat = W_dec z + b_dec."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))
        return {"x_hat": z @ self.W_dec + self.b_dec, "z": z, "indices": idx, "pre": pre}


def load(path, device="cpu"):
    sd = torch.load(path, map_location=device)
    sd = sd.get("state_dict", sd)
    d_model, d_sae = sd["W_enc"].shape
    sae = TopKSAE(d_model, d_sae, k=int(torch.load(path, map_location="cpu").get("k", 32)))
    sae.load_state_dict(sd, strict=True)      # strict: a renamed key must fail loudly, not silently
    return sae.to(device).eval().requires_grad_(False)
'''


def card(rows):
    tbl = "\n".join(
        f"| {r['backbone']} | {r['layer']} | {r['d_model']} | {r['d_sae']} | "
        f"{r['fvu'] if r['fvu'] is not None else '--'} | "
        f"{r['alive_frac'] if r['alive_frac'] is not None else '--'} | "
        f"{'yes' if r['used_in_paper'] else ''} | `{r['path']}` |" for r in rows)
    return f"""---
license: cc-by-nc-4.0
tags:
  - sparse-autoencoder
  - interpretability
  - vision-language
---

# MoGround vision-language SAEs

Top-$K$ sparse autoencoders ($K{{=}}32$) trained on the last-token residual of a frozen VLM as it
processes an image-caption pair, which is the position its answer is read from. Trained
unsupervised on reconstruction alone, over 1,242,328 image-caption pairs drawn from CC3M (natural
photos), PlotQA (charts), WikiArt (paintings) and OpenI (radiology).

The encoder and decoder are separate matrices rather than transposes of each other, both biases are
learned, and the decoder rows are kept at unit norm. Width is 8x the model dimension. Training used
batch size 4096, learning rate 3e-4 and no weight decay.

| backbone | layer | d_model | d_sae | FVU | alive frac. | used in the paper | file |
|---|---|---|---|---|---|---|---|
{tbl}

FVU is the share of variance the reconstruction fails to explain, so lower is better. Alive
fraction is the share of features that fire at least once on an evaluation batch. Both are measured
on held-out activations from the pretraining distribution.

**Expect a much higher FVU off that distribution.** On MoGround's multiple-choice prompts the
layer-28 checkpoint reconstructs to FVU 0.78 about its own centre, and the centre itself has moved:
the mean MoGround residual sits 107 away from the decoder bias, against a mean activation norm of
147. Measured against MoGround's own mean the apparent FVU is above 1. That is distribution shift
between natural-caption pretraining and answer-time MCQ prompts, not a damaged checkpoint.

## Use

```python
from load_sae import load
sae = load("qwen/layer28/sae.pt")
out = sae(residual)          # dict with x_hat, z, indices, pre
```

`MANIFEST.json` carries a sha256 and the size of every checkpoint.

## What these are for, and what they are not

They were trained to ask whether modality distraction is readable, and removable, in a sparse
basis. It is readable: a probe on the SAE code separates distracted from robust items above a
label-permutation null. It is not removable this way: ablating the most distraction-predictive
features does no better than ablating the same number of random ones. Treat these as an
interpretability tool, not as a mitigation.
"""


if __name__ == "__main__":
    main()
