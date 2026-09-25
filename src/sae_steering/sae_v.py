"""Loader + vanilla SAE module for the PKU-Alignment SAE-V release.

The HF release at PKU-Alignment/SAE-V/SAEV_LLaVA_NeXT-7b_OBELICS/ ships a
standard L1-ReLU SAE (no fork required). The cfg's normalize_activations
field ("expected_average_only_in") has been folded into W_enc/b_enc at
save time, so inference is just plain encode/decode (verified by the
spike: FVU=0.0122 on Mistral-7B L16).

Sparsify Top-K (lmms-lab L24) and vanilla L1-ReLU SAEs both expose
encode(x) and decode(z); this module mirrors the same surface so the
steering hook can be written once.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


SAE_V_REPO = "PKU-Alignment/SAE-V"
SAE_V_LLAVA_NEXT_SUBDIR = "SAEV_LLaVA_NeXT-7b_OBELICS"


class VanillaSAE(nn.Module):
    """L1-ReLU SAE: z = relu(x W_enc + b_enc); x_hat = z W_dec + b_dec.

    Shapes:
      W_enc: (d_in, d_sae)
      b_enc: (d_sae,)
      W_dec: (d_sae, d_in)
      b_dec: (d_in,)

    encode/decode return dense [..., d_sae] / [..., d_in] tensors.
    """

    def __init__(self, W_enc, b_enc, W_dec, b_dec):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.d_in = int(W_enc.shape[0])
        self.d_sae = int(W_enc.shape[1])

    def encode(self, x):
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


def load_sae_v_llava_next(device: str = "cuda", dtype: torch.dtype = torch.float32) -> tuple[VanillaSAE, dict]:
    """Download + load the LLaVA-NeXT-Mistral-7B SAE-V (L16) from HF."""
    cfg_path = hf_hub_download(SAE_V_REPO, f"{SAE_V_LLAVA_NEXT_SUBDIR}/cfg.json")
    w_path = hf_hub_download(SAE_V_REPO, f"{SAE_V_LLAVA_NEXT_SUBDIR}/sae_weights.safetensors")
    cfg = json.loads(Path(cfg_path).read_text())
    assert cfg["activation_fn"] == "relu" and cfg["architecture"] == "standard"
    assert cfg["model_name"] == "llava-hf/llava-v1.6-mistral-7b-hf"
    assert cfg["hook_name"] == "blocks.16.hook_resid_post"
    assert cfg["d_in"] == 4096 and cfg["d_sae"] == 65536
    sd = load_file(w_path)
    sae = VanillaSAE(
        W_enc=sd["W_enc"].to(dtype),
        b_enc=sd["b_enc"].to(dtype),
        W_dec=sd["W_dec"].to(dtype),
        b_dec=sd["b_dec"].to(dtype),
    ).to(device).eval()
    return sae, cfg
