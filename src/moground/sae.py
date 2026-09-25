"""Top-K Sparse Autoencoder (Gao et al. 2024).

Forward: z = ReLU(TopK(W_enc (x - b_dec) + b_enc)),  x_hat = W_dec z + b_dec.
Decoder rows are kept unit-norm; encoder is initialized as decoder transpose.
b_dec is initialized to the mean of training activations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_model: int
    d_sae: int
    k: int


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        W = torch.randn(cfg.d_sae, cfg.d_model)
        W = F.normalize(W, dim=1)
        self.W_dec = nn.Parameter(W.clone())                # [d_sae, d_model]
        self.W_enc = nn.Parameter(W.clone().T.contiguous()) # [d_model, d_sae]
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

    @torch.no_grad()
    def init_b_dec(self, x_mean: torch.Tensor) -> None:
        self.b_dec.copy_(x_mean.to(self.b_dec.dtype))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.copy_(F.normalize(self.W_dec, dim=1))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        vals = F.relu(vals)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z, idx

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        pre = self.encode_pre(x)
        z, idx = self.topk(pre)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z, "indices": idx, "pre": pre}


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


@torch.no_grad()
def remove_decoder_parallel_grad(sae: TopKSAE) -> None:
    """Project out the component of W_dec.grad parallel to W_dec rows
    (so the unit-norm constraint stays tight under Adam updates)."""
    if sae.W_dec.grad is None:
        return
    g = sae.W_dec.grad
    w = sae.W_dec
    parallel = (g * w).sum(dim=1, keepdim=True) * w
    g.sub_(parallel)
