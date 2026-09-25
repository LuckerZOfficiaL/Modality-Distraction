"""Hook decoder layers of Qwen2.5-VL and extract last-non-pad-input-token states.

Usage:
    extractor = ActivationExtractor(model, layer_indices=range(36))
    with extractor:
        # run the model normally
        out = model(**inputs)
        # extractor.layer_states[i] -> [batch, seq, d_model]
    last_per_row = extractor.last_token_states(inputs.attention_mask)  # [batch, n_layers, d_model]
"""
from __future__ import annotations

from typing import Iterable

import torch


class ActivationExtractor:
    def __init__(self, model, layer_indices: Iterable[int]):
        self.model = model
        self.layer_indices = sorted(set(int(i) for i in layer_indices))
        self._handles: list = []
        self.layer_states: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.layer_states[layer_idx] = h.detach()
        return hook

    def __enter__(self):
        layers = self.model.model.language_model.layers
        for idx in self.layer_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))
        self.layer_states = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []

    def last_token_states(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return [batch, n_layers, d_model] gathered at each row's last unmasked token.

        Works for both right-padding (mask = [1,1,1,0,0]) and left-padding
        (mask = [0,0,1,1,1]); identical to mask.sum()-1 under right-pad.

        Layers are stacked in the order of self.layer_indices.
        """
        # last index where mask==1 per row
        last_idx = (attention_mask.cumsum(dim=1) * attention_mask).argmax(dim=1)  # [batch]
        out = []
        for layer in self.layer_indices:
            h = self.layer_states[layer]  # [batch, seq, d_model]
            gathered = h[torch.arange(h.size(0), device=h.device), last_idx]
            out.append(gathered)
        return torch.stack(out, dim=1)  # [batch, n_layers, d_model]
