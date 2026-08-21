"""Rotary position embedding cache and application."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    cosine: Tensor
    sine: Tensor

    def __init__(self, head_dim: int, max_sequence_length: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, inputs: Tensor, *, offset: int = 0) -> Tensor:
        sequence_length = inputs.shape[-2]
        end = offset + sequence_length
        if end > self.cosine.shape[0]:
            raise ValueError(
                f"RoPE position {end} exceeds cache length {self.cosine.shape[0]}"
            )
        cosine = self.cosine[offset:end].to(device=inputs.device, dtype=inputs.dtype)
        sine = self.sine[offset:end].to(device=inputs.device, dtype=inputs.dtype)
        cosine = cosine.view(1, 1, sequence_length, -1)
        sine = sine.view(1, 1, sequence_length, -1)
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        output = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return output.flatten(-2)
