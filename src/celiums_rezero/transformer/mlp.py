"""SwiGLU feed-forward branch."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from celiums_rezero.transformer.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        inner = config.feed_forward_size
        self.gate_proj = nn.Linear(config.d_model, inner, bias=False)
        self.up_proj = nn.Linear(config.d_model, inner, bias=False)
        self.down_proj = nn.Linear(inner, config.d_model, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        output: Tensor = self.down_proj(
            functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        )
        return output
