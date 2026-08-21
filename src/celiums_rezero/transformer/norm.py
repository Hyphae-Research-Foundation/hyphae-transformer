"""Small explicit RMS normalization module."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dimension: int,
        epsilon: float = 1e-5,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension, device=device))
        self.weight._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(inputs.dtype) * self.weight.to(inputs.dtype)
