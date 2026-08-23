"""Learnable residual gates with an explicit optimizer contract."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class ReZeroGate(nn.Module):
    """Scale a residual branch with a learnable FP32 parameter.

    The parameter remains FP32 when the model is cast to a lower precision. The
    multiplication is performed in the branch dtype so the gate composes with AMP,
    BF16, and compiled graphs without promoting the full activation tensor.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        *,
        init: float = 0.0,
        trainable: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.value = nn.Parameter(
            torch.full(shape or (), init, dtype=torch.float32, device=device),
            requires_grad=trainable,
        )
        self.value._celiums_gate = True  # type: ignore[attr-defined]
        self.value._no_weight_decay = True  # type: ignore[attr-defined]

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> ReZeroGate:
        super()._apply(fn, recurse=recurse)  # type: ignore[no-untyped-call]
        if self.value.dtype != torch.float32:
            self.value.data = self.value.data.float()
            if self.value.grad is not None:
                self.value.grad.data = self.value.grad.data.float()
        return self

    def forward(self, branch: Tensor) -> Tensor:
        return branch * self.value.to(device=branch.device, dtype=branch.dtype)

    def extra_repr(self) -> str:
        value = self.value.detach()
        return f"shape={tuple(value.shape)}, value={value.flatten().tolist()}"


def is_gate_parameter(parameter: nn.Parameter) -> bool:
    """Return whether a parameter declares itself as a Hyphae Transformer residual gate."""

    return bool(getattr(parameter, "_celiums_gate", False))
