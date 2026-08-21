"""Signal, gradient, and Jacobian diagnostics for residual architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from celiums_rezero.core.gates import is_gate_parameter


def root_mean_square(value: Tensor) -> Tensor:
    return value.float().square().mean().sqrt()


@dataclass(frozen=True, slots=True)
class TensorStats:
    rms: float
    maximum: float
    minimum: float
    finite: bool

    @classmethod
    def from_tensor(cls, value: Tensor) -> TensorStats:
        detached = value.detach().float()
        return cls(
            rms=float(root_mean_square(detached)),
            maximum=float(detached.max()),
            minimum=float(detached.min()),
            finite=bool(torch.isfinite(detached).all()),
        )


@dataclass(frozen=True, slots=True)
class GateStats:
    name: str
    value: TensorStats
    gradient: TensorStats | None


def collect_gate_stats(model: nn.Module) -> list[GateStats]:
    output: list[GateStats] = []
    for name, parameter in model.named_parameters():
        if is_gate_parameter(parameter):
            output.append(
                GateStats(
                    name=name,
                    value=TensorStats.from_tensor(parameter),
                    gradient=(
                        TensorStats.from_tensor(parameter.grad)
                        if parameter.grad is not None
                        else None
                    ),
                )
            )
    return output


def assert_finite_model(model: nn.Module, *, include_gradients: bool = True) -> None:
    """Fail with the exact parameter or buffer name that became non-finite."""

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite parameter: {name}")
        if (
            include_gradients
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ):
            raise FloatingPointError(f"non-finite gradient: {name}")
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and not torch.isfinite(buffer).all():
            raise FloatingPointError(f"non-finite buffer: {name}")


def exact_jacobian_singular_values(
    function: nn.Module,
    inputs: Tensor,
) -> Tensor:
    """Compute exact singular values for intentionally small diagnostic inputs."""

    def flattened(value: Tensor) -> Tensor:
        result = function(value)
        if not isinstance(result, Tensor):
            raise TypeError("diagnostic module must return a Tensor")
        return result.reshape(-1)

    jacobian: Tensor = torch.autograd.functional.jacobian(  # type: ignore[no-untyped-call]
        flattened, inputs, vectorize=True
    )
    matrix = jacobian.reshape(jacobian.shape[0], -1).float()
    values: Tensor = torch.linalg.svdvals(matrix)
    return values


def effective_rank(value: Tensor, *, epsilon: float = 1e-12) -> float:
    """Entropy-based effective rank of the final feature dimension."""

    matrix = value.detach().float().reshape(-1, value.shape[-1])
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    if float(total) <= epsilon:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * (probabilities + epsilon).log()).sum()
    return float(entropy.exp())
