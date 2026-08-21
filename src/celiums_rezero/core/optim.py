"""Optimizer grouping that does not depend on fragile parameter-name rules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch import nn

from celiums_rezero.core.gates import is_gate_parameter


def build_optimizer_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    gate_lr: float | None = None,
) -> list[dict[str, Any]]:
    """Create exhaustive, non-overlapping decay, no-decay, and gate groups."""

    if lr <= 0:
        raise ValueError("lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if gate_lr is not None and gate_lr <= 0:
        raise ValueError("gate_lr must be positive")

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    gates: list[nn.Parameter] = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if is_gate_parameter(parameter):
            gates.append(parameter)
        elif parameter.ndim < 2 or bool(getattr(parameter, "_no_weight_decay", False)):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {"name": "decay", "params": decay, "lr": lr, "weight_decay": weight_decay}
        )
    if no_decay:
        groups.append({"name": "no_decay", "params": no_decay, "lr": lr, "weight_decay": 0.0})
    if gates:
        groups.append(
            {
                "name": "gates",
                "params": gates,
                "lr": gate_lr if gate_lr is not None else lr,
                "weight_decay": 0.0,
            }
        )

    _validate_groups(model.parameters(), groups)
    return groups


def _validate_groups(
    parameters: Iterable[nn.Parameter], groups: list[dict[str, Any]]
) -> None:
    expected = {id(parameter) for parameter in parameters if parameter.requires_grad}
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    if len(grouped) != len(set(grouped)):
        raise RuntimeError("optimizer parameter groups overlap")
    if set(grouped) != expected:
        raise RuntimeError("optimizer parameter groups do not cover every trainable parameter")
