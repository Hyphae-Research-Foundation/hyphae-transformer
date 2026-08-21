from __future__ import annotations

import torch
from torch import nn

from celiums_rezero.core.diagnostics import exact_jacobian_singular_values
from celiums_rezero.core.gates import ReZeroGate
from celiums_rezero.core.optim import build_optimizer_groups


class ToyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.branch = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.branch.weight)
        self.gate = ReZeroGate()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.gate(self.branch(inputs))


def test_zero_gate_is_exact_identity_and_isometric() -> None:
    block = ToyBlock(4)
    inputs = torch.randn(4, requires_grad=True)
    output = block(inputs)
    torch.testing.assert_close(output, inputs, rtol=0, atol=0)
    singular_values = exact_jacobian_singular_values(block, inputs)
    torch.testing.assert_close(singular_values, torch.ones_like(singular_values))


def test_first_backward_opens_gate_before_branch() -> None:
    block = ToyBlock(8)
    inputs = torch.randn(3, 8)
    target = torch.randn(3, 8)
    loss = (block(inputs) - target).square().mean()
    loss.backward()
    assert block.gate.value.grad is not None
    assert float(block.gate.value.grad.abs()) > 0
    assert block.branch.weight.grad is not None
    torch.testing.assert_close(
        block.branch.weight.grad,
        torch.zeros_like(block.branch.weight.grad),
        rtol=0,
        atol=0,
    )


def test_branch_receives_gradient_after_gate_opens() -> None:
    block = ToyBlock(8)
    block.gate.value.data.fill_(0.1)
    loss = block(torch.randn(3, 8)).square().mean()
    loss.backward()
    assert block.branch.weight.grad is not None
    assert float(block.branch.weight.grad.abs().sum()) > 0


def test_gate_stays_float32_when_model_casts() -> None:
    block = ToyBlock(8).to(dtype=torch.bfloat16)
    assert block.gate.value.dtype is torch.float32
    output = block(torch.randn(2, 8, dtype=torch.bfloat16))
    assert output.dtype is torch.bfloat16


def test_optimizer_groups_are_exhaustive_and_gates_do_not_decay() -> None:
    block = ToyBlock(4)
    groups = build_optimizer_groups(
        block, lr=1e-3, gate_lr=1e-2, weight_decay=0.1
    )
    assert {group["name"] for group in groups} == {"decay", "gates"}
    gate_group = next(group for group in groups if group["name"] == "gates")
    assert gate_group["lr"] == 1e-2
    assert gate_group["weight_decay"] == 0
