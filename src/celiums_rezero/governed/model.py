"""Trainable action and evidence-pointer control head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ControlLogits:
    action_logits: torch.Tensor
    evidence_logits: torch.Tensor


class GovernedControlHead(nn.Module):
    def __init__(self, hidden_size: int, *, pointer_rank: int = 32) -> None:
        super().__init__()
        self.action = nn.Linear(hidden_size, 3)
        self.context = nn.Linear(hidden_size, pointer_rank, bias=False)
        self.evidence = nn.Linear(hidden_size, pointer_rank, bias=False)
        self.scale = pointer_rank**-0.5

    def forward(
        self,
        context_features: torch.Tensor,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> ControlLogits:
        if evidence_mask.dtype is not torch.bool:
            raise ValueError("evidence mask must be boolean")
        context = self.context(context_features).unsqueeze(1)
        evidence = self.evidence(evidence_features)
        pointers = (context * evidence).sum(-1) * self.scale
        pointers = pointers.masked_fill(~evidence_mask, -torch.inf)
        return ControlLogits(self.action(context_features), pointers)


def decode_control(
    logits: ControlLogits,
    evidence_mask: torch.Tensor,
    *,
    blocked: torch.Tensor,
    conflicting: torch.Tensor,
    minimum_confidence: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.action_logits, -1)
    confidence, actions = probabilities.max(-1)
    pointers = (torch.sigmoid(logits.evidence_logits) >= 0.5) & evidence_mask
    answer = 0
    abstain = 2
    forced = blocked | conflicting | (confidence < minimum_confidence)
    actions = torch.where(forced, torch.full_like(actions, abstain), actions)
    no_pointer = ~pointers.any(-1)
    actions = torch.where(
        (actions == answer) & no_pointer, torch.full_like(actions, abstain), actions
    )
    pointers = pointers & (actions == answer).unsqueeze(-1)
    return actions, pointers
