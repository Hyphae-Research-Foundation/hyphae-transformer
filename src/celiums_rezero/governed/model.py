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
    def __init__(
        self,
        hidden_size: int,
        *,
        pointer_rank: int = 32,
        normalized_features: bool = False,
        use_evidence_scores: bool = False,
        pointer_policy_score: float | None = None,
        pointer_policy_scale: float = 1.0,
        use_host_control_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_host_control_features = use_host_control_features
        self.action = nn.Linear(hidden_size + (5 if use_host_control_features else 0), 3)
        self.context = nn.Linear(hidden_size, pointer_rank, bias=False)
        self.evidence = nn.Linear(hidden_size, pointer_rank, bias=False)
        self.scale = pointer_rank**-0.5
        self.normalized_features = normalized_features
        self.evidence_score = nn.Linear(1, 1) if use_evidence_scores else None
        self.pointer_policy_score = pointer_policy_score
        self.pointer_policy_scale = pointer_policy_scale

    def forward(
        self,
        context_features: torch.Tensor,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_scores: torch.Tensor | None = None,
        host_control_features: torch.Tensor | None = None,
    ) -> ControlLogits:
        if evidence_mask.dtype is not torch.bool:
            raise ValueError("evidence mask must be boolean")
        if self.normalized_features:
            context_features = nn.functional.normalize(context_features, dim=-1)
            evidence_features = nn.functional.normalize(evidence_features, dim=-1)
        action_features = context_features
        if self.use_host_control_features:
            if host_control_features is None or host_control_features.shape != (
                context_features.shape[0],
                5,
            ):
                raise ValueError("host control features are required by this control head")
            action_features = torch.cat((context_features, host_control_features), dim=-1)
        context = self.context(context_features).unsqueeze(1)
        evidence = self.evidence(evidence_features)
        pointers = (context * evidence).sum(-1) * self.scale
        if self.evidence_score is not None:
            if evidence_scores is None or evidence_scores.shape != evidence_mask.shape:
                raise ValueError("evidence scores are required by this control head")
            pointers = pointers + self.evidence_score(evidence_scores.unsqueeze(-1)).squeeze(-1)
        if self.pointer_policy_score is not None:
            if evidence_scores is None or evidence_scores.shape != evidence_mask.shape:
                raise ValueError("evidence scores are required by the pointer policy prior")
            pointers = pointers + self.pointer_policy_scale * (
                evidence_scores - self.pointer_policy_score
            )
        pointers = pointers.masked_fill(~evidence_mask, -torch.inf)
        return ControlLogits(self.action(action_features), pointers)


def decode_control(
    logits: ControlLogits,
    evidence_mask: torch.Tensor,
    *,
    blocked: torch.Tensor,
    conflicting: torch.Tensor,
    minimum_confidence: float = 0.7,
    pointer_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.action_logits, -1)
    confidence, actions = probabilities.max(-1)
    pointers = (torch.sigmoid(logits.evidence_logits) >= pointer_threshold) & evidence_mask
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
