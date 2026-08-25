"""Trainable action and evidence-pointer control head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from celiums_rezero.transformer.block import TransformerBlock
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.norm import RMSNorm


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


class ReZeroSequenceControlHead(nn.Module):
    """Sequential governed controller using the shared-gate ReZero topology."""

    def __init__(
        self,
        hidden_size: int,
        *,
        control_size: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        pointer_policy_score: float = 0.72,
        pointer_policy_scale: float = 20.0,
        use_host_control_features: bool = True,
        host_control_size: int = 5,
        action_policy_prior_scale: float = 0.0,
        action_residual_bound: float | None = None,
        pointer_residual_bound: float | None = None,
        maximum_evidence_items: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or control_size < 2 or control_size % 2:
            raise ValueError("hidden and control sizes are invalid")
        if maximum_evidence_items < 1:
            raise ValueError("maximum evidence items must be positive")
        self.hidden_size = hidden_size
        self.control_size = control_size
        self.maximum_evidence_items = maximum_evidence_items
        self.use_host_control_features = use_host_control_features
        self.host_control_size = host_control_size
        self.action_policy_prior_scale = action_policy_prior_scale
        self.action_residual_bound = action_residual_bound
        self.pointer_residual_bound = pointer_residual_bound
        if action_residual_bound is not None and action_residual_bound <= 0:
            raise ValueError("action residual bound must be positive")
        if pointer_residual_bound is not None and pointer_residual_bound <= 0:
            raise ValueError("pointer residual bound must be positive")
        self.pointer_policy_score = pointer_policy_score
        self.pointer_policy_scale = pointer_policy_scale
        config = ModelConfig(
            vocab_size=2,
            max_sequence_length=maximum_evidence_items + 1,
            n_layers=n_layers,
            d_model=control_size,
            n_heads=n_heads,
            residual_strategy=ResidualStrategy.REZERO_RMS_SHARED,
            tie_embeddings=False,
        )
        self.context_projection = nn.Linear(hidden_size, control_size, bias=False)
        self.evidence_projection = nn.Linear(hidden_size, control_size, bias=False)
        self.type_embedding = nn.Embedding(2, control_size)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(n_layers))
        self.final_norm = RMSNorm(control_size, config.rms_norm_epsilon)
        self.action = nn.Linear(
            control_size + (host_control_size if use_host_control_features else 0), 3
        )
        self.pointer = nn.Linear(control_size, 1, bias=False)

    def forward(
        self,
        context_features: torch.Tensor,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_scores: torch.Tensor | None = None,
        host_control_features: torch.Tensor | None = None,
    ) -> ControlLogits:
        if context_features.ndim != 2 or context_features.shape[-1] != self.hidden_size:
            raise ValueError("context features have an invalid shape")
        expected_evidence = (
            context_features.shape[0],
            self.maximum_evidence_items,
            self.hidden_size,
        )
        if evidence_features.shape != expected_evidence:
            raise ValueError("evidence features have an invalid shape")
        expected_mask = expected_evidence[:2]
        if evidence_mask.dtype is not torch.bool or evidence_mask.shape != expected_mask:
            raise ValueError("evidence mask must be boolean with the configured shape")
        if evidence_scores is None or evidence_scores.shape != expected_mask:
            raise ValueError("evidence scores are required by the ReZero controller")
        context = self.context_projection(context_features).unsqueeze(1)
        evidence = self.evidence_projection(evidence_features)
        types = torch.cat(
            (
                torch.zeros((context.shape[0], 1), dtype=torch.long, device=context.device),
                torch.ones(expected_mask, dtype=torch.long, device=context.device),
            ),
            dim=1,
        )
        hidden = torch.cat((context, evidence), dim=1) + self.type_embedding(types)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        sequence_mask = torch.cat(
            (
                torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device),
                evidence_mask,
            ),
            dim=1,
        )
        action_features = (hidden * sequence_mask.unsqueeze(-1)).sum(1) / sequence_mask.sum(
            1, keepdim=True
        )
        if self.use_host_control_features:
            if host_control_features is None or host_control_features.shape != (
                context_features.shape[0],
                self.host_control_size,
            ):
                raise ValueError("host control features are required by this controller")
            action_features = torch.cat((action_features, host_control_features), dim=-1)
        action_logits = self.action(action_features)
        if self.action_residual_bound is not None:
            action_logits = self.action_residual_bound * torch.tanh(action_logits)
        if self.action_policy_prior_scale:
            if host_control_features is None or self.host_control_size < 5:
                raise ValueError("action policy prior requires a host policy certificate")
            certificate = host_control_features[:, -5:]
            if not torch.allclose(
                certificate.sum(-1), torch.ones(certificate.shape[0], device=certificate.device)
            ):
                raise ValueError("host policy decision certificate must be one-hot")
            policy_actions = torch.stack(
                (
                    certificate[:, 0],
                    certificate[:, 1] + certificate[:, 2],
                    certificate[:, 3] + certificate[:, 4],
                ),
                dim=-1,
            )
            action_logits = action_logits + self.action_policy_prior_scale * policy_actions
        pointers = self.pointer(hidden[:, 1:]).squeeze(-1)
        if self.pointer_residual_bound is None:
            pointers = pointers + self.pointer_policy_scale * (
                evidence_scores - self.pointer_policy_score
            )
        else:
            pointers = self.pointer_residual_bound * torch.tanh(pointers)
            support = torch.where(
                evidence_scores >= self.pointer_policy_score,
                torch.ones_like(evidence_scores),
                -torch.ones_like(evidence_scores),
            )
            pointers = pointers + self.pointer_policy_scale * support
        pointers = pointers.masked_fill(~evidence_mask, -torch.inf)
        return ControlLogits(action_logits, pointers)


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
