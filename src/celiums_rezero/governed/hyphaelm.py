"""ReZero neuropilot over bounded host certificates."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from celiums_rezero.governed.data import HOST_CONTROL_SIZES
from celiums_rezero.governed.navigation import HOST_CONTROL_CONTRACT
from celiums_rezero.transformer.block import TransformerBlock
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.norm import RMSNorm

NAVIGATION_ACTION_COUNT = 3
SEARCH_INDEX = 0
ANSWER_INDEX = 1
ABSTAIN_INDEX = 2


class ReZeroNeuroPilot(nn.Module):
    """Bounded ReZero controller for search/answer/abstain navigation steps."""

    def __init__(
        self,
        hidden_size: int,
        *,
        control_size: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        host_control_contract: str = HOST_CONTROL_CONTRACT,
        action_policy_prior_scale: float = 20.0,
        action_residual_bound: float = 1.0,
        pointer_policy_score: float = 0.72,
        pointer_policy_scale: float = 20.0,
        maximum_evidence_items: int = 8,
        action_terminal_bound: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or control_size < 2 or control_size % 2:
            raise ValueError("hidden and control sizes are invalid")
        if maximum_evidence_items < 1:
            raise ValueError("maximum evidence items must be positive")
        if action_residual_bound <= 0:
            raise ValueError("action residual bound must be positive")
        if action_terminal_bound < 0:
            raise ValueError("action terminal bound must be non-negative")
        if host_control_contract not in HOST_CONTROL_SIZES:
            raise ValueError("host control contract is invalid")
        self.hidden_size = hidden_size
        self.control_size = control_size
        self.maximum_evidence_items = maximum_evidence_items
        self.host_control_size = HOST_CONTROL_SIZES[host_control_contract]
        self.action_policy_prior_scale = action_policy_prior_scale
        self.action_residual_bound = action_residual_bound
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
        self.action = nn.Linear(control_size + self.host_control_size, NAVIGATION_ACTION_COUNT)
        self.pointer = nn.Linear(control_size, 1, bias=False)
        self.action_terminal_bound = action_terminal_bound
        self.action_terminal = nn.Parameter(torch.zeros(NAVIGATION_ACTION_COUNT))

    def forward(
        self,
        context_features: Tensor,
        evidence_features: Tensor,
        evidence_mask: Tensor,
        evidence_scores: Tensor,
        host_control_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        if evidence_scores.shape != expected_mask:
            raise ValueError("evidence scores have an invalid shape")
        if host_control_features.shape != (
            context_features.shape[0],
            self.host_control_size,
        ):
            raise ValueError("host control certificate is invalid")
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
        pooled = (hidden * sequence_mask.unsqueeze(-1)).sum(1) / sequence_mask.sum(1, keepdim=True)
        action_logits = self.action(torch.cat((pooled, host_control_features), dim=-1))
        action_logits = self.action_residual_bound * torch.tanh(action_logits)
        prior = torch.zeros_like(action_logits)
        certificate = host_control_features[:, -5:]
        prior[:, SEARCH_INDEX] = 0.0
        prior[:, ANSWER_INDEX] = certificate[:, 0]
        prior[:, ABSTAIN_INDEX] = certificate[:, 3] + certificate[:, 4]
        action_logits = action_logits + self.action_policy_prior_scale * prior
        if self.action_terminal_bound > 0:
            action_logits = action_logits + self.action_terminal_bound * torch.tanh(
                self.action_terminal
            )
        pointers = self.pointer(hidden[:, 1:]).squeeze(-1)
        pointers = self.action_residual_bound * torch.tanh(pointers)
        support = torch.where(
            evidence_scores >= self.pointer_policy_score,
            torch.ones_like(evidence_scores),
            -torch.ones_like(evidence_scores),
        )
        pointers = pointers + self.pointer_policy_scale * support
        pointers = pointers.masked_fill(~evidence_mask, -torch.inf)
        return action_logits, pointers
