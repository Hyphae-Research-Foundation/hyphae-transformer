"""Transformer block with controlled residual topology."""

from __future__ import annotations

from typing import Literal, overload

from torch import Tensor, nn

from celiums_rezero.core.gates import ReZeroGate
from celiums_rezero.transformer.attention import CausalSelfAttention, KeyValueCache
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.mlp import SwiGLU
from celiums_rezero.transformer.norm import RMSNorm


class IdentityNorm(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class TransformerBlock(nn.Module):
    attention_norm: nn.Module
    mlp_norm: nn.Module
    attention: CausalSelfAttention
    mlp: SwiGLU
    attention_gate: ReZeroGate | None
    mlp_gate: ReZeroGate | None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        strategy = config.residual_strategy
        if strategy.uses_branch_norm:
            self.attention_norm = RMSNorm(config.d_model, config.rms_norm_epsilon)
            self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_epsilon)
        else:
            self.attention_norm = IdentityNorm()
            self.mlp_norm = IdentityNorm()
        self.attention = CausalSelfAttention(config)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)
        self.strategy = strategy

        if strategy.uses_gates:
            self.attention_gate = ReZeroGate(init=config.gate_init)
            if strategy.shares_gate:
                self.mlp_gate = self.attention_gate
            else:
                self.mlp_gate = ReZeroGate(init=config.gate_init)
        else:
            self.attention_gate = None
            self.mlp_gate = None

    def _residual(self, inputs: Tensor, branch: Tensor, gate: ReZeroGate | None) -> Tensor:
        branch = self.dropout(branch)
        return inputs + (branch if gate is None else gate(branch))

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        past_key_value: KeyValueCache | None = None,
        use_cache: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        past_key_value: KeyValueCache | None = None,
        use_cache: Literal[True],
    ) -> tuple[Tensor, KeyValueCache]: ...

    def forward(
        self,
        inputs: Tensor,
        *,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KeyValueCache]:
        normalized = self.attention_norm(inputs)
        if use_cache:
            attention, present_key_value = self.attention(
                normalized,
                past_key_value=past_key_value,
                use_cache=True,
            )
        else:
            attention = self.attention(normalized)
        inputs = self._residual(
            inputs,
            attention,
            self.attention_gate,
        )
        output = self._residual(
            inputs,
            self.mlp(self.mlp_norm(inputs)),
            self.mlp_gate,
        )
        return (output, present_key_value) if use_cache else output


def residual_gate_count(block: TransformerBlock) -> int:
    if block.strategy is ResidualStrategy.PRE_RMS:
        return 0
    return 1 if block.strategy.shares_gate else 2
