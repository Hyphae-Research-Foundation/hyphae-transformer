"""Validated model configuration and residual strategy definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ResidualStrategy(StrEnum):
    PRE_RMS = "pre_rms"
    REZERO_CANONICAL = "rezero_canonical"
    REZERO_SPLIT = "rezero_split"
    REZERO_RMS_SHARED = "rezero_rms_shared"
    CRZ_RMS = "crz_rms"

    @property
    def uses_branch_norm(self) -> bool:
        return self in {
            ResidualStrategy.PRE_RMS,
            ResidualStrategy.REZERO_RMS_SHARED,
            ResidualStrategy.CRZ_RMS,
        }

    @property
    def uses_gates(self) -> bool:
        return self is not ResidualStrategy.PRE_RMS

    @property
    def shares_gate(self) -> bool:
        return self in {
            ResidualStrategy.REZERO_CANONICAL,
            ResidualStrategy.REZERO_RMS_SHARED,
        }


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 512
    max_sequence_length: int = 256
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int | None = None
    d_ff: int | None = None
    rope_theta: float = 10_000.0
    rms_norm_epsilon: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_strategy: ResidualStrategy = ResidualStrategy.CRZ_RMS
    gate_init: float = 0.0
    tie_embeddings: bool = True
    embedding_std: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.residual_strategy, str):
            object.__setattr__(
                self, "residual_strategy", ResidualStrategy(self.residual_strategy)
            )
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.d_model < 2 or self.d_model % 2:
            raise ValueError("d_model must be an even integer")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("n_heads must divide d_model")
        n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        if n_kv_heads < 1 or self.n_heads % n_kv_heads:
            raise ValueError("n_kv_heads must divide n_heads")
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.feed_forward_size < 1:
            raise ValueError("d_ff must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_epsilon <= 0:
            raise ValueError("rms_norm_epsilon must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not self.residual_strategy.uses_gates and self.gate_init != 0:
            raise ValueError("gate_init only applies to gated residual strategies")
        object.__setattr__(self, "n_kv_heads", n_kv_heads)
        object.__setattr__(self, "d_ff", self.feed_forward_size)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def feed_forward_size(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        proposed = int(8 * self.d_model / 3)
        return 64 * ((proposed + 63) // 64)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["residual_strategy"] = self.residual_strategy.value
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**values)
