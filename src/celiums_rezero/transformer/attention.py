"""Grouped-query causal self-attention using PyTorch SDPA."""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from celiums_rezero.transformer.config import ModelConfig
from celiums_rezero.transformer.rope import RotaryEmbedding

type KeyValueCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        assert config.n_kv_heads is not None
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.d_model, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.d_model, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, self.n_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_sequence_length, config.rope_theta
        )

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
        if past_key_value is not None and not use_cache:
            raise ValueError("past_key_value requires use_cache=True")
        batch_size, sequence_length, _ = inputs.shape
        queries = self.q_proj(inputs).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        )
        keys = self.k_proj(inputs).view(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        )
        values = self.v_proj(inputs).view(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        )
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        past_length = 0
        if past_key_value is not None:
            past_keys, past_values = past_key_value
            expected_prefix = (batch_size, self.n_kv_heads)
            if past_keys.shape != past_values.shape or past_keys.ndim != 4:
                raise ValueError("cached keys and values must have matching rank-4 shapes")
            if past_keys.shape[:2] != expected_prefix or past_keys.shape[-1] != self.head_dim:
                raise ValueError("cache shape does not match attention configuration")
            if past_keys.device != keys.device or past_keys.dtype != keys.dtype:
                raise ValueError("cache device and dtype must match current keys")
            past_length = past_keys.shape[-2]

        queries = self.rope(queries, offset=past_length)
        keys = self.rope(keys, offset=past_length)
        if past_key_value is not None:
            keys = torch.cat((past_keys, keys), dim=-2)
            values = torch.cat((past_values, values), dim=-2)
        present_key_value = (keys, values)

        if self.n_kv_heads != self.n_heads:
            repetitions = self.n_heads // self.n_kv_heads
            keys = keys.repeat_interleave(repetitions, dim=1)
            values = values.repeat_interleave(repetitions, dim=1)

        attention_mask = None
        is_causal = past_length == 0
        if past_length > 0 and sequence_length > 1:
            query_positions = past_length + torch.arange(
                sequence_length, device=queries.device
            )
            key_positions = torch.arange(keys.shape[-2], device=queries.device)
            attention_mask = key_positions[None, :] <= query_positions[:, None]
        attended = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        output = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.n_heads * self.head_dim
        )
        projected: Tensor = self.out_proj(output)
        return (projected, present_key_value) if use_cache else projected
