"""Complete decoder-only language model for controlled residual experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from celiums_rezero.transformer.attention import KeyValueCache
from celiums_rezero.transformer.block import TransformerBlock
from celiums_rezero.transformer.config import ModelConfig
from celiums_rezero.transformer.norm import RMSNorm

type PastKeyValues = tuple[KeyValueCache, ...]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: PastKeyValues | None = None


class ReZeroLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.reset_parameters()
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self) -> None:
        embedding_std = (
            self.config.d_model**-0.5
            if self.config.embedding_std is None
            else self.config.embedding_std
        )
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=embedding_std)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=embedding_std)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        *,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and targets is not None:
            raise ValueError("targets cannot be used with cached inference")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one cache per transformer block")
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[-2]
            if any(cache[0].shape[-2] != past_length for cache in past_key_values):
                raise ValueError("all transformer caches must have the same sequence length")
        if past_length + token_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        hidden = self.embedding_dropout(self.token_embedding(token_ids))
        present_key_values: list[KeyValueCache] = []
        for index, block in enumerate(self.blocks):
            if use_cache:
                hidden, present = block(
                    hidden,
                    past_key_value=(
                        None if past_key_values is None else past_key_values[index]
                    ),
                    use_cache=True,
                )
                present_key_values.append(present)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden)).float()
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return ModelOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(present_key_values) if use_cache else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not use_cache or self.training:
            return self._generate_uncached(
                token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
        output = token_ids
        past_key_values: PastKeyValues | None = None
        for _ in range(max_new_tokens):
            if past_key_values is None:
                current = output[:, -self.config.max_sequence_length :]
            else:
                current = output[:, -1:]
            model_output = self(
                current,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = self._sample_next_token(
                model_output.logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            output = torch.cat((output, next_token), dim=1)
            past_key_values = model_output.past_key_values
            assert past_key_values is not None
            if past_key_values[0][0].shape[-2] == self.config.max_sequence_length:
                past_key_values = None
        return output

    def _generate_uncached(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        output = token_ids
        for _ in range(max_new_tokens):
            context = output[:, -self.config.max_sequence_length :]
            next_token = self._sample_next_token(
                self(context).logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            output = torch.cat((output, next_token), dim=1)
        return output

    @staticmethod
    def _sample_next_token(
        logits: Tensor,
        *,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -math.inf)
        probabilities = functional.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator)

    def parameter_count(self, *, exclude_embeddings: bool = False) -> int:
        excluded = {id(self.token_embedding.weight)} if exclude_embeddings else set()
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in excluded
        )
