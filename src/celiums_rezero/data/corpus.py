"""Deterministic batches and exhaustive evaluation for continuous token corpora."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import log

import torch
from torch import Tensor

from celiums_rezero.transformer.model import ReZeroLM


class ContinuousSequenceSource:
    def __init__(self, tokens: Tensor, *, sequence_length: int, seed: int) -> None:
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a one-dimensional torch.long tensor")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if tokens.numel() <= sequence_length:
            raise ValueError("corpus must contain more tokens than sequence_length")
        self.tokens = tokens.cpu()
        self.sequence_length = sequence_length
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.offsets = torch.arange(sequence_length + 1)

    def batch(self, batch_size: int, *, device: torch.device | str) -> tuple[Tensor, Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        starts = torch.randint(
            0,
            self.tokens.numel() - self.sequence_length,
            (batch_size,),
            generator=self.generator,
        )
        sequences = self.tokens[starts[:, None] + self.offsets]
        return sequences[:, :-1].to(device), sequences[:, 1:].to(device)

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.generator.set_state(state["generator_state"])


@dataclass(frozen=True, slots=True)
class CorpusEvaluation:
    nll: float
    bits_per_token: float
    tokens: int


@torch.inference_mode()
def evaluate_corpus(
    model: ReZeroLM,
    tokens: Tensor,
    *,
    batch_size: int = 8,
    deadline: float | None = None,
) -> CorpusEvaluation:
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("tokens must be a one-dimensional torch.long tensor")
    if tokens.numel() < 2:
        raise ValueError("evaluation corpus must contain at least two tokens")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        sequence_length = model.config.max_sequence_length
        transition_count = tokens.numel() - 1
        full_window_count = transition_count // sequence_length
        total_nll = 0.0
        evaluated_tokens = 0

        if full_window_count:
            end = full_window_count * sequence_length + 1
            windows = tokens[:end].unfold(0, sequence_length + 1, sequence_length)
            for start in range(0, full_window_count, batch_size):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("evaluation wall-time budget exceeded")
                batch = windows[start : start + batch_size].long().to(device)
                output = model(batch[:, :-1], batch[:, 1:])
                if output.loss is None:
                    raise RuntimeError("evaluation forward did not return a loss")
                count = batch[:, :-1].numel()
                total_nll += float(output.loss) * count
                evaluated_tokens += count

        remainder = transition_count - evaluated_tokens
        if remainder:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("evaluation wall-time budget exceeded")
            batch = tokens[evaluated_tokens:].unsqueeze(0).long().to(device)
            output = model(batch[:, :-1], batch[:, 1:])
            if output.loss is None:
                raise RuntimeError("evaluation forward did not return a loss")
            total_nll += float(output.loss) * remainder
            evaluated_tokens += remainder
    finally:
        model.train(was_training)
    mean_nll = total_nll / evaluated_tokens
    return CorpusEvaluation(
        nll=mean_nll,
        bits_per_token=mean_nll / log(2),
        tokens=evaluated_tokens,
    )
