"""Deterministic structured sequences for correctness and promotion smokes."""

from __future__ import annotations

import torch
from torch import Tensor


class SyntheticSequenceSource:
    """Generate next-token data with local and long-range deterministic structure."""

    def __init__(self, *, vocab_size: int, sequence_length: int, seed: int) -> None:
        if vocab_size < 16:
            raise ValueError("synthetic vocab_size must be at least 16")
        if sequence_length < 4:
            raise ValueError("sequence_length must be at least 4")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def batch(self, batch_size: int, *, device: torch.device | str) -> tuple[Tensor, Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        starts = torch.randint(
            3,
            self.vocab_size,
            (batch_size, 2),
            generator=self.generator,
            dtype=torch.long,
        )
        sequence = torch.empty(batch_size, self.sequence_length + 1, dtype=torch.long)
        sequence[:, :2] = starts
        modulus = self.vocab_size - 3
        for position in range(2, self.sequence_length + 1):
            sequence[:, position] = (
                sequence[:, position - 1]
                + sequence[:, position - 2]
                + position
            ) % modulus + 3
        return sequence[:, :-1].to(device), sequence[:, 1:].to(device)

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.generator.set_state(state["generator_state"])
