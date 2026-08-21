"""A lossless byte tokenizer for historical enwiki8 comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class ByteTokenizer:
    """Map raw bytes to token IDs while reserving three special tokens."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = 259

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        tokens = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        values = bytes(
            token - self.byte_offset
            for token in tokens
            if self.byte_offset <= token < self.vocab_size
        )
        return values.decode("utf-8", errors="replace")


def load_byte_corpus(
    path: Path | str,
    *,
    start: int = 0,
    limit: int | None = None,
    byte_offset: int = ByteTokenizer.byte_offset,
) -> Tensor:
    if start < 0 or byte_offset < 0:
        raise ValueError("start and byte_offset cannot be negative")
    with Path(path).open("rb") as source:
        source.seek(start)
        raw = source.read() if limit is None else source.read(limit)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if len(raw) < 2:
        raise ValueError("corpus must contain at least two bytes")
    values = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    if byte_offset:
        values += byte_offset
    return torch.from_numpy(values)
