"""Frozen feature-backbone protocol and deterministic local fixture."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class PinnedBackboneIdentity:
    family: str
    model_id: str
    revision: str
    artifact_manifest_sha256: str
    tokenizer_manifest_sha256: str
    runtime_version: str
    feature_contract: str
    hidden_size: int


class FrozenTextBackbone(Protocol):
    @property
    def identity(self) -> PinnedBackboneIdentity: ...

    def encode(self, texts: tuple[str, ...], *, device: torch.device) -> torch.Tensor: ...

    def state_fingerprint(self) -> str: ...


def validate_frozen_features(
    backbone: FrozenTextBackbone, features: torch.Tensor, *, items: int
) -> None:
    if (
        features.shape != (items, backbone.identity.hidden_size)
        or features.dtype is not torch.float32
        or features.requires_grad
        or not torch.isfinite(features).all()
    ):
        raise ValueError("frozen backbone returned invalid feature tensors")


class FixtureBackboneV1:
    hidden_size = 128
    identity = PinnedBackboneIdentity(
        family="fixture",
        model_id="fixture://governed-byte-hash-v1",
        revision="fixture-v1",
        artifact_manifest_sha256="0" * 64,
        tokenizer_manifest_sha256="0" * 64,
        runtime_version="python-hashlib-v1",
        feature_contract="utf8-signed-ngram-l2-f32-v1",
        hidden_size=hidden_size,
    )

    def encode(self, texts: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        output = torch.zeros((len(texts), self.hidden_size), dtype=torch.float32)
        for row, text in enumerate(texts):
            encoded = text.encode()
            grams = [
                encoded[index : index + width]
                for width in (1, 2)
                for index in range(len(encoded) - width + 1)
            ]
            for gram in grams:
                digest = hashlib.sha256(gram).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.hidden_size
                output[row, bucket] += 1.0 if digest[4] & 1 else -1.0
        output = torch.nn.functional.normalize(output, dim=-1)
        return output.to(device).detach()

    def state_fingerprint(self) -> str:
        return hashlib.sha256(repr(self.identity).encode()).hexdigest()
