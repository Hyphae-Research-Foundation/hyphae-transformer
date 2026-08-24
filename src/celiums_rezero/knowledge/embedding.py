"""Pinned local-only semantic embedding providers shared by ingest and retrieval."""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import ClassVar, Protocol

import torch

from celiums_rezero.lab.serialization import canonical_json


class EmbeddingProvider(Protocol):
    @property
    def profile(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    model_id: str
    revision: str
    artifact_manifest_sha256: str
    dimensions: int
    maximum_tokens: int
    pooling: str
    normalization: str
    runtime_version: str

    @property
    def profile(self) -> str:
        return f"sememb_{hashlib.sha256(canonical_json(self).encode()).hexdigest()}"


class MiniLML6V2EmbeddingProvider:
    """English 384D MiniLM embeddings with mean pooling and L2 normalization."""

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    dimensions = 384
    maximum_tokens = 256
    required_transformers = "5.14.1"
    required_artifacts: ClassVar[dict[str, tuple[int, str]]] = {
        "1_Pooling/config.json": (
            190,
            "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
        ),
        "config.json": (
            612,
            "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        ),
        "config_sentence_transformers.json": (
            116,
            "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
        ),
        "model.safetensors": (
            90_868_376,
            "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        ),
        "modules.json": (
            349,
            "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
        ),
        "sentence_bert_config.json": (
            53,
            "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
        ),
        "special_tokens_map.json": (
            112,
            "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
        ),
        "tokenizer.json": (
            466_247,
            "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        ),
        "tokenizer_config.json": (
            350,
            "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
        ),
        "vocab.txt": (
            231_508,
            "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        ),
    }

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve(strict=True)
        self._verify_artifacts()
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise RuntimeError("MiniLM embedder requires transformers==5.14.1") from error
        if getattr(transformers, "__version__", None) != self.required_transformers:
            raise RuntimeError("MiniLM embedder requires transformers==5.14.1 exactly")
        tokenizer_class = transformers.BertTokenizerFast
        model_class = transformers.BertModel
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = model_class.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to("cpu")
        self.model.requires_grad_(False)
        self.model.eval()
        self.identity = EmbeddingIdentity(
            self.model_id,
            self.revision,
            self.artifact_manifest_sha256(),
            self.dimensions,
            self.maximum_tokens,
            "attention-mask-mean-f32-v1",
            "l2-f32-v1",
            f"torch-{torch.__version__};transformers-{self.required_transformers}",
        )

    @property
    def profile(self) -> str:
        return self.identity.profile

    @torch.inference_mode()
    def embed(self, text: str) -> tuple[float, ...]:
        if not text.strip() or len(text.encode()) > 65_536:
            raise ValueError("embedding input is empty or exceeds its byte bound")
        batch = self.tokenizer(
            text,
            truncation=True,
            max_length=self.maximum_tokens,
            return_tensors="pt",
        )
        output = self.model(**batch, return_dict=True)
        hidden = output.last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)[0].cpu()
        if normalized.shape != (self.dimensions,) or not torch.isfinite(normalized).all():
            raise RuntimeError("embedding output shape or values are invalid")
        norm = float(torch.linalg.vector_norm(normalized).item())
        if abs(norm - 1.0) > 1e-5:
            raise RuntimeError("embedding output is not unit normalized")
        return tuple(float(value) for value in normalized.tolist())

    @classmethod
    def artifact_manifest_sha256(cls) -> str:
        return hashlib.sha256(
            canonical_json(cls.required_artifacts).encode()
        ).hexdigest()

    def _verify_artifacts(self) -> None:
        observed = {
            str(path.relative_to(self.model_path))
            for path in self.model_path.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(self.model_path).parts
        }
        if observed != set(self.required_artifacts):
            raise ValueError("MiniLM artifact set differs from its pin")
        for name, (expected_bytes, expected_digest) in self.required_artifacts.items():
            path = self.model_path / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"MiniLM artifact is absent or unsafe: {name}")
            if _sha256(path) != expected_digest:
                raise ValueError(f"MiniLM artifact digest mismatch: {name}")
        config = json.loads((self.model_path / "config.json").read_text())
        pooling = json.loads((self.model_path / "1_Pooling/config.json").read_text())
        modules = json.loads((self.model_path / "modules.json").read_text())
        if (
            config.get("model_type") != "bert"
            or config.get("architectures") != ["BertModel"]
            or config.get("hidden_size") != self.dimensions
            or "auto_map" in config
            or pooling
            != {
                "word_embedding_dimension": self.dimensions,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            }
            or [item.get("type") for item in modules]
            != [
                "sentence_transformers.models.Transformer",
                "sentence_transformers.models.Pooling",
                "sentence_transformers.models.Normalize",
            ]
        ):
            raise ValueError("MiniLM architecture or pooling contract differs")


def minilm_l6_v2_preflight(model_path: Path) -> dict[str, object]:
    provider = MiniLML6V2EmbeddingProvider(model_path)
    query = provider.embed("What is the approved maintenance window?")
    passage = provider.embed(
        "Service policy: approved maintenance window is 02:00-04:00 UTC."
    )
    distractor = provider.embed("The ocean is blue.")
    query_digest = hashlib.sha256(struct.pack("<384f", *query)).hexdigest()
    return {
        "schema": "hyphae-transformer.minilm-l6-v2-preflight/v1",
        "ready": True,
        "model_id": provider.model_id,
        "revision": provider.revision,
        "artifact_manifest_sha256": provider.artifact_manifest_sha256(),
        "profile": provider.profile,
        "dimensions": provider.dimensions,
        "query_f32_sha256": query_digest,
        "matching_cosine": sum(left * right for left, right in zip(query, passage, strict=True)),
        "distractor_cosine": sum(
            left * right for left, right in zip(query, distractor, strict=True)
        ),
    }


def minilm_l6_v2_factory(*, tenant: object, config: dict[str, object]) -> object:
    del tenant
    if set(config) != {"model_path"} or not isinstance(config["model_path"], str):
        raise ValueError("MiniLM provider config must contain only model_path")
    return MiniLML6V2EmbeddingProvider(Path(config["model_path"]))


def checked_embedding(provider: EmbeddingProvider, text: str) -> tuple[float, ...]:
    values = provider.embed(text)
    if (
        isinstance(provider.dimensions, bool)
        or provider.dimensions < 1
        or len(values) != provider.dimensions
        or any(not isfinite(value) for value in values)
    ):
        raise ValueError("embedding provider returned invalid dimensions or values")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
