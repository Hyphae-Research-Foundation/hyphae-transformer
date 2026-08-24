from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from celiums_rezero.knowledge.embedding import (
    MiniLML6V2EmbeddingProvider,
    checked_embedding,
)


def pinned_model(tmp_path: Path) -> Path:
    source = Path("/tmp/opencode/minilm-l6-v2")
    if not source.is_dir():
        pytest.skip("pinned MiniLM fixture is not available")
    target = tmp_path / "model"
    shutil.copytree(source, target)
    return target


def test_minilm_manifest_is_exact() -> None:
    assert MiniLML6V2EmbeddingProvider.artifact_manifest_sha256() == (
        hashlib.sha256(
            json.dumps(
                MiniLML6V2EmbeddingProvider.required_artifacts,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert MiniLML6V2EmbeddingProvider.dimensions == 384
    assert MiniLML6V2EmbeddingProvider.artifact_manifest_sha256() == (
        "e5d9d07b6db0c99cc4a2afa92047d57b84c3cb6ed48137ad3612601fdbe21411"
    )


def test_minilm_provider_mean_pools_and_normalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = pinned_model(tmp_path)

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args: object, **kwargs: object) -> Tokenizer:
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
            return cls()

        def __call__(self, *_args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            assert kwargs["max_length"] == 256
            return {
                "input_ids": torch.tensor([[1, 2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }

    class Model(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, *_args: object, **kwargs: object) -> Model:
            assert kwargs["use_safetensors"] is True
            return cls()

        def forward(self, **_kwargs: object) -> SimpleNamespace:
            hidden = torch.zeros((1, 3, 384), dtype=torch.float32)
            hidden[0, 0, 0] = 1
            hidden[0, 1, 1] = 1
            hidden[0, 2, 2] = 100
            return SimpleNamespace(last_hidden_state=hidden)

    fake = SimpleNamespace(
        __version__="5.14.1",
        BertTokenizerFast=Tokenizer,
        BertModel=Model,
    )
    monkeypatch.setattr("importlib.import_module", lambda name: fake)
    provider = MiniLML6V2EmbeddingProvider(model_path)
    values = checked_embedding(provider, "test")
    assert len(values) == 384
    assert values[0] == pytest.approx(2**-0.5)
    assert values[1] == pytest.approx(2**-0.5)
    assert values[2] == 0


def test_minilm_provider_rejects_artifact_drift(tmp_path: Path) -> None:
    model_path = pinned_model(tmp_path)
    (model_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        MiniLML6V2EmbeddingProvider(model_path)


def test_checked_embedding_rejects_wrong_dimension() -> None:
    class Broken:
        profile = "broken"
        dimensions = 384

        def embed(self, text: str) -> tuple[float, ...]:
            del text
            return (0.0, 1.0)

    with pytest.raises(ValueError, match="dimensions"):
        checked_embedding(Broken(), "query")


@pytest.mark.skipif(
    not Path("/tmp/opencode/minilm-l6-v2").is_dir(),
    reason="pinned MiniLM fixture is not available",
)
def test_real_minilm_golden_vector_and_semantic_margin() -> None:
    provider = MiniLML6V2EmbeddingProvider(Path("/tmp/opencode/minilm-l6-v2"))
    query = provider.embed("What is the approved maintenance window?")
    passage = provider.embed(
        "Service policy: approved maintenance window is 02:00-04:00 UTC."
    )
    distractor = provider.embed("The ocean is blue.")
    assert hashlib.sha256(struct.pack("<384f", *query)).hexdigest() == (
        "f5a4ad5d449f71d68de1a920a101dc5d72084073876c676475d841bbf11e447b"
    )
    matching = sum(left * right for left, right in zip(query, passage, strict=True))
    unrelated = sum(left * right for left, right in zip(query, distractor, strict=True))
    assert matching > 0.72
    assert unrelated < 0.1
