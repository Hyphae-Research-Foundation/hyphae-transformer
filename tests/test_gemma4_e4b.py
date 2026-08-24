from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from celiums_rezero.governed.data import load_governed_dataset
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.schemas import (
    DatasetSplit,
    GovernedDatasetManifest,
)
from celiums_rezero.knowledge.schemas import SufficiencyPolicy

ROOT = Path(__file__).resolve().parents[1]
mars_converter = importlib.util.module_from_spec(
    spec := importlib.util.spec_from_file_location(
        "mars_converter", ROOT / "scripts" / "build_mars_governed_dataset.py"
    )
)
assert spec.loader is not None
spec.loader.exec_module(mars_converter)
gemma_preflight = importlib.util.module_from_spec(
    spec := importlib.util.spec_from_file_location(
        "gemma_preflight", ROOT / "scripts" / "preflight_gemma4_e4b.py"
    )
)
assert spec.loader is not None
spec.loader.exec_module(gemma_preflight)


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *_args: object, **kwargs: object) -> FakeTokenizer:
        assert kwargs["local_files_only"] is True
        assert kwargs["padding_side"] == "left"
        return cls()

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[0, 4, 5], [6, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }


class FakeGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @classmethod
    def from_pretrained(cls, *_args: object, **kwargs: object) -> FakeGemma:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        return cls()

    def forward(self, **_kwargs: object) -> SimpleNamespace:
        hidden = torch.tensor(
            [
                [[0.0] * 2560, [1.0] * 2560, [2.0] * 2560],
                [[10.0] * 2560, [11.0] * 2560, [12.0] * 2560],
            ]
        )
        return SimpleNamespace(hidden_states=tuple(hidden for _ in range(43)))


def test_gemma_backbone_pools_last_unmasked_token_with_left_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "config.json"
    artifact.write_bytes(b"pinned")
    monkeypatch.setattr(
        Gemma4E4BFrozenBackbone,
        "required_artifacts",
        {
            "config.json": (
                len(b"pinned"),
                hashlib.sha256(b"pinned").hexdigest(),
            )
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="5.14.1",
            AutoModelForMultimodalLM=FakeGemma,
            AutoTokenizer=FakeTokenizer,
        ),
    )
    backbone = Gemma4E4BFrozenBackbone(tmp_path, device="cpu")
    features = backbone.encode(("first", "second"), device=torch.device("cpu"))
    assert features.shape == (2, 2560)
    assert features[:, 0].tolist() == [2.0, 12.0]
    assert not features.requires_grad
    head = nn.Linear(2560, 1)
    head(features).sum().backward()
    assert head.weight.grad is not None
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


def test_mars_converter_is_reproducible_and_partitions_complete_worlds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    dev, evaluation_dev = _mars_records(40, prefix="dev")
    test, evaluation_test = _mars_records(20, prefix="test")
    files = {
        "dev.jsonl": _jsonl(dev),
        "evaluation-dev.jsonl": _jsonl(evaluation_dev),
        "evaluation-test.jsonl": _jsonl(evaluation_test),
        "manifest.json": b"{}\n",
        "test.jsonl": _jsonl(test),
        "validation-report.json": b"{}\n",
    }
    for name, content in files.items():
        (release / name).write_bytes(content)
    monkeypatch.setattr(
        mars_converter,
        "PINNED_FILES",
        {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    )
    first = mars_converter.build_dataset(release, tmp_path / "first")
    second = mars_converter.build_dataset(release, tmp_path / "second")
    assert first == second
    assert str(first["dataset_id"]).startswith("gtd_")
    assert len(str(first["dataset_id"])) == 68
    assert {
        name: item["records"] for name, item in first["splits"].items()
    } == {"train": 380, "validation": 190, "test": 380, "adversarial": 190}
    assert all(
        (tmp_path / "first" / name).read_bytes()
        == (tmp_path / "second" / name).read_bytes()
        for name in (
            "manifest.json",
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "adversarial.jsonl",
        )
    )
    manifest = GovernedDatasetManifest(
        splits=tuple(
            (name, DatasetSplit(**item))
            for name, item in sorted(first["splits"].items())
        ),
        policy=SufficiencyPolicy(**first["policy"]),
        maximum_evidence_items=int(first["maximum_evidence_items"]),
        dataset_id=str(first["dataset_id"]),
    )
    loaded = load_governed_dataset(tmp_path / "first", manifest)
    assert len(loaded.train) == 380
    assert len({item.provenance.rsplit(":", 1)[-1] for item in loaded.train}) == 20

    (release / "dev.jsonl").write_bytes(files["dev.jsonl"] + b"{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        mars_converter.build_dataset(release, tmp_path / "tampered")


def test_preflight_verifies_pinned_artifact_size_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights")
    monkeypatch.setattr(
        gemma_preflight,
        "REQUIRED_ARTIFACTS",
        {
            "model.safetensors": (
                len(b"weights"),
                hashlib.sha256(b"weights").hexdigest(),
            )
        },
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.14.1"))
    result = gemma_preflight.run_preflight(tmp_path, require_gpu=False)
    assert result["ready"] is True
    artifact.write_bytes(b"tampered")
    result = gemma_preflight.run_preflight(tmp_path, require_gpu=False)
    assert result["ready"] is False
    assert "artifact_mismatch:model.safetensors" in result["blockers"]


def _mars_records(
    worlds: int, *, prefix: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    canonical = []
    evaluation = []
    for world in range(worlds):
        world_id = f"world-{prefix}-{world:02d}"
        for index in range(19):
            episode_id = f"episode-{prefix}-{world:02d}-{index:02d}"
            source_id = f"source-{episode_id}"
            text = f"Evidence for {episode_id}"
            mode = index % 3
            canonical.append(
                {
                    "record_id": episode_id,
                    "world_id": world_id,
                    "claims": (
                        [{"status": "supported", "support": [f"{source_id}#span"]}]
                        if mode == 0
                        else []
                    ),
                    "expected_outcome": {
                        "requires_abstention": mode == 2,
                        "action_state": "none",
                    },
                }
            )
            evaluation.append(
                {
                    "episode_id": episode_id,
                    "instruction": f"Question for {episode_id}",
                    "expected_outcome": {},
                    "backend": {
                        "evidence": (
                            [
                                {
                                    "authority_type": "hyphae",
                                    "content": text,
                                    "digest": {
                                        "algorithm": "sha256",
                                        "value": hashlib.sha256(text.encode()).hexdigest(),
                                    },
                                    "evidence_id": source_id,
                                    "model_visible": True,
                                    "revision": hashlib.sha256(source_id.encode()).hexdigest(),
                                }
                            ]
                            if mode == 0
                            else []
                        )
                    },
                }
            )
    return canonical, evaluation


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in records
    )
