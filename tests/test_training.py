from __future__ import annotations

import json
from pathlib import Path

import torch

from celiums_rezero.lab.registry import Registry
from celiums_rezero.lab.runner import (
    run_manifest,
    run_registered_corpus,
    run_registered_synthetic,
    staged_config,
    staged_corpus_config,
)
from celiums_rezero.lab.schemas import Budget, Hypothesis, RunManifest, RunStage, RunStatus
from celiums_rezero.lab.serialization import to_primitive
from celiums_rezero.training.trainer import (
    CheckpointManager,
    TrainConfig,
    train_corpus,
    train_synthetic,
)
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.model import ReZeroLM


def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_sequence_length=8,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        residual_strategy=ResidualStrategy.CRZ_RMS,
    )


def test_synthetic_training_produces_finite_summary() -> None:
    summary = train_synthetic(
        ReZeroLM(model_config()),
        TrainConfig(steps=3, batch_size=2, learning_rate=1e-3, device="cpu"),
    )
    assert summary.tokens == 48
    assert summary.wall_seconds > 0
    assert summary.tokens_per_second > 0
    assert summary.gate_values


def test_corpus_training_reports_validation_and_test_metrics() -> None:
    tokens = torch.arange(128, dtype=torch.long) % model_config().vocab_size
    torch.manual_seed(11)
    first_model = ReZeroLM(model_config())
    second_model = ReZeroLM(model_config())
    second_model.load_state_dict(first_model.state_dict())
    config = TrainConfig(
        steps=2,
        batch_size=2,
        evaluation_batch_size=2,
        seed=5,
        device="cpu",
    )
    summary = train_corpus(
        first_model,
        tokens,
        config,
        validation_tokens=tokens[:25],
        test_tokens=tokens[25:50],
    )
    repeated = train_corpus(
        second_model,
        tokens,
        config,
        validation_tokens=tokens[:25],
        test_tokens=tokens[25:50],
    )
    assert summary.tokens == 32
    assert summary.initial_loss == repeated.initial_loss
    assert summary.final_loss == repeated.final_loss
    assert summary.validation_nll == repeated.validation_nll
    assert summary.validation_nll is not None
    assert summary.validation_bits_per_token is not None
    assert summary.test_nll is not None
    assert summary.test_bits_per_token is not None


def test_lab_runner_writes_complete_evidence(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    budget = Budget(max_wall_seconds=60, max_failures=0)
    item = Hypothesis(
        claim="Synthetic run remains finite.",
        baseline="pre_rms",
        candidate="crz_rms",
        context={"dataset": "synthetic"},
        independent_variables=("strategy",),
        dependent_variables=("loss",),
        prediction="finite",
        minimum_effect=0,
        falsification=("Non-finite state.",),
        budget=budget,
    )
    registry.register_hypothesis(item)
    assert item.hypothesis_id is not None
    training = TrainConfig(steps=2, batch_size=2, seed=9, device="cpu")
    manifest = RunManifest(
        hypothesis_id=item.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=9,
        config=staged_config(model=model_config(), training=training),
        budget=budget,
    )
    result = run_registered_synthetic(registry, manifest)
    repeated = run_registered_synthetic(registry, manifest)
    assert result.status is RunStatus.COMPLETED
    assert repeated == result
    assert manifest.run_id is not None
    run_path = registry.runs / manifest.run_id
    assert (run_path / "manifest.json").exists()
    assert (run_path / "result.json").exists()
    assert (run_path / "report.html").exists()
    assert (run_path / "checkpoints" / "latest.pt").exists()
    assert (run_path / "checkpoints" / "history.jsonl").exists()


def test_lab_runner_records_corpus_metrics(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    paths = [data / name for name in ("train.txt", "valid.txt", "test.txt")]
    for path in paths:
        path.write_bytes(bytes(range(61)))

    registry = Registry(tmp_path / "registry")
    budget = Budget(max_wall_seconds=60, max_failures=0)
    item = Hypothesis(
        claim="Corpus run remains finite.",
        baseline="pre_rms",
        candidate="crz_rms",
        context={"dataset": "fixture"},
        independent_variables=("strategy",),
        dependent_variables=("validation_nll",),
        prediction="finite",
        minimum_effect=0,
        falsification=("Non-finite state.",),
        budget=budget,
    )
    registry.register_hypothesis(item)
    assert item.hypothesis_id is not None
    training = TrainConfig(steps=1, batch_size=2, evaluation_batch_size=2, device="cpu")
    manifest = RunManifest(
        hypothesis_id=item.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=training.seed,
        config=staged_corpus_config(
            model=model_config(),
            training=training,
            train_path=paths[0],
            validation_path=paths[1],
            test_path=paths[2],
        ),
        budget=budget,
        data_revision="fixture-v1",
    )
    result = run_registered_corpus(registry, manifest)
    metric_names = {metric.name for metric in result.metrics}
    assert result.status is RunStatus.COMPLETED
    assert {"validation_nll", "test_nll"} <= metric_names


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    config = model_config()
    torch.manual_seed(17)
    uninterrupted = ReZeroLM(config)
    interrupted = ReZeroLM(config)
    interrupted.load_state_dict(uninterrupted.state_dict())
    full = TrainConfig(steps=4, batch_size=2, seed=11, device="cpu")
    expected = train_synthetic(uninterrupted, full)

    manager = CheckpointManager(tmp_path / "checkpoint", every_steps=1)
    train_synthetic(
        interrupted,
        TrainConfig(steps=2, batch_size=2, seed=11, device="cpu"),
        checkpoint_manager=manager,
    )
    resumed = train_synthetic(interrupted, full, checkpoint_manager=manager)
    assert resumed.initial_loss == expected.initial_loss
    assert resumed.final_loss == expected.final_loss
    assert resumed.tokens == expected.tokens
    for expected_parameter, resumed_parameter in zip(
        uninterrupted.parameters(), interrupted.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed_parameter, expected_parameter)
    assert manager.history_path.read_text().count("\n") == 4


def test_serialized_manifest_dispatches_to_allowlisted_runner(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    budget = Budget(max_wall_seconds=60, max_failures=0)
    item = Hypothesis(
        claim="Serialized synthetic run remains finite.",
        baseline="pre_rms",
        candidate="crz_rms",
        context={"dataset": "synthetic"},
        independent_variables=("strategy",),
        dependent_variables=("loss",),
        prediction="finite",
        minimum_effect=0,
        falsification=("Non-finite state.",),
        budget=budget,
    )
    registry.register_hypothesis(item)
    assert item.hypothesis_id is not None
    training = TrainConfig(steps=1, batch_size=1, seed=3, device="cpu")
    original = RunManifest(
        hypothesis_id=item.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=3,
        config=staged_config(model=model_config(), training=training),
        budget=budget,
    )
    encoded = json.loads(json.dumps(to_primitive(original)))
    restored = RunManifest.from_dict(encoded)
    result = run_manifest(registry, restored)
    assert result.status is RunStatus.COMPLETED


def test_runner_stops_when_wall_budget_expires(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    budget = Budget(max_wall_seconds=1e-9, max_failures=0)
    item = Hypothesis(
        claim="Expired run stops.",
        baseline="pre_rms",
        candidate="crz_rms",
        context={"dataset": "synthetic"},
        independent_variables=("strategy",),
        dependent_variables=("status",),
        prediction="stopped",
        minimum_effect=0,
        falsification=("Run exceeds budget.",),
        budget=budget,
    )
    registry.register_hypothesis(item)
    assert item.hypothesis_id is not None
    training = TrainConfig(steps=2, batch_size=1, seed=3, device="cpu")
    manifest = RunManifest(
        hypothesis_id=item.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=3,
        config=staged_config(model=model_config(), training=training),
        budget=budget,
    )
    result = run_manifest(registry, manifest)
    assert result.status is RunStatus.STOPPED
    assert result.failure is not None and "TimeoutError" in result.failure
