"""Allowlisted local runner joining Core measurements to immutable Lab evidence."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torch import Tensor

from celiums_rezero.data.bytes import load_byte_corpus
from celiums_rezero.data.prepare import verify_file
from celiums_rezero.lab.budgets import BudgetTracker
from celiums_rezero.lab.registry import Registry
from celiums_rezero.lab.report import render_run_report
from celiums_rezero.lab.schemas import (
    Metric,
    MetricCurve,
    MetricPoint,
    RunManifest,
    RunResult,
    RunStatus,
    Verdict,
)
from celiums_rezero.lab.serialization import read_json
from celiums_rezero.training.trainer import (
    CheckpointManager,
    TrainConfig,
    TrainSummary,
    seed_everything,
    train_corpus,
    train_synthetic,
)
from celiums_rezero.transformer.config import ModelConfig
from celiums_rezero.transformer.model import ReZeroLM


def run_registered_synthetic(registry: Registry, manifest: RunManifest) -> RunResult:
    if manifest.config.get("runner") != "synthetic_v1":
        raise ValueError("synthetic runner requires a synthetic_v1 manifest")
    return _run_registered(registry, manifest, corpus=False)


def run_registered_corpus(
    registry: Registry,
    manifest: RunManifest,
    *,
    data_root: Path | None = None,
) -> RunResult:
    if manifest.config.get("runner") not in {
        "continuous_byte_corpus_v1",
        "continuous_byte_corpus_v2",
    }:
        raise ValueError("corpus runner requires a continuous byte corpus manifest")
    return _run_registered(registry, manifest, corpus=True, data_root=data_root)


def run_manifest(
    registry: Registry,
    manifest: RunManifest,
    *,
    data_root: Path | None = None,
) -> RunResult:
    runner = manifest.config.get("runner")
    if runner == "governed_control_v1":
        raise ValueError(
            "governed control manifests require run_registered_governed with a data root"
        )
    if runner == "synthetic_v1":
        if data_root is not None:
            raise ValueError("synthetic manifests do not use a data root")
        return run_registered_synthetic(registry, manifest)
    if runner in {"continuous_byte_corpus_v1", "continuous_byte_corpus_v2"}:
        return run_registered_corpus(registry, manifest, data_root=data_root)
    if not isinstance(runner, str):
        raise ValueError(f"unknown manifest runner: {runner}")
    raise ValueError(f"unknown manifest runner: {runner}")


def _run_registered(
    registry: Registry,
    manifest: RunManifest,
    *,
    corpus: bool,
    data_root: Path | None = None,
) -> RunResult:
    assert manifest.run_id is not None
    registry.register_run(manifest)
    existing = registry.run_result(manifest.run_id)
    if existing is not None:
        return _result_from_dict(existing)
    resolved_data_root = _resolve_data_root(manifest, data_root) if corpus else None
    checkpoint_manager = CheckpointManager(
        registry.runs / manifest.run_id / "checkpoints",
        every_steps=max(1, int(manifest.config.get("checkpoint_every_steps", 1))),
    )
    tracker = BudgetTracker(manifest.budget)
    deadline = time.perf_counter() + manifest.budget.max_wall_seconds
    started = datetime.now(UTC).isoformat()
    try:
        model_values = manifest.config["model"]
        train_values = manifest.config["training"]
        if not isinstance(model_values, dict) or not isinstance(train_values, dict):
            raise TypeError("manifest model and training configs must be dictionaries")
        training = TrainConfig(**train_values)
        seed_everything(training.seed)
        model = ReZeroLM(ModelConfig.from_dict(model_values))
        summary = (
            _train_manifest_corpus(
                model,
                training,
                manifest.config,
                data_root=resolved_data_root,
                checkpoint_manager=checkpoint_manager,
                deadline=deadline,
            )
            if corpus
            else train_synthetic(
                model,
                training,
                checkpoint_manager=checkpoint_manager,
                deadline=deadline,
            )
        )
        tracker.update(
            wall_seconds=summary.wall_seconds,
            device_hours=summary.wall_seconds / 3600,
        )
        parameters = model.parameter_count()
        artifact_paths = (
            checkpoint_manager.path,
            checkpoint_manager.history_path,
        )
        artifact_bytes = sum(path.stat().st_size for path in artifact_paths if path.exists())
        tracker.update(artifact_bytes=artifact_bytes)
        metrics = [
            Metric("initial_loss", summary.initial_loss, "nats", "minimize"),
            Metric("final_loss", summary.final_loss, "nats", "minimize"),
            Metric("best_loss", summary.best_loss, "nats", "minimize"),
            Metric("training_tokens", float(summary.tokens), "tokens"),
            Metric("parameters", float(parameters), "parameters"),
            Metric("wall_seconds", summary.wall_seconds, "seconds", "minimize"),
            Metric("device_hours", summary.wall_seconds / 3600, "hours", "minimize"),
            Metric(
                "estimated_training_flops",
                float(6 * parameters * summary.tokens),
                "FLOPs",
                "minimize",
            ),
            Metric("tokens_per_second", summary.tokens_per_second, "tokens/s", "maximize"),
            Metric("peak_memory_bytes", float(summary.peak_memory_bytes), "bytes", "minimize"),
        ]
        for name in (
            "validation_nll",
            "validation_bits_per_token",
            "test_nll",
            "test_bits_per_token",
        ):
            value = getattr(summary, name)
            if value is not None:
                unit = "bits/token" if "bits" in name else "nats"
                metrics.append(Metric(name, value, unit, "minimize"))
        if summary.validation_nll_threshold is not None:
            metrics.append(
                Metric(
                    "validation_nll_threshold",
                    summary.validation_nll_threshold,
                    "nats",
                    "minimize",
                )
            )
            metrics.append(
                Metric(
                    "validation_threshold_reached",
                    1.0 if summary.tokens_to_threshold is not None else 0.0,
                    "boolean",
                    "maximize",
                )
            )
            if summary.tokens_to_threshold is not None:
                metrics.append(
                    Metric(
                        "tokens_to_threshold",
                        float(summary.tokens_to_threshold),
                        "tokens",
                        "minimize",
                    )
                )
        gate_summary = ", ".join(
            f"{name}={value:.4g}" for name, value in sorted(summary.gate_values.items())
        )
        result = RunResult(
            run_id=manifest.run_id,
            status=RunStatus.COMPLETED,
            metrics=tuple(metrics),
            verdict=Verdict.INCONCLUSIVE,
            summary=(
                f"{'Corpus' if corpus else 'Synthetic'} stage completed. "
                f"Gates: {gate_summary or 'not applicable'}."
            ),
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            artifacts=tuple(
                str(path.relative_to(registry.runs / manifest.run_id))
                for path in artifact_paths
                if path.exists()
            ),
            curves=(
                MetricCurve(
                    name="validation_nll",
                    unit="nats",
                    direction="minimize",
                    points=tuple(
                        MetricPoint(point.step, point.training_tokens, point.nll)
                        for point in summary.validation_curve
                    ),
                ),
            )
            if summary.validation_curve
            else (),
        )
    except TimeoutError as error:
        result = RunResult(
            run_id=manifest.run_id,
            status=RunStatus.STOPPED,
            metrics=(),
            verdict=Verdict.INVALID,
            summary="The staged run stopped at its wall-time budget.",
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            failure=f"{type(error).__name__}: {error}",
        )
    except Exception as error:
        result = RunResult(
            run_id=manifest.run_id,
            status=RunStatus.FAILED,
            metrics=(),
            verdict=Verdict.INVALID,
            summary="The staged run failed and produced no scientific verdict.",
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            failure=f"{type(error).__name__}: {error}",
        )
    registry.complete_run(result)
    run_directory = registry.runs / manifest.run_id
    render_run_report(
        run_directory,
        manifest=read_json(run_directory / "manifest.json"),
        result=read_json(run_directory / "result.json"),
    )
    return result


def _result_from_dict(values: dict[str, object]) -> RunResult:
    raw_metrics = values.get("metrics")
    if not isinstance(raw_metrics, list):
        raise TypeError("stored run metrics must be a list")
    metrics: list[Metric] = []
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            raise TypeError("stored run metric must be a dictionary")
        metrics.append(
            Metric(
                name=str(raw_metric["name"]),
                value=float(raw_metric["value"]),
                unit=str(raw_metric.get("unit", "")),
                direction=str(raw_metric.get("direction", "neutral")),
            )
        )
    artifacts = values.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise TypeError("stored run artifacts must be a list")
    failure = values.get("failure")
    raw_curves = values.get("curves", [])
    if not isinstance(raw_curves, list):
        raise TypeError("stored run curves must be a list")
    curves = tuple(
        MetricCurve(
            name=str(curve["name"]),
            unit=str(curve.get("unit", "")),
            direction=str(curve.get("direction", "neutral")),
            points=tuple(
                MetricPoint(
                    step=int(point["step"]),
                    training_tokens=int(point["training_tokens"]),
                    value=float(point["value"]),
                )
                for point in curve.get("points", [])
            ),
        )
        for curve in raw_curves
        if isinstance(curve, dict)
    )
    return RunResult(
        run_id=str(values["run_id"]),
        status=RunStatus(str(values["status"])),
        metrics=tuple(metrics),
        verdict=Verdict(str(values["verdict"])),
        summary=str(values["summary"]),
        started_at=str(values["started_at"]),
        finished_at=str(values["finished_at"]),
        failure=None if failure is None else str(failure),
        artifacts=tuple(str(artifact) for artifact in artifacts),
        curves=curves,
    )


def _train_manifest_corpus(
    model: ReZeroLM,
    training: TrainConfig,
    config: dict[str, Any],
    *,
    data_root: Path | None,
    checkpoint_manager: CheckpointManager,
    deadline: float,
) -> TrainSummary:
    data = config.get("data")
    if not isinstance(data, dict):
        raise TypeError("manifest data config must be a dictionary")
    paths = data.get("paths")
    limits = data.get("limits", {})
    starts = data.get("starts", {})
    hashes = data.get("sha256", {})
    byte_offset = data.get("byte_offset", 3)
    if not isinstance(paths, dict):
        raise TypeError("manifest data paths must be a dictionary")
    if not all(isinstance(item, dict) for item in (limits, starts, hashes)):
        raise TypeError("manifest data paths, ranges, and hashes must be dictionaries")
    if not isinstance(byte_offset, int):
        raise TypeError("manifest byte_offset must be an integer")

    def load_split(name: str) -> Tensor:
        path = paths.get(name)
        limit = limits.get(name)
        start = starts.get(name, 0)
        checksum = hashes.get(name)
        if not isinstance(path, str):
            raise TypeError(f"manifest data path for {name} must be a string")
        if limit is not None and not isinstance(limit, int):
            raise TypeError(f"manifest data limit for {name} must be an integer")
        if not isinstance(start, int) or not isinstance(checksum, str):
            raise TypeError(f"manifest data start and checksum for {name} are required")
        resolved_path = _resolve_data_path(path, data_root)
        if limit is None:
            limit = resolved_path.stat().st_size - start
        if start < 0 or limit < 2:
            raise ValueError(f"manifest data range for {name} is invalid")
        if start + limit > resolved_path.stat().st_size:
            raise ValueError(f"manifest data range for {name} exceeds the file")
        verify_file(resolved_path, checksum)
        return load_byte_corpus(
            resolved_path,
            start=start,
            limit=limit,
            byte_offset=byte_offset,
        )

    return train_corpus(
        model,
        load_split("train"),
        training,
        validation_tokens=load_split("validation"),
        test_tokens=load_split("test"),
        checkpoint_manager=checkpoint_manager,
        deadline=deadline,
    )


def _resolve_data_root(manifest: RunManifest, data_root: Path | None) -> Path | None:
    if manifest.config.get("runner") == "continuous_byte_corpus_v1":
        return None
    if data_root is None:
        raise ValueError("continuous_byte_corpus_v2 requires an execution-time data root")
    return data_root.resolve(strict=True)


def _resolve_data_path(locator: str, data_root: Path | None) -> Path:
    if data_root is None:
        return Path(locator)
    if "\\" in locator:
        raise ValueError(f"data locator is not portable: {locator}")
    relative = Path(locator)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        raise ValueError(f"data locator is unsafe: {locator}")
    resolved = (data_root / relative).resolve(strict=True)
    if not resolved.is_relative_to(data_root) or not resolved.is_file():
        raise ValueError(f"data locator escapes the bound root: {locator}")
    return resolved


def staged_config(
    *,
    model: ModelConfig,
    training: TrainConfig,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "runner": "synthetic_v1",
        "model": model.to_dict(),
        "training": {
            "steps": training.steps,
            "batch_size": training.batch_size,
            "learning_rate": training.learning_rate,
            "gate_learning_rate": training.gate_learning_rate,
            "weight_decay": training.weight_decay,
            "gradient_clip": training.gradient_clip,
            "evaluation_batch_size": training.evaluation_batch_size,
            "seed": training.seed,
            "device": training.device,
        },
    }
    if training.validation_every_steps is not None:
        config["training"]["validation_every_steps"] = training.validation_every_steps
    if training.validation_nll_threshold is not None:
        config["training"]["validation_nll_threshold"] = training.validation_nll_threshold
    return config


def staged_corpus_config(
    *,
    model: ModelConfig,
    training: TrainConfig,
    data_root: Path,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    train_start: int = 0,
    validation_start: int = 0,
    test_start: int = 0,
    byte_offset: int = 3,
) -> dict[str, Any]:
    from celiums_rezero.data.prepare import file_sha256

    config = staged_config(model=model, training=training)
    root = data_root.resolve(strict=True)

    def locator(path: Path) -> str:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise ValueError(f"corpus path is outside data_root: {path}")
        return resolved.relative_to(root).as_posix()

    config["runner"] = "continuous_byte_corpus_v2"
    config["data"] = {
        "tokenizer": "byte_v1",
        "byte_offset": byte_offset,
        "paths": {
            "train": locator(train_path),
            "validation": locator(validation_path),
            "test": locator(test_path),
        },
        "limits": {
            "train": train_limit,
            "validation": evaluation_limit,
            "test": evaluation_limit,
        },
        "starts": {
            "train": train_start,
            "validation": validation_start,
            "test": test_start,
        },
        "sha256": {
            "train": file_sha256(train_path),
            "validation": file_sha256(validation_path),
            "test": file_sha256(test_path),
        },
    }
    return config
