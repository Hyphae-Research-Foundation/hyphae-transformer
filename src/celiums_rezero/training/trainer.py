"""Small deterministic trainer used by local pilots and Lab runners."""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from celiums_rezero.core.diagnostics import assert_finite_model, collect_gate_stats
from celiums_rezero.core.optim import build_optimizer_groups
from celiums_rezero.data.corpus import (
    ContinuousSequenceSource,
    CorpusEvaluation,
    evaluate_corpus,
)
from celiums_rezero.data.synthetic import SyntheticSequenceSource
from celiums_rezero.transformer.model import ReZeroLM


class SequenceSource(Protocol):
    def batch(
        self, batch_size: int, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainConfig:
    steps: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    gate_learning_rate: float = 3e-3
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    evaluation_batch_size: int = 8
    validation_every_steps: int | None = None
    validation_nll_threshold: float | None = None
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.evaluation_batch_size) < 1:
            raise ValueError("steps and batch sizes must be positive")
        if min(self.learning_rate, self.gate_learning_rate, self.gradient_clip) <= 0:
            raise ValueError("learning rates and gradient_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.validation_every_steps is not None and self.validation_every_steps < 1:
            raise ValueError("validation_every_steps must be positive")
        if self.validation_nll_threshold is not None:
            if not isfinite(self.validation_nll_threshold) or self.validation_nll_threshold <= 0:
                raise ValueError("validation_nll_threshold must be finite and positive")
            if self.validation_every_steps is None:
                raise ValueError("validation_nll_threshold requires validation_every_steps")


@dataclass(frozen=True, slots=True)
class ValidationPoint:
    step: int
    training_tokens: int
    evaluation_tokens: int
    nll: float
    bits_per_token: float


@dataclass(frozen=True, slots=True)
class TrainSummary:
    initial_loss: float
    final_loss: float
    best_loss: float
    tokens: int
    wall_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int
    gate_values: dict[str, float]
    validation_nll: float | None = None
    validation_bits_per_token: float | None = None
    test_nll: float | None = None
    test_bits_per_token: float | None = None
    validation_curve: tuple[ValidationPoint, ...] = ()
    validation_nll_threshold: float | None = None
    tokens_to_threshold: int | None = None


@dataclass(frozen=True, slots=True)
class TrainProgress:
    step: int
    loss: float
    tokens: int
    elapsed_seconds: float
    gradient_norm: float
    gate_values: dict[str, float]
    gate_gradients: dict[str, float | None]
    validation: ValidationPoint | None = None


@dataclass(frozen=True, slots=True)
class TrainingState:
    completed_steps: int = 0
    tokens: int = 0
    losses: tuple[float, ...] = ()
    elapsed_seconds: float = 0.0
    validation_curve: tuple[ValidationPoint, ...] = ()


class CheckpointManager:
    def __init__(self, directory: Path, *, every_steps: int = 1) -> None:
        if every_steps < 1:
            raise ValueError("checkpoint interval must be positive")
        self.directory = directory
        self.every_steps = every_steps
        self.path = directory / "latest.pt"
        self.history_path = directory / "history.jsonl"

    def load(
        self,
        model: ReZeroLM,
        optimizer: torch.optim.Optimizer,
        source: SequenceSource,
        *,
        device: torch.device,
        protocol: dict[str, object] | None = None,
    ) -> TrainingState:
        if not self.path.exists():
            return TrainingState()
        payload = torch.load(self.path, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("version") not in {1, 2}:
            raise ValueError("unsupported checkpoint payload")
        stored_protocol = payload.get("protocol")
        if stored_protocol is not None and protocol is not None and stored_protocol != protocol:
            raise ValueError("checkpoint training protocol does not match the requested run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        source.load_state_dict(payload["source"])
        random.setstate(payload["python_random_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_state = payload.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        raw_curve = payload.get("validation_curve", [])
        if not isinstance(raw_curve, list):
            raise TypeError("checkpoint validation curve must be a list")
        state = TrainingState(
            completed_steps=int(payload["completed_steps"]),
            tokens=int(payload["tokens"]),
            losses=tuple(float(loss) for loss in payload["losses"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            validation_curve=tuple(ValidationPoint(**point) for point in raw_curve),
        )
        self._reconcile_history(state.completed_steps)
        return state

    def record(
        self,
        progress: TrainProgress,
        model: ReZeroLM,
        optimizer: torch.optim.Optimizer,
        source: SequenceSource,
        losses: list[float],
        validation_curve: list[ValidationPoint],
        *,
        force_checkpoint: bool = False,
        protocol: dict[str, object] | None = None,
    ) -> None:
        import json

        self.directory.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as history:
            history.write(json.dumps(asdict(progress), sort_keys=True) + "\n")
            history.flush()
            os.fsync(history.fileno())
        if progress.step % self.every_steps and not force_checkpoint:
            return
        payload = {
            "version": 2,
            "completed_steps": progress.step,
            "tokens": progress.tokens,
            "losses": losses,
            "elapsed_seconds": progress.elapsed_seconds,
            "validation_curve": [asdict(point) for point in validation_curve],
            "protocol": protocol,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "source": source.state_dict(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary = self.path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(self.path)

    def _reconcile_history(self, completed_steps: int) -> None:
        import json

        if not self.history_path.exists():
            return
        records = [json.loads(line) for line in self.history_path.read_text().splitlines()]
        committed = [record for record in records if int(record["step"]) <= completed_steps]
        steps = [int(record["step"]) for record in committed]
        if steps != list(range(1, completed_steps + 1)):
            raise ValueError("training history does not match committed checkpoint steps")
        self.history_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in committed)
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_synthetic(
    model: ReZeroLM,
    config: TrainConfig,
    *,
    checkpoint_manager: CheckpointManager | None = None,
    deadline: float | None = None,
) -> TrainSummary:
    seed_everything(config.seed)
    source = SyntheticSequenceSource(
        vocab_size=model.config.vocab_size,
        sequence_length=model.config.max_sequence_length,
        seed=config.seed,
    )
    return _train_source(
        model,
        source,
        config,
        checkpoint_manager=checkpoint_manager,
        deadline=deadline,
    )


def train_corpus(
    model: ReZeroLM,
    train_tokens: torch.Tensor,
    config: TrainConfig,
    *,
    validation_tokens: torch.Tensor | None = None,
    test_tokens: torch.Tensor | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    deadline: float | None = None,
) -> TrainSummary:
    seed_everything(config.seed)
    source = ContinuousSequenceSource(
        train_tokens,
        sequence_length=model.config.max_sequence_length,
        seed=config.seed,
    )
    return _train_source(
        model,
        source,
        config,
        validation_tokens=validation_tokens,
        test_tokens=test_tokens,
        checkpoint_manager=checkpoint_manager,
        deadline=deadline,
    )


def _train_source(
    model: ReZeroLM,
    source: SequenceSource,
    config: TrainConfig,
    *,
    validation_tokens: torch.Tensor | None = None,
    test_tokens: torch.Tensor | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    deadline: float | None = None,
) -> TrainSummary:
    if config.validation_nll_threshold is not None and validation_tokens is None:
        raise ValueError("validation threshold requires a validation corpus")
    device = torch.device(config.device)
    model.to(device)
    model.train()
    groups = build_optimizer_groups(
        model,
        lr=config.learning_rate,
        gate_lr=config.gate_learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    checkpoint_protocol = _checkpoint_protocol(model, config)
    state = (
        checkpoint_manager.load(
            model,
            optimizer,
            source,
            device=device,
            protocol=checkpoint_protocol,
        )
        if checkpoint_manager is not None
        else TrainingState()
    )
    if state.completed_steps > config.steps:
        raise ValueError("checkpoint has more completed steps than requested")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses = list(state.losses)
    tokens = state.tokens
    validation_curve = list(state.validation_curve)
    started = time.perf_counter()
    for step in range(state.completed_steps + 1, config.steps + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("training wall-time budget exceeded")
        inputs, targets = source.batch(config.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, targets)
        if output.loss is None:
            raise RuntimeError("training forward did not return a loss")
        output.loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        assert_finite_model(model)
        gate_stats = collect_gate_stats(model)
        optimizer.step()
        assert_finite_model(model, include_gradients=False)
        losses.append(float(output.loss.detach()))
        tokens += inputs.numel()
        validation_point = None
        validation_due = (
            validation_tokens is not None
            and config.validation_every_steps is not None
            and (step % config.validation_every_steps == 0 or step == config.steps)
        )
        if validation_due:
            assert validation_tokens is not None
            evaluation = _evaluate_without_rng_side_effects(
                model,
                validation_tokens,
                batch_size=config.evaluation_batch_size,
                deadline=deadline,
            )
            validation_point = ValidationPoint(
                step=step,
                training_tokens=tokens,
                evaluation_tokens=evaluation.tokens,
                nll=evaluation.nll,
                bits_per_token=evaluation.bits_per_token,
            )
            validation_curve.append(validation_point)
        if checkpoint_manager is not None:
            elapsed_seconds = state.elapsed_seconds + time.perf_counter() - started
            checkpoint_manager.record(
                TrainProgress(
                    step=step,
                    loss=losses[-1],
                    tokens=tokens,
                    elapsed_seconds=elapsed_seconds,
                    gradient_norm=float(gradient_norm),
                    gate_values={stat.name: stat.value.rms for stat in gate_stats},
                    gate_gradients={
                        stat.name: None if stat.gradient is None else stat.gradient.rms
                        for stat in gate_stats
                    },
                    validation=validation_point,
                ),
                model,
                optimizer,
                source,
                losses,
                validation_curve,
                force_checkpoint=validation_due or step == config.steps,
                protocol=checkpoint_protocol,
            )
    training_seconds = state.elapsed_seconds + time.perf_counter() - started
    gate_values = {
        stat.name: stat.value.rms for stat in collect_gate_stats(model)
    }
    validation = None
    if validation_tokens is not None:
        validation_corpus = validation_tokens
        if validation_curve and validation_curve[-1].step == config.steps:
            last = validation_curve[-1]
            validation = CorpusEvaluation(last.nll, last.bits_per_token, last.evaluation_tokens)
        else:
            validation = _evaluate_without_rng_side_effects(
                model,
                validation_corpus,
                batch_size=config.evaluation_batch_size,
                deadline=deadline,
            )
    test = (
        evaluate_corpus(
            model,
            test_tokens,
            batch_size=config.evaluation_batch_size,
            deadline=deadline,
        )
        if test_tokens is not None
        else None
    )
    wall_seconds = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return TrainSummary(
        initial_loss=losses[0],
        final_loss=losses[-1],
        best_loss=min(losses),
        tokens=tokens,
        wall_seconds=wall_seconds,
        tokens_per_second=tokens / max(training_seconds, 1e-12),
        peak_memory_bytes=peak_memory,
        gate_values=gate_values,
        validation_nll=None if validation is None else validation.nll,
        validation_bits_per_token=(
            None if validation is None else validation.bits_per_token
        ),
        test_nll=None if test is None else test.nll,
        test_bits_per_token=None if test is None else test.bits_per_token,
        validation_curve=tuple(validation_curve),
        validation_nll_threshold=config.validation_nll_threshold,
        tokens_to_threshold=_tokens_to_threshold(
            validation_curve,
            config.validation_nll_threshold,
        ),
    )


def _evaluate_without_rng_side_effects(
    model: ReZeroLM,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    deadline: float | None,
) -> CorpusEvaluation:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return evaluate_corpus(model, tokens, batch_size=batch_size, deadline=deadline)
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _tokens_to_threshold(
    curve: list[ValidationPoint],
    threshold: float | None,
) -> int | None:
    if threshold is None:
        return None
    return next((point.training_tokens for point in curve if point.nll <= threshold), None)


def _checkpoint_protocol(model: ReZeroLM, config: TrainConfig) -> dict[str, object]:
    return {
        "model": model.config.to_dict(),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "gate_learning_rate": config.gate_learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip": config.gradient_clip,
        "evaluation_batch_size": config.evaluation_batch_size,
        "validation_every_steps": config.validation_every_steps,
        "validation_nll_threshold": config.validation_nll_threshold,
        "seed": config.seed,
    }
