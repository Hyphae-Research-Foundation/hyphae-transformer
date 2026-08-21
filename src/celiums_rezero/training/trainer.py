"""Small deterministic trainer used by local pilots and Lab runners."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from celiums_rezero.core.diagnostics import assert_finite_model, collect_gate_stats
from celiums_rezero.core.optim import build_optimizer_groups
from celiums_rezero.data.corpus import ContinuousSequenceSource, evaluate_corpus
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
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.evaluation_batch_size) < 1:
            raise ValueError("steps and batch sizes must be positive")
        if min(self.learning_rate, self.gate_learning_rate, self.gradient_clip) <= 0:
            raise ValueError("learning rates and gradient_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


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


@dataclass(frozen=True, slots=True)
class TrainProgress:
    step: int
    loss: float
    tokens: int
    elapsed_seconds: float
    gradient_norm: float
    gate_values: dict[str, float]
    gate_gradients: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class TrainingState:
    completed_steps: int = 0
    tokens: int = 0
    losses: tuple[float, ...] = ()
    elapsed_seconds: float = 0.0


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
    ) -> TrainingState:
        if not self.path.exists():
            return TrainingState()
        payload = torch.load(self.path, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported checkpoint payload")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        source.load_state_dict(payload["source"])
        random.setstate(payload["python_random_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_state = payload.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        return TrainingState(
            completed_steps=int(payload["completed_steps"]),
            tokens=int(payload["tokens"]),
            losses=tuple(float(loss) for loss in payload["losses"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
        )

    def record(
        self,
        progress: TrainProgress,
        model: ReZeroLM,
        optimizer: torch.optim.Optimizer,
        source: SequenceSource,
        losses: list[float],
    ) -> None:
        import json

        self.directory.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as history:
            history.write(json.dumps(asdict(progress), sort_keys=True) + "\n")
        if progress.step % self.every_steps:
            return
        payload = {
            "version": 1,
            "completed_steps": progress.step,
            "tokens": progress.tokens,
            "losses": losses,
            "elapsed_seconds": progress.elapsed_seconds,
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
    state = (
        checkpoint_manager.load(model, optimizer, source, device=device)
        if checkpoint_manager is not None
        else TrainingState()
    )
    if state.completed_steps > config.steps:
        raise ValueError("checkpoint has more completed steps than requested")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses = list(state.losses)
    tokens = state.tokens
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
                ),
                model,
                optimizer,
                source,
                losses,
            )
    training_seconds = state.elapsed_seconds + time.perf_counter() - started
    gate_values = {
        stat.name: stat.value.rms for stat in collect_gate_stats(model)
    }
    validation = (
        evaluate_corpus(
            model,
            validation_tokens,
            batch_size=config.evaluation_batch_size,
            deadline=deadline,
        )
        if validation_tokens is not None
        else None
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
    )
