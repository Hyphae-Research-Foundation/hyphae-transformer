"""Typed scientific contracts for hypotheses, budgets, and immutable runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any

from celiums_rezero.lab.serialization import content_hash

RUNNERS = {"synthetic_v1", "continuous_byte_corpus_v1", "continuous_byte_corpus_v2"}
ID_PATTERN = re.compile(r"^[A-Z]-[0-9a-f]{12,16}$")


class RunStage(StrEnum):
    STATIC = "static"
    MINI_PILOT = "mini_pilot"
    PILOT = "pilot"
    FULL_LOCAL = "full_local"
    FULL_CLOUD = "full_cloud"

    @property
    def rank(self) -> int:
        return list(RunStage).index(self)


class RunStatus(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Verdict(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


class MemoryRelation(StrEnum):
    NECESSARY = "necessary"
    HELPS = "helps"
    DOES_NOT_CONTRIBUTE = "does_not_contribute"
    DESTABILIZES = "destabilizes"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Budget:
    max_wall_seconds: float
    max_device_hours: float = 0.0
    max_cost_usd: float = 0.0
    max_failures: int = 1
    max_artifact_bytes: int = 100_000_000

    def __post_init__(self) -> None:
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        if min(self.max_device_hours, self.max_cost_usd) < 0:
            raise ValueError("budget limits cannot be negative")
        if self.max_failures < 0 or self.max_artifact_bytes < 1:
            raise ValueError("invalid failure or artifact budget")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    claim: str
    baseline: str
    candidate: str
    context: dict[str, Any]
    independent_variables: tuple[str, ...]
    dependent_variables: tuple[str, ...]
    prediction: str
    minimum_effect: float
    falsification: tuple[str, ...]
    budget: Budget
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("claim cannot be empty")
        if not self.baseline or not self.candidate:
            raise ValueError("baseline and candidate are required")
        if self.baseline == self.candidate:
            raise ValueError("baseline and candidate must differ")
        if self.minimum_effect < 0:
            raise ValueError("minimum_effect cannot be negative")
        if not self.dependent_variables:
            raise ValueError("at least one dependent variable is required")
        if not self.falsification:
            raise ValueError("at least one falsification criterion is required")
        if self.hypothesis_id is None:
            values = {
                "claim": self.claim,
                "baseline": self.baseline,
                "candidate": self.candidate,
                "context": self.context,
                "independent_variables": self.independent_variables,
                "dependent_variables": self.dependent_variables,
                "prediction": self.prediction,
                "minimum_effect": self.minimum_effect,
                "falsification": self.falsification,
                "budget": self.budget,
            }
            object.__setattr__(self, "hypothesis_id", f"H-{content_hash(values, length=12)}")


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    value: float
    unit: str = ""
    direction: str = "neutral"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name cannot be empty")
        if not isfinite(self.value):
            raise ValueError("metric value must be finite")
        if self.direction not in {"minimize", "maximize", "neutral"}:
            raise ValueError("metric direction is invalid")


@dataclass(frozen=True, slots=True)
class MetricPoint:
    step: int
    training_tokens: int
    value: float

    def __post_init__(self) -> None:
        if min(self.step, self.training_tokens) < 0 or not isfinite(self.value):
            raise ValueError("metric curve point is invalid")


@dataclass(frozen=True, slots=True)
class MetricCurve:
    name: str
    unit: str
    direction: str
    points: tuple[MetricPoint, ...]

    def __post_init__(self) -> None:
        if not self.name or self.direction not in {"minimize", "maximize", "neutral"}:
            raise ValueError("metric curve metadata is invalid")
        coordinates = [(point.step, point.training_tokens) for point in self.points]
        if coordinates != sorted(coordinates) or len(set(coordinates)) != len(coordinates):
            raise ValueError("metric curve coordinates must be strictly increasing")


@dataclass(frozen=True, slots=True)
class RunManifest:
    hypothesis_id: str
    stage: RunStage
    seed: int
    config: dict[str, Any]
    budget: Budget
    code_revision: str = "uncommitted"
    data_revision: str = "synthetic"
    status: RunStatus = RunStatus.REGISTERED
    parent_run_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = 1
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported manifest schema version")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if not ID_PATTERN.fullmatch(self.hypothesis_id):
            raise ValueError("hypothesis_id is invalid")
        runner = self.config.get("runner")
        if runner not in RUNNERS:
            raise ValueError(f"unknown manifest runner: {runner}")
        training = self.config.get("training")
        if not isinstance(training, dict) or training.get("seed") != self.seed:
            raise ValueError("manifest and training seeds must match")
        computed_run_id = self.computed_run_id()
        if self.run_id is None:
            object.__setattr__(self, "run_id", computed_run_id)
        elif self.run_id != computed_run_id:
            raise ValueError("run_id does not match manifest contents")

    def computed_run_id(self) -> str:
        identity = {
            "hypothesis_id": self.hypothesis_id,
            "stage": self.stage,
            "seed": self.seed,
            "config": self.config,
            "code_revision": self.code_revision,
            "data_revision": self.data_revision,
            "parent_run_id": self.parent_run_id,
        }
        return f"R-{content_hash(identity)}"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RunManifest:
        expected = {
            "hypothesis_id",
            "stage",
            "seed",
            "config",
            "budget",
            "code_revision",
            "data_revision",
            "status",
            "parent_run_id",
            "created_at",
            "run_id",
            "schema_version",
        }
        unknown = set(values) - expected
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        required = {"hypothesis_id", "stage", "seed", "config", "budget"}
        missing = required - set(values)
        if missing:
            raise ValueError(f"missing manifest fields: {sorted(missing)}")
        budget = values["budget"]
        config = values["config"]
        if not isinstance(budget, dict) or not isinstance(config, dict):
            raise TypeError("manifest config and budget must be dictionaries")
        return cls(
            hypothesis_id=str(values["hypothesis_id"]),
            stage=RunStage(str(values["stage"])),
            seed=int(values["seed"]),
            config=config,
            budget=Budget(**budget),
            code_revision=str(values.get("code_revision", "uncommitted")),
            data_revision=str(values.get("data_revision", "synthetic")),
            status=RunStatus(str(values.get("status", RunStatus.REGISTERED.value))),
            parent_run_id=(
                None if values.get("parent_run_id") is None else str(values["parent_run_id"])
            ),
            created_at=str(values.get("created_at", datetime.now(UTC).isoformat())),
            run_id=None if values.get("run_id") is None else str(values["run_id"]),
            schema_version=int(values.get("schema_version", 1)),
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: RunStatus
    metrics: tuple[Metric, ...]
    verdict: Verdict
    summary: str
    started_at: str
    finished_at: str
    failure: str | None = None
    artifacts: tuple[str, ...] = ()
    curves: tuple[MetricCurve, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    statement: str
    relation: MemoryRelation
    conditions: dict[str, Any]
    evidence_run_ids: tuple[str, ...]
    confidence: float
    entry_id: str | None = None

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("memory statement cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not self.evidence_run_ids:
            raise ValueError("memory requires evidence")
        if self.entry_id is None:
            object.__setattr__(
                self,
                "entry_id",
                f"M-{content_hash({'statement': self.statement, 'conditions': self.conditions})}",
            )
