"""Strict contracts for governed control-head trajectories and evaluation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

from celiums_rezero.knowledge.schemas import EvidenceHit, SufficiencyDecision, SufficiencyPolicy
from celiums_rezero.lab.serialization import content_hash

_ID = re.compile(r"^(?:step|trajectory|scenario)_[0-9a-f]{16}$")


class ControlAction(StrEnum):
    ANSWER = "answer"
    REQUEST_EVIDENCE = "request_evidence"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class ControlTarget:
    action: ControlAction
    evidence_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.evidence_handles) != len(set(self.evidence_handles)):
            raise ValueError("target evidence handles must be unique")
        if self.action is ControlAction.ANSWER and not self.evidence_handles:
            raise ValueError("answer target requires evidence handles")
        if self.action is not ControlAction.ANSWER and self.evidence_handles:
            raise ValueError("non-answer target cannot carry evidence handles")


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    trajectory_id: str
    scenario_id: str
    step_index: int
    query: str
    generation_id: str
    evidence: tuple[EvidenceHit, ...]
    approximate: bool
    conflicting: bool
    blocked: bool
    target: ControlTarget
    provenance: str
    record_id: str | None = None

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.trajectory_id) or not _ID.fullmatch(self.scenario_id):
            raise ValueError("trajectory or scenario ID is invalid")
        if self.step_index < 0 or not self.query.strip() or not self.generation_id:
            raise ValueError("trajectory index, query, and generation are required")
        handles = [hit.handle for hit in self.evidence]
        if len(handles) != len(set(handles)):
            raise ValueError("trajectory evidence handles must be unique")
        if any(
            hit.content_digest != hashlib.sha256(hit.text.encode()).hexdigest()
            for hit in self.evidence
        ):
            raise ValueError("trajectory evidence digest does not match its text")
        selected = set(self.target.evidence_handles)
        if not selected <= set(handles):
            raise ValueError("target handle is absent from trajectory evidence")
        by_handle = {hit.handle: hit for hit in self.evidence}
        if any(
            not by_handle[handle].trusted or not by_handle[handle].active
            for handle in selected
        ):
            raise ValueError("target selected inactive or untrusted evidence")
        if not self.provenance:
            raise ValueError("trajectory provenance is required")
        identity = {
            "schema": "governed-trajectory-step-v1",
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "step_index": self.step_index,
            "query": self.query,
            "generation_id": self.generation_id,
            "evidence": self.evidence,
            "approximate": self.approximate,
            "conflicting": self.conflicting,
            "blocked": self.blocked,
            "target": self.target,
            "provenance": self.provenance,
        }
        expected = f"step_{content_hash(identity)}"
        if self.record_id is None:
            object.__setattr__(self, "record_id", expected)
        elif self.record_id != expected:
            raise ValueError("trajectory record ID does not match its contents")

    def validate_policy(self, policy: SufficiencyPolicy) -> None:
        from celiums_rezero.knowledge.schemas import EvidenceBundle, TenantId

        bundle = EvidenceBundle(
            tenant=TenantId("training_fixture"),
            query_digest=hashlib.sha256(self.query.encode()).hexdigest(),
            corpus_generation=self.generation_id,
            hits=self.evidence,
            approximate=self.approximate,
            conflicting=self.conflicting,
            blocked=self.blocked,
        )
        decision = policy.decide(bundle)
        expected = {
            SufficiencyDecision.SUPPORTED: ControlAction.ANSWER,
            SufficiencyDecision.PARTIAL: ControlAction.REQUEST_EVIDENCE,
            SufficiencyDecision.ABSENT: ControlAction.REQUEST_EVIDENCE,
            SufficiencyDecision.CONFLICT: ControlAction.ABSTAIN,
            SufficiencyDecision.BLOCKED: ControlAction.ABSTAIN,
        }[decision]
        if self.target.action is not expected:
            raise ValueError("trajectory target disagrees with sufficiency policy")
        if self.target.action is ControlAction.ANSWER:
            active = sorted(
                (hit for hit in self.evidence if hit.active and hit.trusted),
                key=lambda hit: (-hit.score, hit.content_digest),
            )
            if (
                not active
                or active[0].handle not in self.target.evidence_handles
                or any(
                    by_handle.score < policy.minimum_score
                    for by_handle in (
                        next(hit for hit in active if hit.handle == handle)
                        for handle in self.target.evidence_handles
                    )
                )
            ):
                raise ValueError("answer pointers do not select policy-supporting evidence")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    path: str
    sha256: str
    records: int

    def __post_init__(self) -> None:
        path = __import__("pathlib").PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or not self.path:
            raise ValueError("dataset split path is unsafe")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or self.records < 1:
            raise ValueError("dataset split digest or count is invalid")


@dataclass(frozen=True, slots=True)
class GovernedDatasetManifest:
    splits: tuple[tuple[str, DatasetSplit], ...]
    policy: SufficiencyPolicy
    maximum_evidence_items: int = 8
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        if {name for name, _ in self.splits} != {
            "train",
            "validation",
            "test",
            "adversarial",
        } or len(self.splits) != 4:
            raise ValueError("governed dataset requires four exact splits")
        if not 1 <= self.maximum_evidence_items <= 64:
            raise ValueError("maximum evidence item bound is invalid")
        identity = {
            "schema": "governed-trajectory-dataset-v1",
            "splits": self.splits,
            "policy": self.policy,
            "maximum_evidence_items": self.maximum_evidence_items,
        }
        expected = f"gtd_{content_hash(identity, length=64)}"
        if self.dataset_id is None:
            object.__setattr__(self, "dataset_id", expected)
        elif self.dataset_id != expected:
            raise ValueError("dataset ID does not match its manifest")


@dataclass(frozen=True, slots=True)
class GovernedDataset:
    manifest: GovernedDatasetManifest
    train: tuple[TrajectoryStep, ...]
    validation: tuple[TrajectoryStep, ...]
    test: tuple[TrajectoryStep, ...]
    adversarial: tuple[TrajectoryStep, ...]
