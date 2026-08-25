"""Deterministic multi-step navigation trajectories over governed fixtures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from celiums_rezero.governed.data import host_control_values
from celiums_rezero.governed.schemas import GovernedDataset, TrajectoryStep
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyDecision,
    SufficiencyPolicy,
    TenantId,
)

NAVIGATION_ACTIONS = ("search", "answer", "abstain")
SEARCH_BOUND = 2
HOST_CONTROL_CONTRACT = "host-policy-certificate-v2"


@dataclass(frozen=True, slots=True)
class NavigationStep:
    record: TrajectoryStep
    step_action: str
    search_steps_used: int
    host_certificate: tuple[float, ...]


def derive_navigation_dataset(
    dataset: GovernedDataset,
    policy: SufficiencyPolicy,
    *,
    provenance: str,
    search_bound: int = SEARCH_BOUND,
) -> tuple[tuple[NavigationStep, ...], ...]:
    del provenance
    if search_bound < 1:
        raise ValueError("navigation search bound must be positive")
    return tuple(
        _derive_trajectory(record, policy, search_bound=search_bound)
        for record in (
            *dataset.train,
            *dataset.validation,
            *dataset.test,
            *dataset.adversarial,
        )
    )


def _derive_trajectory(
    record: TrajectoryStep,
    policy: SufficiencyPolicy,
    *,
    search_bound: int,
) -> tuple[NavigationStep, ...]:
    bundle = EvidenceBundle(
        tenant=TenantId("training_fixture"),
        query_digest=hashlib.sha256(record.query.encode()).hexdigest(),
        corpus_generation=record.generation_id,
        hits=tuple(hit for hit in record.evidence if hit.active and hit.trusted),
        approximate=record.approximate,
        conflicting=record.conflicting,
        blocked=record.blocked,
    )
    decision = policy.decide(bundle)
    certificate = host_control_values(bundle, policy, HOST_CONTROL_CONTRACT)
    steps: list[NavigationStep] = []
    if decision is SufficiencyDecision.PARTIAL or (
        decision is SufficiencyDecision.ABSENT and bool(bundle.hits)
    ):
        steps.extend(
            NavigationStep(
                record=record,
                step_action="search",
                search_steps_used=used,
                host_certificate=certificate,
            )
            for used in range(1, search_bound + 1)
        )
    final_action = "answer" if decision is SufficiencyDecision.SUPPORTED else "abstain"
    steps.append(
        NavigationStep(
            record=record,
            step_action=final_action,
            search_steps_used=search_bound,
            host_certificate=certificate,
        )
    )
    return tuple(steps)


def action_labels(
    trajectories: tuple[tuple[NavigationStep, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(step.step_action for step in trajectory) for trajectory in trajectories)


def calibrated_hits(
    hits: tuple[EvidenceHit, ...],
    *,
    score_scale: float,
    step_action: str,
) -> tuple[EvidenceHit, ...]:
    if score_scale <= 0:
        raise ValueError("calibration scale must be positive")
    if step_action != "search" or len(hits) < 2:
        return hits
    return tuple(
        EvidenceHit(
            handle=hit.handle,
            source_id=hit.source_id,
            source_version=hit.source_version,
            text=hit.text,
            score=1.0,
            content_digest=hit.content_digest,
            trusted=hit.trusted,
            active=hit.active,
        )
        for hit in hits
    )


def derive_navigation_dataset_v2(
    dataset: GovernedDataset,
    policy: SufficiencyPolicy,
    *,
    provenance: str,
    search_bound: int = SEARCH_BOUND,
    score_scale: float,
) -> tuple[tuple[NavigationStep, ...], ...]:
    del provenance
    if search_bound < 1:
        raise ValueError("navigation search bound must be positive")
    trajectories: list[tuple[NavigationStep, ...]] = []
    for record in (
        *dataset.train,
        *dataset.validation,
        *dataset.test,
        *dataset.adversarial,
    ):
        trajectories.append(_derive_trajectory(record, policy, search_bound=search_bound))
        trajectories.extend(_derive_search_initiation(record, policy, score_scale=score_scale))
    return tuple(trajectories)


def _derive_search_initiation(
    record: TrajectoryStep,
    policy: SufficiencyPolicy,
    *,
    score_scale: float,
) -> tuple[tuple[NavigationStep, ...], ...]:
    hits = tuple(hit for hit in record.evidence if hit.active and hit.trusted)
    if len(hits) < 2 or record.blocked or record.conflicting:
        return ()
    calibrated = calibrated_hits(hits, score_scale=score_scale, step_action="search")
    bundle = EvidenceBundle(
        tenant=TenantId("training_fixture"),
        query_digest=hashlib.sha256(record.query.encode()).hexdigest(),
        corpus_generation=record.generation_id,
        hits=calibrated,
        approximate=record.approximate,
        conflicting=record.conflicting,
        blocked=record.blocked,
    )
    if policy.decide(bundle) is not SufficiencyDecision.PARTIAL:
        return ()
    certificate = host_control_values(bundle, policy, HOST_CONTROL_CONTRACT)
    return (
        (
            NavigationStep(
                record=record,
                step_action="search",
                search_steps_used=0,
                host_certificate=certificate,
            ),
        ),
    )


def search_decision_recall(
    predictions: tuple[tuple[str, ...], ...],
    trajectories: tuple[tuple[NavigationStep, ...], ...],
) -> float:
    if len(predictions) != len(trajectories):
        raise ValueError("prediction and trajectory counts differ")
    total = 0
    correct = 0
    for predicted, trajectory in zip(predictions, trajectories, strict=True):
        if len(predicted) != len(trajectory):
            raise ValueError("predicted step count differs")
        for action, step in zip(predicted, trajectory, strict=True):
            if step.step_action != "search":
                continue
            total += 1
            correct += action == "search"
    return 1.0 if total == 0 else float(correct) / total


def expected_pointer_targets(step: NavigationStep) -> tuple[str, ...]:
    if step.step_action != "answer":
        return ()
    return step.record.target.evidence_handles


def evidence_for_step(step: NavigationStep) -> tuple[EvidenceHit, ...]:
    if step.step_action == "search":
        return ()
    return tuple(hit for hit in step.record.evidence if hit.active and hit.trusted)
