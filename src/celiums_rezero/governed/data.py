"""Strict trajectory loading and deterministic feature batches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from celiums_rezero.governed.backbone import FrozenTextBackbone, validate_frozen_features
from celiums_rezero.governed.schemas import (
    ControlAction,
    ControlTarget,
    GovernedDataset,
    GovernedDatasetManifest,
    TrajectoryStep,
)
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyDecision,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.lab.serialization import canonical_json


@dataclass(frozen=True, slots=True)
class GovernedBatch:
    context: torch.Tensor
    evidence: torch.Tensor
    evidence_scores: torch.Tensor
    host_control_features: torch.Tensor
    evidence_mask: torch.Tensor
    action_targets: torch.Tensor
    pointer_targets: torch.Tensor


HOST_CONTROL_V1 = "host-policy-summary-v1"
HOST_CONTROL_V2 = "host-policy-certificate-v2"
HOST_CONTROL_SIZES = {HOST_CONTROL_V1: 5, HOST_CONTROL_V2: 17}


def load_trajectory_split(path: Path, policy: SufficiencyPolicy) -> tuple[TrajectoryStep, ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("trajectory split must be a regular non-symlink file")
    records: list[TrajectoryStep] = []
    for raw in path.read_text(encoding="ascii").splitlines():
        value = json.loads(raw, object_pairs_hook=_unique_object)
        if canonical_json(value) != raw or not isinstance(value, dict):
            raise ValueError("trajectory record must be canonical JSON")
        record = _trajectory(cast(dict[str, object], value))
        record.validate_policy(policy)
        records.append(record)
    scenarios = [record.scenario_id for record in records]
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("fixture split requires one record per scenario")
    return tuple(records)


def load_governed_dataset(root: Path, manifest: GovernedDatasetManifest) -> GovernedDataset:
    root = root.resolve(strict=True)
    loaded: dict[str, tuple[TrajectoryStep, ...]] = {}
    scenarios: set[str] = set()
    trajectories: set[str] = set()
    records: set[str] = set()
    contents: set[str] = set()
    paths: set[Path] = set()
    for name, split in manifest.splits:
        path = (root / split.path).resolve(strict=True)
        if root not in path.parents or path in paths or path.is_symlink():
            raise ValueError("dataset split path escaped, repeated, or is a symlink")
        paths.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != split.sha256:
            raise ValueError("dataset split digest does not match")
        items = load_trajectory_split(path, manifest.policy)
        if len(items) != split.records:
            raise ValueError("dataset split record count does not match")
        if any(len(item.evidence) > manifest.maximum_evidence_items for item in items):
            raise ValueError("dataset record exceeds the evidence-item contract")
        for item in items:
            if (
                item.scenario_id in scenarios
                or item.trajectory_id in trajectories
                or (item.record_id or "") in records
            ):
                raise ValueError("dataset identity leaked across splits")
            content_digest = hashlib.sha256(
                canonical_json(
                    {
                        "query": item.query,
                        "evidence": sorted(
                            (
                                hit.content_digest,
                                hit.score,
                                hit.trusted,
                                hit.active,
                            )
                            for hit in item.evidence
                        ),
                        "approximate": item.approximate,
                        "conflicting": item.conflicting,
                        "blocked": item.blocked,
                    }
                ).encode()
            ).hexdigest()
            if content_digest in contents:
                raise ValueError("dataset content leaked across splits")
            scenarios.add(item.scenario_id)
            trajectories.add(item.trajectory_id)
            records.add(item.record_id or "")
            contents.add(content_digest)
        loaded[name] = items
    return GovernedDataset(
        manifest,
        loaded["train"],
        loaded["validation"],
        loaded["test"],
        loaded["adversarial"],
    )


def make_batch(
    records: tuple[TrajectoryStep, ...],
    backbone: FrozenTextBackbone,
    *,
    maximum_evidence_items: int,
    device: torch.device,
    host_control_contract: str = HOST_CONTROL_V1,
    policy: SufficiencyPolicy | None = None,
) -> GovernedBatch:
    if not records:
        raise ValueError("governed batch cannot be empty")
    ordered_evidence = tuple(
        tuple(
            hit
            for hit in sorted(record.evidence, key=lambda item: item.content_digest)
            if hit.trusted and hit.active
        )
        for record in records
    )
    if any(len(items) > maximum_evidence_items for items in ordered_evidence):
        raise ValueError("batch evidence exceeds the configured item bound")
    context_texts = tuple(
        canonical_json(
            {
                "schema": "governed-control-context-v1",
                "query": record.query,
                "evidence": [
                    {
                        "text": hit.text,
                        "score": hit.score,
                        "trusted": hit.trusted,
                        "active": hit.active,
                    }
                    for hit in items
                ],
                "approximate": record.approximate,
                "conflicting": record.conflicting,
                "blocked": record.blocked,
            }
        )
        for record, items in zip(records, ordered_evidence, strict=True)
    )
    context = backbone.encode(context_texts, device=device)
    validate_frozen_features(backbone, context, items=len(records))
    hidden = context.shape[-1]
    evidence = torch.zeros(
        (len(records), maximum_evidence_items, hidden), dtype=torch.float32, device=device
    )
    mask = torch.zeros((len(records), maximum_evidence_items), dtype=torch.bool, device=device)
    scores = torch.zeros(
        (len(records), maximum_evidence_items), dtype=torch.float32, device=device
    )
    pointers = torch.zeros_like(mask, dtype=torch.float32)
    actions = torch.empty(len(records), dtype=torch.long, device=device)
    selected_policy = SufficiencyPolicy() if policy is None else policy
    host_control = torch.tensor(
        [
            host_control_values(
                _record_bundle(record, items), selected_policy, host_control_contract
            )
            for record, items in zip(records, ordered_evidence, strict=True)
        ],
        dtype=torch.float32,
        device=device,
    )
    action_index = {action: index for index, action in enumerate(ControlAction)}
    flattened = tuple(hit for items in ordered_evidence for hit in items)
    flattened_features = (
        backbone.encode(tuple(hit.text for hit in flattened), device=device)
        if flattened
        else torch.empty((0, hidden), dtype=torch.float32, device=device)
    )
    validate_frozen_features(backbone, flattened_features, items=len(flattened))
    offset = 0
    for row, (record, ordered) in enumerate(
        zip(records, ordered_evidence, strict=True)
    ):
        if ordered:
            selected = set(record.target.evidence_handles)
            if not selected <= {hit.handle for hit in ordered}:
                raise ValueError("target evidence was removed by the item bound")
            encoded = flattened_features[offset : offset + len(ordered)]
            evidence[row, : len(ordered)] = encoded
            scores[row, : len(ordered)] = torch.tensor(
                [hit.score for hit in ordered], dtype=torch.float32, device=device
            )
            mask[row, : len(ordered)] = True
            for column, hit in enumerate(ordered):
                pointers[row, column] = float(hit.handle in selected)
            offset += len(ordered)
        actions[row] = action_index[record.target.action]
    return GovernedBatch(context, evidence, scores, host_control, mask, actions, pointers)


def materialize_governed_batch(
    records: tuple[TrajectoryStep, ...],
    backbone: FrozenTextBackbone,
    *,
    maximum_evidence_items: int,
    feature_batch_size: int,
    device: torch.device,
    host_control_contract: str = HOST_CONTROL_V1,
    policy: SufficiencyPolicy | None = None,
) -> GovernedBatch:
    if feature_batch_size < 1:
        raise ValueError("feature batch size must be positive")
    batches = tuple(
        make_batch(
            records[start : start + feature_batch_size],
            backbone,
            maximum_evidence_items=maximum_evidence_items,
            device=device,
            host_control_contract=host_control_contract,
            policy=policy,
        )
        for start in range(0, len(records), feature_batch_size)
    )
    if not batches:
        raise ValueError("governed batch cannot be empty")
    return GovernedBatch(
        context=torch.cat(tuple(batch.context for batch in batches)),
        evidence=torch.cat(tuple(batch.evidence for batch in batches)),
        evidence_scores=torch.cat(tuple(batch.evidence_scores for batch in batches)),
        host_control_features=torch.cat(
            tuple(batch.host_control_features for batch in batches)
        ),
        evidence_mask=torch.cat(tuple(batch.evidence_mask for batch in batches)),
        action_targets=torch.cat(tuple(batch.action_targets for batch in batches)),
        pointer_targets=torch.cat(tuple(batch.pointer_targets for batch in batches)),
    )


def host_control_values(
    bundle: EvidenceBundle,
    policy: SufficiencyPolicy,
    contract: str = HOST_CONTROL_V1,
) -> tuple[float, ...]:
    if contract not in HOST_CONTROL_SIZES:
        raise ValueError("host control feature contract is invalid")
    active = tuple(hit for hit in bundle.hits if hit.active and hit.trusted)
    scores = sorted((hit.score for hit in active), reverse=True)
    top = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else 0.0
    base = (
        float(bundle.blocked),
        float(bundle.conflicting),
        float(not active),
        top,
        float(len(active)),
    )
    if contract == HOST_CONTROL_V1:
        return base
    decision = policy.decide(bundle)
    decision_order = (
        SufficiencyDecision.SUPPORTED,
        SufficiencyDecision.PARTIAL,
        SufficiencyDecision.ABSENT,
        SufficiencyDecision.CONFLICT,
        SufficiencyDecision.BLOCKED,
    )
    certificate = (
        float(bundle.approximate),
        second,
        top - second,
        float(sum(score >= policy.minimum_score for score in scores)),
        policy.minimum_score,
        policy.minimum_margin,
        float(policy.minimum_trusted_hits),
        *(float(decision is item) for item in decision_order),
    )
    values = (*base, *certificate)
    if len(values) != HOST_CONTROL_SIZES[contract]:
        raise RuntimeError("host control feature contract size is invalid")
    return values


def _record_bundle(
    record: TrajectoryStep, evidence: tuple[EvidenceHit, ...]
) -> EvidenceBundle:
    return EvidenceBundle(
        tenant=TenantId("training_fixture"),
        query_digest=hashlib.sha256(record.query.encode()).hexdigest(),
        corpus_generation=record.generation_id,
        hits=evidence,
        approximate=record.approximate,
        conflicting=record.conflicting,
        blocked=record.blocked,
    )


def _trajectory(value: dict[str, object]) -> TrajectoryStep:
    fields = {
        "schema",
        "record_id",
        "trajectory_id",
        "scenario_id",
        "step_index",
        "query",
        "generation_id",
        "evidence",
        "approximate",
        "conflicting",
        "blocked",
        "target",
        "provenance",
    }
    if set(value) != fields or value["schema"] != "governed-trajectory-step-v1":
        raise ValueError("trajectory fields or schema are invalid")
    evidence = value["evidence"]
    target = value["target"]
    if not isinstance(evidence, list) or not isinstance(target, dict):
        raise ValueError("trajectory evidence or target is invalid")
    hits = tuple(_hit(item) for item in evidence)
    handles = target.get("evidence_handles")
    if not isinstance(handles, list) or set(target) != {"action", "evidence_handles"}:
        raise ValueError("trajectory target fields are invalid")
    return TrajectoryStep(
        trajectory_id=_string(value["trajectory_id"]),
        scenario_id=_string(value["scenario_id"]),
        step_index=_integer(value["step_index"]),
        query=_string(value["query"]),
        generation_id=_string(value["generation_id"]),
        evidence=hits,
        approximate=_boolean(value["approximate"]),
        conflicting=_boolean(value["conflicting"]),
        blocked=_boolean(value["blocked"]),
        target=ControlTarget(
            ControlAction(_string(target["action"])),
            tuple(_string(item) for item in handles),
        ),
        provenance=_string(value["provenance"]),
        record_id=_string(value["record_id"]),
    )


def _hit(value: object) -> EvidenceHit:
    fields = {
        "handle",
        "source_id",
        "source_version",
        "text",
        "score",
        "content_digest",
        "trusted",
        "active",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("trajectory evidence fields are invalid")
    return EvidenceHit(
        handle=_string(value["handle"]),
        source_id=_string(value["source_id"]),
        source_version=_string(value["source_version"]),
        text=_string(value["text"]),
        score=float(value["score"]),
        content_digest=_string(value["content_digest"]),
        trusted=_boolean(value["trusted"]),
        active=_boolean(value["active"]),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("trajectory JSON contains duplicate keys")
        result[key] = value
    return result


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("trajectory field must be a string")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("trajectory field must be an integer")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("trajectory field must be boolean")
    return value
