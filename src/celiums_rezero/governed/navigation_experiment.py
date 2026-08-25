"""Preregistered ReZero neuropilot training over fixture navigation trajectories."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from torch import nn

from celiums_rezero.core.gates import is_gate_parameter
from celiums_rezero.core.optim import build_optimizer_groups
from celiums_rezero.governed.backbone import FrozenTextBackbone, validate_frozen_features
from celiums_rezero.governed.hyphaelm import (
    ABSTAIN_INDEX,
    ANSWER_INDEX,
    SEARCH_INDEX,
    ReZeroNeuroPilot,
)
from celiums_rezero.governed.navigation import (
    NAVIGATION_ACTIONS,
    NavigationStep,
    action_labels,
    evidence_for_step,
    expected_pointer_targets,
)
from celiums_rezero.governed.schemas import GovernedDataset
from celiums_rezero.lab.serialization import canonical_json

SCHEMA = "hyphae-transformer.gemma4-e4b-rezero-navigation-experiment/v1"
POINTER_LOGIT_THRESHOLD = 0.0


class NavigationTrainingSummary(TypedDict):
    initial_loss: float
    selected_loss: float
    final_loss: float
    selected_epoch: int
    checkpoint_sha256: str


class NavigationEvaluation(TypedDict):
    action_accuracy: float
    answer_recall: float
    abstention_recall: float
    search_decision_recall: float
    evidence_exact_match: float
    unsafe_answer_rate: float
    passed: bool


class NavigationSeedReport(TypedDict):
    seed: int
    training: NavigationTrainingSummary
    validation: NavigationEvaluation


@dataclass(frozen=True, slots=True)
class NavigationTrainConfig:
    epochs: int
    learning_rate: float
    evidence_loss_weight: float = 2.0
    gradient_clip: float = 1.0
    seed: int = 17
    device: str = "cpu"
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    checkpoint_selection: str = "minimum_training_loss"

    def __post_init__(self) -> None:
        values = (self.learning_rate, self.evidence_loss_weight, self.gradient_clip)
        if self.epochs < 1 or any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("navigation training configuration is invalid")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("navigation optimizer is invalid")
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("navigation weight decay is invalid")
        if self.checkpoint_selection != "minimum_training_loss":
            raise ValueError("navigation checkpoint selection is invalid")


@dataclass(frozen=True, slots=True)
class NavigationBatch:
    context: torch.Tensor
    evidence: torch.Tensor
    evidence_scores: torch.Tensor
    host_control: torch.Tensor
    evidence_mask: torch.Tensor
    action_targets: torch.Tensor
    pointer_targets: torch.Tensor


def build_navigation_batch(
    steps: tuple[NavigationStep, ...],
    backbone: FrozenTextBackbone,
    *,
    maximum_evidence_items: int,
    device: torch.device,
) -> NavigationBatch:
    if not steps:
        raise ValueError("navigation batch cannot be empty")
    evidence_sets = tuple(evidence_for_step(step) for step in steps)
    if any(len(items) > maximum_evidence_items for items in evidence_sets):
        raise ValueError("navigation evidence exceeds the configured item bound")
    context_texts = tuple(
        canonical_json(
            {
                "schema": "governed-navigation-context-v1",
                "query": step.record.query,
                "evidence": [hit.text for hit in items],
                "search_steps_used": step.search_steps_used,
            }
        )
        for step, items in zip(steps, evidence_sets, strict=True)
    )
    context = backbone.encode(context_texts, device=device)
    validate_frozen_features(backbone, context, items=len(steps))
    hidden = context.shape[-1]
    evidence = torch.zeros(
        (len(steps), maximum_evidence_items, hidden), dtype=torch.float32, device=device
    )
    mask = torch.zeros((len(steps), maximum_evidence_items), dtype=torch.bool, device=device)
    scores = torch.zeros((len(steps), maximum_evidence_items), dtype=torch.float32, device=device)
    pointers = torch.zeros_like(mask, dtype=torch.float32)
    action_index = {name: index for index, name in enumerate(NAVIGATION_ACTIONS)}
    actions = torch.empty(len(steps), dtype=torch.long, device=device)
    flattened = tuple(hit for items in evidence_sets for hit in items)
    flattened_features = (
        backbone.encode(tuple(hit.text for hit in flattened), device=device)
        if flattened
        else torch.empty((0, hidden), dtype=torch.float32, device=device)
    )
    validate_frozen_features(backbone, flattened_features, items=len(flattened))
    host_control = torch.tensor(
        [step.host_certificate for step in steps], dtype=torch.float32, device=device
    )
    offset = 0
    for row, (step, items) in enumerate(zip(steps, evidence_sets, strict=True)):
        if items:
            encoded = flattened_features[offset : offset + len(items)]
            evidence[row, : len(items)] = encoded
            scores[row, : len(items)] = torch.tensor(
                [hit.score for hit in items], dtype=torch.float32, device=device
            )
            mask[row, : len(items)] = True
            selected = set(expected_pointer_targets(step))
            for column, hit in enumerate(items):
                pointers[row, column] = float(hit.handle in selected)
            offset += len(items)
        actions[row] = action_index[step.step_action]
    return NavigationBatch(context, evidence, scores, host_control, mask, actions, pointers)


def train_navigation_pilot(
    *,
    backbone: FrozenTextBackbone,
    pilot: ReZeroNeuroPilot,
    train_steps: tuple[NavigationStep, ...],
    config: NavigationTrainConfig,
    checkpoint: Path,
    maximum_evidence_items: int,
    deadline: float | None = None,
) -> tuple[float, float, float, int]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config.device)
    pilot.to(device).train()
    has_gates = any(is_gate_parameter(parameter) for parameter in pilot.parameters())
    optimizer = (
        torch.optim.AdamW(
            build_optimizer_groups(
                pilot, lr=config.learning_rate, weight_decay=config.weight_decay
            ),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.optimizer == "adamw" and has_gates
        else torch.optim.AdamW(
            pilot.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if config.optimizer == "adamw"
        else torch.optim.SGD(pilot.parameters(), lr=config.learning_rate)
    )
    before = canonical_json(backbone.identity)
    state_before = backbone.state_fingerprint()
    batch = build_navigation_batch(
        train_steps,
        backbone,
        maximum_evidence_items=maximum_evidence_items,
        device=device,
    )
    losses: list[float] = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected_epoch = config.epochs
    best_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("navigation wall-time budget exceeded")
        optimizer.zero_grad(set_to_none=True)
        action_logits, pointers = pilot(
            batch.context,
            batch.evidence,
            batch.evidence_mask,
            batch.evidence_scores,
            batch.host_control,
        )
        action_loss = nn.functional.cross_entropy(action_logits, batch.action_targets)
        finite = pointers.masked_fill(~batch.evidence_mask, 0)
        pointer_loss = nn.functional.binary_cross_entropy_with_logits(
            finite[batch.evidence_mask], batch.pointer_targets[batch.evidence_mask]
        )
        loss = action_loss + config.evidence_loss_weight * pointer_loss
        if not torch.isfinite(loss):
            raise RuntimeError("navigation loss is non-finite")
        torch.autograd.backward(loss)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in pilot.parameters()
        ):
            raise RuntimeError("navigation pilot gradient is non-finite")
        nn.utils.clip_grad_norm_(pilot.parameters(), config.gradient_clip)
        optimizer.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            selected_state = {
                name: tensor.detach().cpu().clone() for name, tensor in pilot.state_dict().items()
            }
            selected_epoch = epoch
    if selected_state is not None:
        pilot.load_state_dict(selected_state)
    if canonical_json(backbone.identity) != before or backbone.state_fingerprint() != state_before:
        raise RuntimeError("backbone changed during navigation training")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "version": 1,
            "head": pilot.state_dict(),
            "backbone": before,
            "action_order": list(NAVIGATION_ACTIONS),
            "maximum_evidence_items": maximum_evidence_items,
        },
        temporary,
    )
    temporary.replace(checkpoint)
    return losses[0], best_loss, losses[-1], selected_epoch


def evaluate_navigation(
    *,
    backbone: FrozenTextBackbone,
    pilot: ReZeroNeuroPilot,
    steps: tuple[NavigationStep, ...],
    maximum_evidence_items: int,
    device: torch.device,
    gates: dict[str, float],
) -> NavigationEvaluation:
    batch = build_navigation_batch(
        steps, backbone, maximum_evidence_items=maximum_evidence_items, device=device
    )
    pilot.eval()
    with torch.no_grad():
        action_logits, pointers = pilot(
            batch.context,
            batch.evidence,
            batch.evidence_mask,
            batch.evidence_scores,
            batch.host_control,
        )
    predicted = action_logits.argmax(-1)
    action_accuracy = float((predicted == batch.action_targets).float().mean())
    answer_rows = batch.action_targets == ANSWER_INDEX
    search_rows = batch.action_targets == SEARCH_INDEX
    abstain_rows = batch.action_targets == ABSTAIN_INDEX
    answer_recall = (
        float((predicted[answer_rows] == ANSWER_INDEX).float().mean()) if answer_rows.any() else 1.0
    )
    search_recall = (
        float((predicted[search_rows] == SEARCH_INDEX).float().mean()) if search_rows.any() else 1.0
    )
    abstention_recall = (
        float((predicted[abstain_rows] == ABSTAIN_INDEX).float().mean())
        if abstain_rows.any()
        else 1.0
    )
    exact = (torch.sigmoid(pointers) >= POINTER_LOGIT_THRESHOLD) == batch.pointer_targets.bool()
    evidence_exact = float(exact[answer_rows].all(-1).float().mean()) if answer_rows.any() else 1.0
    unsafe_rows = ~answer_rows
    unsafe = (
        float((predicted[unsafe_rows] == ANSWER_INDEX).float().mean()) if unsafe_rows.any() else 0.0
    )
    passed = (
        action_accuracy >= gates["action_accuracy"]
        and answer_recall >= gates["answer_recall"]
        and abstention_recall >= gates["abstention_recall"]
        and search_recall >= gates["search_decision_recall"]
        and evidence_exact >= gates["evidence_exact_match"]
        and unsafe <= gates["unsafe_answer_rate"]
    )
    return {
        "action_accuracy": action_accuracy,
        "answer_recall": answer_recall,
        "abstention_recall": abstention_recall,
        "search_decision_recall": search_recall,
        "evidence_exact_match": evidence_exact,
        "unsafe_answer_rate": unsafe,
        "passed": passed,
    }


def build_search_report(
    learning_rate: float, seed_reports: list[NavigationSeedReport]
) -> dict[str, object]:
    return {
        "learning_rate": learning_rate,
        "seeds": seed_reports,
        "selection_key": _selection_key(learning_rate, seed_reports),
    }


def run_navigation_experiment(
    *,
    backbone: FrozenTextBackbone,
    preregistration: dict[str, Any],
    preregistration_sha256: str,
    dataset: GovernedDataset,
    trajectories: tuple[tuple[NavigationStep, ...], ...],
    output: Path,
    device: torch.device,
    maximum_evidence_items: int,
) -> dict[str, object]:
    _validate_preregistration(preregistration, dataset)
    search = preregistration["training_search"]
    gates = preregistration["gates"]
    seeds = tuple(int(value) for value in search["seeds"])
    learning_rates = tuple(float(value) for value in search["candidate_learning_rates"])
    action_labels(trajectories)
    train_count = len(dataset.train)
    validation_count = len(dataset.validation)
    train_steps = tuple(step for trajectory in trajectories[:train_count] for step in trajectory)
    validation_steps = tuple(
        step
        for trajectory in trajectories[train_count : train_count + validation_count]
        for step in trajectory
    )
    output.mkdir(parents=True, exist_ok=True)
    search_reports: list[dict[str, object]] = []
    for learning_rate in learning_rates:
        seed_reports: list[NavigationSeedReport] = []
        for seed in seeds:
            checkpoint = output / "search" / str(learning_rate) / f"seed-{seed}.pt"
            pilot = _pilot_from_prereg(preregistration, backbone, maximum_evidence_items)
            initial, selected, final, epoch = train_navigation_pilot(
                backbone=backbone,
                pilot=pilot,
                train_steps=train_steps,
                config=NavigationTrainConfig(
                    epochs=int(search["epochs"]),
                    learning_rate=learning_rate,
                    evidence_loss_weight=float(search["evidence_loss_weight"]),
                    gradient_clip=float(search["gradient_clip"]),
                    seed=seed,
                    device=str(device),
                    optimizer=str(search["optimizer"]),
                    weight_decay=float(search["weight_decay"]),
                    checkpoint_selection=str(search["checkpoint_selection"]),
                ),
                checkpoint=checkpoint,
                maximum_evidence_items=maximum_evidence_items,
            )
            validation = evaluate_navigation(
                backbone=backbone,
                pilot=pilot,
                steps=validation_steps,
                maximum_evidence_items=maximum_evidence_items,
                device=device,
                gates=gates,
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "training": {
                        "initial_loss": initial,
                        "selected_loss": selected,
                        "final_loss": final,
                        "selected_epoch": epoch,
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    },
                    "validation": validation,
                }
            )
        search_reports.append(build_search_report(learning_rate, seed_reports))
    selected_report = min(search_reports, key=_selection_key_value)
    selected_rate = float(cast("float", selected_report["learning_rate"]))
    final_reports: list[dict[str, object]] = []
    for seed_report in cast("list[NavigationSeedReport]", selected_report["seeds"]):
        final_reports.append(dict(seed_report))
    report: dict[str, object] = {
        "schema": SCHEMA,
        "completed": True,
        "passed": all(
            bool(cast("dict[str, object]", item["validation"])["passed"]) for item in final_reports
        ),
        "scope": "gemma",
        "backbone": {
            "model_id": backbone.identity.model_id,
            "revision": backbone.identity.revision,
            "feature_contract": backbone.identity.feature_contract,
        },
        "backbone_unchanged": True,
        "preregistration_sha256": preregistration_sha256,
        "dataset_id": dataset.manifest.dataset_id,
        "selected_learning_rate": selected_rate,
        "search": search_reports,
        "final": final_reports,
    }
    report["report_id"] = "rznp_" + hashlib.sha256(canonical_json(report).encode()).hexdigest()
    (output / "rezero-navigation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _pilot_from_prereg(
    preregistration: dict[str, Any],
    backbone: FrozenTextBackbone,
    maximum_evidence_items: int,
) -> ReZeroNeuroPilot:
    candidate = preregistration["candidate"]
    search = preregistration["training_search"]
    return ReZeroNeuroPilot(
        backbone.identity.hidden_size,
        control_size=int(candidate["control_size"]),
        n_layers=int(candidate["layers"]),
        n_heads=int(candidate["n_heads"]),
        host_control_contract=str(
            candidate.get("host_control_contract", "host-policy-certificate-v2")
        ),
        action_policy_prior_scale=float(candidate.get("action_policy_prior_scale", 20.0)),
        action_residual_bound=float(candidate.get("action_residual_bound", 1.0)),
        pointer_policy_score=float(search["pointer_policy_score"]),
        pointer_policy_scale=float(search["pointer_policy_scale"]),
        maximum_evidence_items=maximum_evidence_items,
    )


def _selection_key_value(report: dict[str, object]) -> tuple[float, ...]:
    key = report["selection_key"]
    assert isinstance(key, (list, tuple))
    return tuple(float(value) for value in key)


def _selection_key(rate: float, seed_reports: list[NavigationSeedReport]) -> tuple[float, ...]:
    validations = [value["validation"] for value in seed_reports]
    training = [value["training"] for value in seed_reports]
    return (
        -sum(1.0 if value["passed"] else 0.0 for value in validations),
        statistics.fmean(value["unsafe_answer_rate"] for value in validations),
        -statistics.fmean(value["search_decision_recall"] for value in validations),
        -statistics.fmean(value["action_accuracy"] for value in validations),
        statistics.fmean(value["selected_loss"] for value in training),
        rate,
    )


def _validate_preregistration(preregistration: dict[str, Any], dataset: GovernedDataset) -> None:
    if (
        preregistration.get("schema")
        != "hyphae-transformer.gemma4-e4b-rezero-navigation-preregistration/v1"
    ):
        raise ValueError("navigation preregistration schema is invalid")
    if preregistration["dataset"]["governed_dataset_id"] != dataset.manifest.dataset_id:
        raise ValueError("navigation dataset identity differs")
    for key in (
        "action_accuracy",
        "answer_recall",
        "abstention_recall",
        "search_decision_recall",
        "evidence_exact_match",
        "unsafe_answer_rate",
    ):
        if key not in preregistration["gates"]:
            raise ValueError("navigation gates are incomplete")
