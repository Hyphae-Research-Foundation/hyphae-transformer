"""Preregistered frozen-backbone ReZero sequence-control experiment."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from celiums_rezero.core.gates import is_gate_parameter
from celiums_rezero.core.optim import build_optimizer_groups
from celiums_rezero.governed.backbone import FrozenTextBackbone
from celiums_rezero.governed.data import (
    GovernedBatch,
    load_governed_dataset,
    materialize_governed_batch,
)
from celiums_rezero.governed.evaluation import ControlEvaluation, evaluate_control_head
from celiums_rezero.governed.model import ReZeroSequenceControlHead
from celiums_rezero.governed.schemas import (
    DatasetSplit,
    GovernedDataset,
    GovernedDatasetManifest,
    TrajectoryStep,
)
from celiums_rezero.governed.trainer import ControlTrainConfig, train_control_head
from celiums_rezero.knowledge.schemas import SufficiencyPolicy
from celiums_rezero.lab.serialization import canonical_json

PREREGISTRATION_SCHEMA = (
    "hyphae-transformer.gemma4-e4b-rezero-sequence-control-preregistration/v1"
)
SELECTION_RANKING = (
    "maximum_validation_passed_seeds",
    "minimum_mean_validation_unsafe_answer_rate",
    "maximum_mean_validation_abstention_recall",
    "maximum_mean_validation_evidence_exact_match",
    "maximum_mean_validation_action_accuracy",
    "minimum_mean_training_final_loss",
    "minimum_learning_rate",
)


def load_governed_dataset_directory(
    root: Path, *, expected_dataset_id: str
) -> GovernedDataset:
    values = json.loads((root / "manifest.json").read_text())
    if values.get("dataset_id") != expected_dataset_id:
        raise ValueError("governed dataset ID differs from preregistration")
    manifest = GovernedDatasetManifest(
        splits=tuple(
            (name, DatasetSplit(**item))
            for name, item in sorted(values["splits"].items())
        ),
        policy=SufficiencyPolicy(**values["policy"]),
        maximum_evidence_items=int(values["maximum_evidence_items"]),
        dataset_id=str(values["dataset_id"]),
    )
    return load_governed_dataset(root, manifest)


def run_rezero_sequence_experiment(
    *,
    backbone: FrozenTextBackbone,
    dataset: GovernedDataset,
    preregistration: dict[str, Any],
    preregistration_sha256: str,
    output: Path,
    device: torch.device,
    feature_batch_size: int,
    scope: str,
) -> dict[str, object]:
    _validate_preregistration(preregistration, dataset, scope=scope)
    if feature_batch_size < 1:
        raise ValueError("feature batch size must be positive")
    training = preregistration["training_search"]
    candidate = preregistration["candidate"]
    gates = preregistration["gates"]
    seeds = tuple(int(value) for value in training["seeds"])
    learning_rates = tuple(float(value) for value in training["candidate_learning_rates"])
    backbone_before = backbone.state_fingerprint()
    output.mkdir(parents=True, exist_ok=True)

    feature_started = time.perf_counter()
    selection_batches = {
        "train": materialize_governed_batch(
            dataset.train,
            backbone,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            feature_batch_size=feature_batch_size,
            device=device,
        ),
        "validation": materialize_governed_batch(
            dataset.validation,
            backbone,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            feature_batch_size=feature_batch_size,
            device=device,
        ),
    }
    _require_unchanged(backbone, backbone_before)

    search_reports: list[dict[str, object]] = []
    for learning_rate in learning_rates:
        seed_reports: list[dict[str, object]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            head = _new_head(backbone, candidate, training, device=device)
            parameter_count = sum(parameter.numel() for parameter in head.parameters())
            if parameter_count > preregistration["structural_gates"][
                "maximum_trainable_parameters"
            ]:
                raise ValueError("ReZero controller exceeds its parameter gate")
            checkpoint = output / "search" / str(learning_rate) / f"seed-{seed}.pt"
            summary = train_control_head(
                backbone,
                head,
                dataset.train,
                _train_config(training, learning_rate=learning_rate, seed=seed, device=device),
                checkpoint=checkpoint,
                maximum_evidence_items=dataset.manifest.maximum_evidence_items,
                batch=selection_batches["train"],
            )
            validation = evaluate_control_head(
                backbone,
                head,
                dataset.validation,
                maximum_evidence_items=dataset.manifest.maximum_evidence_items,
                gates=gates,
                batch=selection_batches["validation"],
                minimum_confidence=float(training["minimum_confidence"]),
                pointer_threshold=float(training["pointer_threshold"]),
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "parameters": parameter_count,
                    "checkpoint_sha256": summary.checkpoint_sha256,
                    "training": asdict(summary),
                    "validation": asdict(validation),
                }
            )
            _require_unchanged(backbone, backbone_before)
        search_reports.append(
            {
                "learning_rate": learning_rate,
                "seeds": seed_reports,
                "selection_key": list(_selection_key(learning_rate, seed_reports)),
            }
        )

    selected = min(search_reports, key=_report_selection_key)
    selected_learning_rate = _as_float(selected["learning_rate"])

    # Test and adversarial features and labels remain unused until selection is fixed.
    final_batches = {
        "test": materialize_governed_batch(
            dataset.test,
            backbone,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            feature_batch_size=feature_batch_size,
            device=device,
        ),
        "adversarial": materialize_governed_batch(
            dataset.adversarial,
            backbone,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            feature_batch_size=feature_batch_size,
            device=device,
        ),
    }
    final_reports: list[dict[str, object]] = []
    selected_seeds = cast(list[dict[str, object]], selected["seeds"])
    selected_by_seed = {_as_int(item["seed"]): item for item in selected_seeds}
    for seed in seeds:
        checkpoint = output / "search" / str(selected_learning_rate) / f"seed-{seed}.pt"
        head = _new_head(backbone, candidate, training, device=device)
        value = torch.load(checkpoint, map_location=device, weights_only=True)
        if not isinstance(value, dict) or not isinstance(value.get("head"), dict):
            raise ValueError("selected ReZero checkpoint is invalid")
        head.load_state_dict(value["head"])
        test = _evaluate(
            backbone, head, dataset.test, final_batches["test"], gates, training
        )
        adversarial = _evaluate(
            backbone,
            head,
            dataset.adversarial,
            final_batches["adversarial"],
            gates,
            training,
        )
        training_summary = cast(dict[str, object], selected_by_seed[seed]["training"])
        final_reports.append(
            {
                "seed": seed,
                "checkpoint_sha256": selected_by_seed[seed]["checkpoint_sha256"],
                "training": training_summary,
                "test": asdict(test),
                "adversarial": asdict(adversarial),
                "passed": (
                    bool(training_summary["backbone_unchanged"])
                    and _as_float(training_summary["final_loss"])
                    <= _as_float(training_summary["initial_loss"])
                    and test.passed
                    and adversarial.passed
                ),
            }
        )
        _require_unchanged(backbone, backbone_before)

    report: dict[str, object] = {
        "schema": "hyphae-transformer.rezero-sequence-control-experiment/v1",
        "completed": True,
        "passed": all(bool(item["passed"]) for item in final_reports),
        "scope": scope,
        "preregistration_sha256": preregistration_sha256,
        "dataset_id": dataset.manifest.dataset_id,
        "backbone": asdict(backbone.identity),
        "backbone_unchanged": backbone.state_fingerprint() == backbone_before,
        "feature_materialization_seconds": time.perf_counter() - feature_started,
        "selection_ranking": list(SELECTION_RANKING),
        "selected_learning_rate": selected_learning_rate,
        "search": search_reports,
        "final": final_reports,
    }
    report["report_id"] = "rzsc_" + hashlib.sha256(
        canonical_json(report).encode()
    ).hexdigest()
    (output / "rezero-sequence-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_rezero_sequence_smoke(
    *,
    backbone: FrozenTextBackbone,
    dataset: GovernedDataset,
    preregistration: dict[str, Any],
    preregistration_sha256: str,
    output: Path,
    device: torch.device,
    feature_batch_size: int,
    maximum_vram_gib: float,
) -> dict[str, object]:
    _validate_preregistration(preregistration, dataset, scope="gemma")
    if feature_batch_size < 1 or maximum_vram_gib <= 0:
        raise ValueError("ReZero smoke limits are invalid")
    training = preregistration["training_search"]
    candidate = preregistration["candidate"]
    backbone_before = backbone.state_fingerprint()
    records = dataset.train[:feature_batch_size]
    batch = materialize_governed_batch(
        records,
        backbone,
        maximum_evidence_items=dataset.manifest.maximum_evidence_items,
        feature_batch_size=feature_batch_size,
        device=device,
    )
    _require_unchanged(backbone, backbone_before)
    torch.manual_seed(int(training["seeds"][0]))
    head = _new_head(backbone, candidate, training, device=device)
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    parameter_limit = int(
        preregistration["structural_gates"]["maximum_trainable_parameters"]
    )
    if parameter_count > parameter_limit:
        raise ValueError("ReZero controller exceeds its parameter gate")
    logits = head(
        batch.context,
        batch.evidence,
        batch.evidence_mask,
        batch.evidence_scores,
        batch.host_control_features,
    )
    action_loss = nn.functional.cross_entropy(logits.action_logits, batch.action_targets)
    finite_pointers = logits.evidence_logits.masked_fill(~batch.evidence_mask, 0)
    answer_rows = batch.evidence_mask & (batch.action_targets == 0).unsqueeze(-1)
    pointer_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            finite_pointers[answer_rows], batch.pointer_targets[answer_rows]
        )
        if answer_rows.any()
        else torch.zeros((), device=device)
    )
    loss = action_loss + float(training["evidence_loss_weight"]) * pointer_loss
    torch.autograd.backward(loss)
    gate_parameters = [
        parameter for parameter in head.parameters() if is_gate_parameter(parameter)
    ]
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in gate_parameters
    )
    groups = build_optimizer_groups(
        head,
        lr=float(training["candidate_learning_rates"][0]),
        weight_decay=float(training["weight_decay"]),
    )
    gate_group = next((group for group in groups if group["name"] == "gates"), None)
    gates_excluded_from_decay = gate_group is not None and gate_group["weight_decay"] == 0
    peak_vram_bytes = (
        torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    )
    maximum_vram_bytes = int(maximum_vram_gib * 1024**3)
    report: dict[str, object] = {
        "schema": "hyphae-transformer.rezero-sequence-control-smoke/v1",
        "completed": True,
        "passed": (
            bool(torch.isfinite(loss))
            and gradients_finite
            and gates_excluded_from_decay
            and backbone.state_fingerprint() == backbone_before
            and peak_vram_bytes <= maximum_vram_bytes
        ),
        "preregistration_sha256": preregistration_sha256,
        "dataset_id": dataset.manifest.dataset_id,
        "backbone": asdict(backbone.identity),
        "backbone_unchanged": backbone.state_fingerprint() == backbone_before,
        "records": len(records),
        "parameters": parameter_count,
        "residual_gates": len(gate_parameters),
        "gate_gradients_finite": gradients_finite,
        "gates_excluded_from_weight_decay": gates_excluded_from_decay,
        "loss": float(loss.detach()),
        "peak_vram_bytes": peak_vram_bytes,
        "maximum_vram_bytes": maximum_vram_bytes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _new_head(
    backbone: FrozenTextBackbone,
    candidate: dict[str, Any],
    training: dict[str, Any],
    *,
    device: torch.device,
) -> ReZeroSequenceControlHead:
    head = ReZeroSequenceControlHead(
        backbone.identity.hidden_size,
        control_size=int(candidate["control_size"]),
        n_layers=int(candidate["layers"]),
        n_heads=int(candidate["n_heads"]),
        pointer_policy_score=float(training["pointer_policy_score"]),
        pointer_policy_scale=float(training["pointer_policy_scale"]),
        maximum_evidence_items=int(candidate["maximum_evidence_items"]),
    ).to(device)
    gates = [parameter for parameter in head.parameters() if is_gate_parameter(parameter)]
    if len(gates) != int(candidate["layers"]) or any(
        float(parameter.detach()) != float(candidate["gate_init"]) for parameter in gates
    ):
        raise ValueError("ReZero controller residual gates differ from preregistration")
    if any(block.attention_gate is not block.mlp_gate for block in head.blocks):
        raise ValueError("ReZero controller does not share one gate per block")
    return head


def _train_config(
    training: dict[str, Any], *, learning_rate: float, seed: int, device: torch.device
) -> ControlTrainConfig:
    return ControlTrainConfig(
        epochs=int(training["epochs"]),
        learning_rate=learning_rate,
        evidence_loss_weight=float(training["evidence_loss_weight"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=seed,
        device=str(device),
        optimizer=str(training["optimizer"]),
        weight_decay=float(training["weight_decay"]),
        pointer_loss_scope=str(training["pointer_loss_scope"]),
    )


def _evaluate(
    backbone: FrozenTextBackbone,
    head: ReZeroSequenceControlHead,
    records: tuple[TrajectoryStep, ...],
    batch: GovernedBatch,
    gates: dict[str, float],
    training: dict[str, Any],
) -> ControlEvaluation:
    return evaluate_control_head(
        backbone,
        head,
        records,
        maximum_evidence_items=int(training["maximum_evidence_items"]),
        gates=gates,
        batch=batch,
        minimum_confidence=float(training["minimum_confidence"]),
        pointer_threshold=float(training["pointer_threshold"]),
    )


def _selection_key(
    learning_rate: float, seed_reports: list[dict[str, object]]
) -> tuple[float, ...]:
    validations = [
        cast(dict[str, object], item["validation"]) for item in seed_reports
    ]
    training = [cast(dict[str, object], item["training"]) for item in seed_reports]
    return (
        -sum(bool(value["passed"]) for value in validations),
        statistics.fmean(_as_float(value["unsafe_answer_rate"]) for value in validations),
        -statistics.fmean(_as_float(value["abstention_recall"]) for value in validations),
        -statistics.fmean(_as_float(value["evidence_exact_match"]) for value in validations),
        -statistics.fmean(_as_float(value["action_accuracy"]) for value in validations),
        statistics.fmean(_as_float(value["final_loss"]) for value in training),
        learning_rate,
    )


def _report_selection_key(report: dict[str, object]) -> tuple[float, ...]:
    value = report.get("selection_key")
    if not isinstance(value, list):
        raise ValueError("ReZero selection report key is invalid")
    return tuple(_as_float(item) for item in value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a numeric experiment value")
    return float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer experiment value")
    return value


def _validate_preregistration(
    preregistration: dict[str, Any], dataset: GovernedDataset, *, scope: str
) -> None:
    if preregistration.get("schema") != PREREGISTRATION_SCHEMA:
        raise ValueError("ReZero sequence preregistration schema is invalid")
    if preregistration["dataset"]["governed_dataset_id"] != dataset.manifest.dataset_id:
        raise ValueError("ReZero sequence dataset differs from preregistration")
    if scope not in {"fixture", "gemma"}:
        raise ValueError("ReZero sequence experiment scope is invalid")
    if scope == "gemma" and (
        preregistration["backbone"]["model_id"] != "google/gemma-4-E4B-it"
    ):
        raise ValueError("Gemma backbone preregistration is invalid")
    candidate = preregistration["candidate"]
    if (
        candidate["residual_strategy"] != "rezero_rms_shared"
        or candidate["sequence"] != "context_then_content_digest_ordered_evidence"
        or candidate["maximum_evidence_items"] != dataset.manifest.maximum_evidence_items
    ):
        raise ValueError("ReZero sequence candidate differs from its protocol")
    training = preregistration["training_search"]
    if tuple(training["selection"]["ranking"]) != SELECTION_RANKING:
        raise ValueError("ReZero sequence selection ranking differs from preregistration")
    learning_rates = training["candidate_learning_rates"]
    seeds = training["seeds"]
    if (
        not learning_rates
        or not seeds
        or len(learning_rates) != len(set(learning_rates))
        or len(seeds) != len(set(seeds))
        or any(float(value) <= 0 for value in learning_rates)
    ):
        raise ValueError("ReZero sequence search space cannot be empty")


def _require_unchanged(backbone: FrozenTextBackbone, expected: str) -> None:
    if backbone.state_fingerprint() != expected:
        raise RuntimeError("frozen backbone changed during ReZero sequence experiment")
