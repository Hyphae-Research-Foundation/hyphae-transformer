"""Immutable Lab runner for governed fixture experiments."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from celiums_rezero.governed.backbone import FixtureBackboneV1
from celiums_rezero.governed.data import load_governed_dataset
from celiums_rezero.governed.evaluation import evaluate_control_head
from celiums_rezero.governed.model import GovernedControlHead
from celiums_rezero.governed.schemas import GovernedDatasetManifest
from celiums_rezero.governed.trainer import ControlTrainConfig, train_control_head
from celiums_rezero.lab.budgets import BudgetTracker
from celiums_rezero.lab.registry import Registry
from celiums_rezero.lab.schemas import Metric, RunManifest, RunResult, RunStatus, Verdict


def run_registered_governed(
    registry: Registry,
    manifest: RunManifest,
    *,
    data_root: Path,
    dataset_manifest: GovernedDatasetManifest,
) -> RunResult:
    if manifest.config.get("runner") != "governed_control_v1":
        raise ValueError("manifest is not a governed control run")
    data_root = data_root.resolve(strict=True)
    for _, split in dataset_manifest.splits:
        path = (data_root / split.path).resolve(strict=True)
        if data_root not in path.parents or not path.is_file():
            raise ValueError("governed execution-time data root is invalid")
    registry.register_run(manifest)
    assert manifest.run_id is not None
    existing = registry.run_result(manifest.run_id)
    if existing is not None:
        from celiums_rezero.lab.runner import _result_from_dict
        restored = _result_from_dict(existing)
        if restored.status is RunStatus.FAILED:
            return restored
        checkpoint_metric = next(
            (metric for metric in restored.metrics if metric.name == "checkpoint_digest_prefix"),
            None,
        )
        checkpoint_path = registry.runs / manifest.run_id / "checkpoints" / "latest.pt"
        if checkpoint_metric is None or not checkpoint_path.is_file():
            raise RuntimeError("governed result is missing its checkpoint evidence")
        observed = float(int(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12], 16))
        if checkpoint_metric.value != observed:
            raise RuntimeError("governed checkpoint evidence changed")
        return restored
    started = datetime.now(UTC).isoformat()
    started_clock = time.perf_counter()
    try:
        if manifest.data_revision != dataset_manifest.dataset_id:
            raise ValueError("governed manifest data revision does not match dataset ID")
        dataset = load_governed_dataset(data_root, dataset_manifest)
        config = ControlTrainConfig(**manifest.config["training"])
        preregistration = manifest.config.get("preregistration")
        if not isinstance(preregistration, dict) or set(preregistration) != {"path", "sha256"}:
            raise ValueError("governed run requires a pinned preregistration")
        prereg_path = (data_root / str(preregistration["path"])).resolve(strict=True)
        if data_root.resolve(strict=True) not in prereg_path.parents:
            raise ValueError("preregistration path escaped data root")
        if hashlib.sha256(prereg_path.read_bytes()).hexdigest() != preregistration["sha256"]:
            raise ValueError("preregistration digest does not match")
        prereg = json.loads(prereg_path.read_text())
        if prereg.get("schema") != "governed-control-preregistration-v1":
            raise ValueError("preregistration schema is invalid")
        if config.seed != prereg["seed"] or {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "evidence_loss_weight": config.evidence_loss_weight,
            "gradient_clip": config.gradient_clip,
            "device": config.device,
        } != prereg["training"]:
            raise ValueError("training configuration differs from preregistration")
        gates = prereg.get("gates")
        if not isinstance(gates, dict):
            raise ValueError("preregistration gates are invalid")
        backbone = FixtureBackboneV1()
        import torch

        torch.manual_seed(config.seed)
        head = GovernedControlHead(backbone.hidden_size)
        checkpoint = registry.runs / manifest.run_id / "checkpoints" / "latest.pt"
        summary = train_control_head(
            backbone, head, dataset.train, config, checkpoint=checkpoint,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            deadline=started_clock + manifest.budget.max_wall_seconds,
        )
        if not summary.backbone_unchanged:
            raise RuntimeError("frozen backbone state changed during training")
        evaluation = evaluate_control_head(
            backbone, head, dataset.test,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            gates=gates,
        )
        adversarial = evaluate_control_head(
            backbone, head, dataset.adversarial,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            gates=gates,
        )
        elapsed = time.perf_counter() - started_clock
        BudgetTracker(manifest.budget).update(
            wall_seconds=elapsed, artifact_bytes=checkpoint.stat().st_size
        )
    except Exception as error:
        result = RunResult(
            run_id=manifest.run_id,
            status=RunStatus.FAILED,
            metrics=(),
            verdict=Verdict.INVALID,
            summary="Governed run failed validation or budget enforcement.",
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            failure=f"{type(error).__name__}: {error}",
        )
        registry.complete_run(result)
        return result
    passed = evaluation.passed and adversarial.passed
    result = RunResult(
        run_id=manifest.run_id,
        status=RunStatus.COMPLETED,
        metrics=(
            Metric("action_accuracy", evaluation.action_accuracy, direction="maximize"),
            Metric("answer_recall", evaluation.answer_recall, direction="maximize"),
            Metric("abstention_recall", evaluation.abstention_recall, direction="maximize"),
            Metric("evidence_exact_match", evaluation.evidence_exact_match, direction="maximize"),
            Metric("unsafe_answer_rate", evaluation.unsafe_answer_rate, direction="minimize"),
            Metric("final_loss", summary.final_loss, direction="minimize"),
            Metric(
                "adversarial_unsafe_answer_rate",
                adversarial.unsafe_answer_rate,
                direction="minimize",
            ),
            Metric("checkpoint_digest_prefix", float(int(summary.checkpoint_sha256[:12], 16))),
        ),
        verdict=Verdict.POSITIVE if passed else Verdict.NEGATIVE,
        summary=(
            "Governed fixture gates passed."
            if passed
            else "A governed gate failed."
        ),
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        artifacts=(str(checkpoint.relative_to(registry.root)),),
    )
    registry.complete_run(result)
    return result
