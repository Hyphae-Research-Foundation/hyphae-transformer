#!/usr/bin/env python3
"""Run the preregistered three-seed frozen Gemma E4B control-head campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from celiums_rezero.governed.data import (
    GovernedBatch,
    load_governed_dataset,
    materialize_governed_batch,
)
from celiums_rezero.governed.evaluation import ControlEvaluation, evaluate_control_head
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.model import GovernedControlHead
from celiums_rezero.governed.schemas import (
    DatasetSplit,
    GovernedDataset,
    GovernedDatasetManifest,
)
from celiums_rezero.governed.trainer import ControlTrainConfig, train_control_head
from celiums_rezero.knowledge.schemas import SufficiencyPolicy

EXPECTED_DATASET_ID = "gtd_b7161eb4c1cf007dca96741ad8acffbe25cd9b6b46681fd48f19f21fde29332f"
EXPECTED_PREREGISTRATION_SCHEMA = (
    "hyphae-transformer.gemma4-e4b-governed-control-preregistration/v1"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--evidence-loss-weight", type=float, required=True)
    parser.add_argument("--gradient-clip", type=float, required=True)
    parser.add_argument("--feature-batch-size", type=int, required=True)
    parser.add_argument("--pointer-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.7)
    arguments = parser.parse_args()
    report = run_campaign(
        model=arguments.model,
        dataset_root=arguments.dataset,
        preregistration_path=arguments.preregistration,
        output=arguments.out,
        seeds=tuple(arguments.seeds),
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        evidence_loss_weight=arguments.evidence_loss_weight,
        gradient_clip=arguments.gradient_clip,
        feature_batch_size=arguments.feature_batch_size,
        pointer_threshold=arguments.pointer_threshold,
        minimum_confidence=arguments.minimum_confidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def run_campaign(
    *,
    model: Path,
    dataset_root: Path,
    preregistration_path: Path,
    output: Path,
    seeds: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    evidence_loss_weight: float,
    gradient_clip: float,
    feature_batch_size: int,
    pointer_threshold: float,
    minimum_confidence: float,
) -> dict[str, object]:
    preregistration = json.loads(preregistration_path.read_text())
    if preregistration.get("schema") not in {
        EXPECTED_PREREGISTRATION_SCHEMA,
        "hyphae-transformer.gemma4-e4b-governed-control-preregistration/v2",
        "hyphae-transformer.gemma4-e4b-governed-control-preregistration/v3",
    }:
        raise ValueError("Gemma preregistration schema is invalid")
    training = preregistration["training"]
    expected = {
        "seeds": list(seeds),
        "epochs": epochs,
        "head_learning_rate": learning_rate,
        "evidence_loss_weight": evidence_loss_weight,
        "gradient_clip": gradient_clip,
    }
    if any(training.get(name) != value for name, value in expected.items()):
        raise ValueError("Gemma training settings differ from preregistration")
    if feature_batch_size != max(training["batch_size_smoke"]):
        raise ValueError("feature batch size was not validated by the smoke")
    recipe = str(preregistration["schema"]).rsplit("/", 1)[-1]
    v2 = recipe == "v2"
    if v2 and (
        training.get("pointer_threshold") != pointer_threshold
        or training.get("minimum_confidence") != minimum_confidence
    ):
        raise ValueError("Gemma decode settings differ from preregistration")
    gates = preregistration["gates"]
    if set(gates) != {
        "action_accuracy",
        "answer_recall",
        "abstention_recall",
        "evidence_exact_match",
        "unsafe_answer_rate",
    }:
        raise ValueError("Gemma preregistration gate set is invalid")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("Gemma training requires ROCm")
    dataset = _load_dataset(dataset_root)
    backbone = Gemma4E4BFrozenBackbone(model, device=str(device))
    backbone_before = backbone.state_fingerprint()
    torch.cuda.reset_peak_memory_stats(device)
    feature_started = time.perf_counter()
    batches: dict[str, GovernedBatch] = {
        "train": materialize_governed_batch(
            dataset.train,
            backbone,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            feature_batch_size=feature_batch_size,
            device=device,
        ),
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
    feature_seconds = time.perf_counter() - feature_started
    if backbone.state_fingerprint() != backbone_before:
        raise RuntimeError("Gemma backbone changed during feature materialization")

    output.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    for seed in seeds:
        torch.manual_seed(seed)
        head = GovernedControlHead(
            backbone.hidden_size,
            pointer_rank=int(training.get("pointer_rank", 32)),
            normalized_features=bool(training.get("normalized_features", False)),
            use_evidence_scores=bool(training.get("use_evidence_scores", False)),
            pointer_policy_score=(
                None
                if training.get("pointer_policy_score") is None
                else float(training["pointer_policy_score"])
            ),
            pointer_policy_scale=float(training.get("pointer_policy_scale", 1.0)),
            use_host_control_features=bool(
                training.get("use_host_control_features", False)
            ),
        ).to(device)
        config = ControlTrainConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            evidence_loss_weight=evidence_loss_weight,
            gradient_clip=gradient_clip,
            seed=seed,
            device=str(device),
            optimizer=str(training.get("optimizer", "adamw")),
            weight_decay=float(training.get("weight_decay", 0.01)),
            pointer_loss_scope=str(training.get("pointer_loss_scope", "all")),
        )
        checkpoint = output / f"seed-{seed}" / "control-head.pt"
        started = time.perf_counter()
        summary = train_control_head(
            backbone,
            head,
            dataset.train,
            config,
            checkpoint=checkpoint,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            batch=batches["train"],
        )
        test = evaluate_control_head(
            backbone,
            head,
            dataset.test,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            gates=gates,
            batch=batches["test"],
            minimum_confidence=minimum_confidence,
            pointer_threshold=pointer_threshold,
        )
        adversarial = evaluate_control_head(
            backbone,
            head,
            dataset.adversarial,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            gates=gates,
            batch=batches["adversarial"],
            minimum_confidence=minimum_confidence,
            pointer_threshold=pointer_threshold,
        )
        seed_report = {
            "seed": seed,
            "training": asdict(summary),
            "test": _evaluation(test),
            "adversarial": _evaluation(adversarial),
            "wall_seconds": time.perf_counter() - started,
            "passed": summary.backbone_unchanged and test.passed and adversarial.passed,
        }
        (checkpoint.parent / "report.json").write_text(
            json.dumps(seed_report, indent=2, sort_keys=True) + "\n"
        )
        seed_reports.append(seed_report)
    backbone_unchanged = backbone.state_fingerprint() == backbone_before
    report = {
        "schema": "hyphae-transformer.gemma4-e4b-training/v1",
        "completed": True,
        "passed": backbone_unchanged and all(item["passed"] for item in seed_reports),
        "model_id": backbone.model_id,
        "model_revision": backbone.revision,
        "dataset_id": dataset.manifest.dataset_id,
        "preregistration_sha256": hashlib.sha256(
            preregistration_path.read_bytes()
        ).hexdigest(),
        "feature_batch_size": feature_batch_size,
        "recipe": recipe,
        "feature_materialization_seconds": feature_seconds,
        "backbone_unchanged": backbone_unchanged,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_architecture": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "seeds": seed_reports,
    }
    (output / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _load_dataset(root: Path) -> GovernedDataset:
    values = json.loads((root / "manifest.json").read_text())
    if values.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("Gemma training dataset ID differs from preregistration")
    manifest = GovernedDatasetManifest(
        splits=tuple(
            (name, DatasetSplit(**item)) for name, item in sorted(values["splits"].items())
        ),
        policy=SufficiencyPolicy(**values["policy"]),
        maximum_evidence_items=values["maximum_evidence_items"],
        dataset_id=values["dataset_id"],
    )
    return load_governed_dataset(root, manifest)


def _evaluation(value: ControlEvaluation) -> dict[str, object]:
    return asdict(value)


if __name__ == "__main__":
    raise SystemExit(main())
