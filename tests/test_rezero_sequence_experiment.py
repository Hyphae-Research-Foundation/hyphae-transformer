from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from celiums_rezero.governed.backbone import FixtureBackboneV1
from celiums_rezero.governed.rezero_experiment import (
    SELECTION_RANKING,
    _selection_key,
    load_governed_dataset_directory,
    run_rezero_sequence_experiment,
    run_rezero_sequence_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments" / "governed" / "mars-v2-e4b-v1"
PREREGISTRATION = (
    ROOT / "experiments" / "canonical" / "gemma4_e4b_rezero_sequence_control_v1.json"
)


def fixture_preregistration() -> tuple[dict[str, object], str]:
    content = PREREGISTRATION.read_bytes()
    value = json.loads(content)
    value["candidate"].update(control_size=32, layers=1, n_heads=4)
    value["training_search"].update(
        candidate_learning_rates=[0.001, 0.01],
        epochs=2,
        seeds=[17, 29],
    )
    return value, hashlib.sha256(content).hexdigest()


def test_selection_key_uses_preregistered_order() -> None:
    assert SELECTION_RANKING[-1] == "minimum_learning_rate"
    first = [
        {
            "validation": {
                "passed": True,
                "unsafe_answer_rate": 0.0,
                "abstention_recall": 1.0,
                "evidence_exact_match": 0.9,
                "action_accuracy": 0.9,
            },
            "training": {"final_loss": 0.5},
        }
    ]
    second = [
        {
            "validation": {
                "passed": False,
                "unsafe_answer_rate": 0.0,
                "abstention_recall": 1.0,
                "evidence_exact_match": 1.0,
                "action_accuracy": 1.0,
            },
            "training": {"final_loss": 0.1},
        }
    ]
    assert _selection_key(0.01, first) < _selection_key(0.001, second)


def test_selection_key_breaks_exact_ties_with_lower_learning_rate() -> None:
    reports = [
        {
            "validation": {
                "passed": True,
                "unsafe_answer_rate": 0.0,
                "abstention_recall": 1.0,
                "evidence_exact_match": 1.0,
                "action_accuracy": 1.0,
            },
            "training": {"final_loss": 0.1},
        }
    ]
    assert _selection_key(0.001, reports) < _selection_key(0.01, reports)


def test_fixture_experiment_is_reproducible_and_keeps_backbone_frozen(
    tmp_path: Path,
) -> None:
    preregistration, digest = fixture_preregistration()
    dataset = load_governed_dataset_directory(
        DATASET,
        expected_dataset_id=str(preregistration["dataset"]["governed_dataset_id"]),
    )
    first = run_rezero_sequence_experiment(
        backbone=FixtureBackboneV1(),
        dataset=dataset,
        preregistration=preregistration,
        preregistration_sha256=digest,
        output=tmp_path / "first",
        device=torch.device("cpu"),
        feature_batch_size=64,
        scope="fixture",
    )
    second = run_rezero_sequence_experiment(
        backbone=FixtureBackboneV1(),
        dataset=dataset,
        preregistration=preregistration,
        preregistration_sha256=digest,
        output=tmp_path / "second",
        device=torch.device("cpu"),
        feature_batch_size=64,
        scope="fixture",
    )
    assert first["selected_learning_rate"] == second["selected_learning_rate"]
    assert first["backbone_unchanged"] is True
    assert second["backbone_unchanged"] is True
    assert [item["checkpoint_sha256"] for item in first["final"]] == [
        item["checkpoint_sha256"] for item in second["final"]
    ]
    assert len(first["search"]) == 2
    assert len(first["final"]) == 2


def test_fixture_smoke_checks_gates_optimizer_and_parameter_bound(tmp_path: Path) -> None:
    preregistration, digest = fixture_preregistration()
    dataset = load_governed_dataset_directory(
        DATASET,
        expected_dataset_id=str(preregistration["dataset"]["governed_dataset_id"]),
    )
    report = run_rezero_sequence_smoke(
        backbone=FixtureBackboneV1(),
        dataset=dataset,
        preregistration=preregistration,
        preregistration_sha256=digest,
        output=tmp_path / "smoke.json",
        device=torch.device("cpu"),
        feature_batch_size=8,
        maximum_vram_gib=1,
    )
    assert report["passed"] is True
    assert report["backbone_unchanged"] is True
    assert report["gate_gradients_finite"] is True
    assert report["gates_excluded_from_weight_decay"] is True
    assert int(report["parameters"]) < 5_000_000


def test_fixture_experiment_rejects_preregistration_drift(tmp_path: Path) -> None:
    preregistration, digest = fixture_preregistration()
    preregistration["training_search"]["selection"]["ranking"] = [
        "minimum_learning_rate"
    ]
    dataset = load_governed_dataset_directory(
        DATASET,
        expected_dataset_id=str(preregistration["dataset"]["governed_dataset_id"]),
    )
    with pytest.raises(ValueError, match="selection ranking"):
        run_rezero_sequence_experiment(
            backbone=FixtureBackboneV1(),
            dataset=dataset,
            preregistration=preregistration,
            preregistration_sha256=digest,
            output=tmp_path / "drift",
            device=torch.device("cpu"),
            feature_batch_size=64,
            scope="fixture",
        )


def test_v2_preregistration_uses_policy_certificate_without_shadow_data() -> None:
    value = json.loads(
        (
            ROOT
            / "experiments"
            / "canonical"
            / "gemma4_e4b_rezero_sequence_control_v2.json"
        ).read_text()
    )
    assert value["candidate"]["host_control_contract"] == "host-policy-certificate-v2"
    assert value["candidate"]["action_policy_prior_scale"] == 20.0
    assert "shadow cases" in value["scope"].lower()
    assert value["structural_gates"]["host_policy_certificate_size"] == 17


def test_rezero_smoke_uses_active_device_for_rocm_memory_stats() -> None:
    source = (ROOT / "scripts" / "smoke_gemma4_e4b_rezero_control.py").read_text()
    experiment = (
        ROOT / "src" / "celiums_rezero" / "governed" / "rezero_experiment.py"
    ).read_text()
    assert "reset_peak_memory_stats()" in source
    assert "reset_peak_memory_stats(device)" not in source
    assert "max_memory_allocated()" in experiment
    assert "max_memory_allocated(device)" not in experiment
