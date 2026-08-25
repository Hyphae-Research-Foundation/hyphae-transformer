from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from celiums_rezero.governed.backbone import FixtureBackboneV1
from celiums_rezero.governed.data import load_governed_dataset
from celiums_rezero.governed.hyphaelm import ReZeroNeuroPilot
from celiums_rezero.governed.navigation import (
    NAVIGATION_ACTIONS,
    SEARCH_BOUND,
    action_labels,
    derive_navigation_dataset,
    search_decision_recall,
)
from celiums_rezero.governed.navigation_experiment import (
    build_navigation_batch,
    run_navigation_experiment,
)
from celiums_rezero.governed.schemas import (
    DatasetSplit,
    GovernedDatasetManifest,
)
from celiums_rezero.knowledge.schemas import SufficiencyPolicy

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments" / "governed" / "mars-v2-e4b-v1"
PREREG = ROOT / "experiments" / "canonical" / "gemma4_e4b_rezero_navigation_v1.json"


def loaded_dataset():
    import json

    manifest_values = json.loads((DATASET / "manifest.json").read_text())
    manifest = GovernedDatasetManifest(
        splits=tuple(
            (name, DatasetSplit(**item)) for name, item in sorted(manifest_values["splits"].items())
        ),
        policy=SufficiencyPolicy(**manifest_values["policy"]),
        maximum_evidence_items=manifest_values["maximum_evidence_items"],
    )
    return load_governed_dataset(DATASET, manifest)


def test_derive_navigation_bounds_search_steps() -> None:
    dataset = loaded_dataset()
    trajectories = derive_navigation_dataset(
        dataset, SufficiencyPolicy(), provenance="test", search_bound=SEARCH_BOUND
    )
    assert len(trajectories) == (
        len(dataset.train) + len(dataset.validation) + len(dataset.test) + len(dataset.adversarial)
    )
    for trajectory in trajectories:
        search_steps = sum(1 for step in trajectory if step.step_action == "search")
        assert search_steps <= SEARCH_BOUND
        assert trajectory[-1].step_action in {"answer", "abstain"}
        assert all(step.search_steps_used <= SEARCH_BOUND for step in trajectory)


def test_search_recall_counts_search_decisions() -> None:
    dataset = loaded_dataset()
    strict_policy = SufficiencyPolicy(minimum_score=0.99)
    trajectories = derive_navigation_dataset(dataset, strict_policy, provenance="test")
    assert any(step.step_action == "search" for trajectory in trajectories for step in trajectory)
    perfect = action_labels(trajectories)
    assert search_decision_recall(perfect, trajectories) == 1.0
    wrong = tuple(
        tuple("abstain" if action == "search" else action for action in trajectory)
        for trajectory in perfect
    )
    assert search_decision_recall(wrong, trajectories) == 0.0


def test_navigation_action_space_is_fixed() -> None:
    assert NAVIGATION_ACTIONS == ("search", "answer", "abstain")


def test_navigation_batch_builds_host_certificates() -> None:
    dataset = loaded_dataset()
    trajectories = derive_navigation_dataset(dataset, SufficiencyPolicy(), provenance="test")
    steps = trajectories[0]
    backbone = FixtureBackboneV1()
    batch = build_navigation_batch(
        steps,
        backbone,
        maximum_evidence_items=dataset.manifest.maximum_evidence_items,
        device=torch.device("cpu"),
    )
    assert batch.host_control.shape == (len(steps), 17)
    assert batch.action_targets.max() <= 2
    assert batch.context.shape[1] == 128


def test_navigation_experiment_runs_bounded_and_deterministic(tmp_path: Path) -> None:
    import json

    dataset = loaded_dataset()
    trajectories = derive_navigation_dataset(dataset, SufficiencyPolicy(), provenance="test")
    prereg = json.loads(PREREG.read_text())
    prereg["training_search"]["epochs"] = 2
    prereg["training_search"]["candidate_learning_rates"] = [0.01]
    backbone = FixtureBackboneV1()
    report = run_navigation_experiment(
        backbone=backbone,
        preregistration=prereg,
        preregistration_sha256=hashlib.sha256(
            json.dumps(prereg, sort_keys=True).encode()
        ).hexdigest(),
        dataset=dataset,
        trajectories=trajectories,
        output=tmp_path,
        device=torch.device("cpu"),
        maximum_evidence_items=dataset.manifest.maximum_evidence_items,
    )
    assert report["schema"] == ("hyphae-transformer.gemma4-e4b-rezero-navigation-experiment/v1")
    assert report["backbone_unchanged"] is True
    assert (tmp_path / "rezero-navigation-report.json").is_file()


def test_pilot_residual_bounds_cannot_reverse_certificates() -> None:
    pilot = ReZeroNeuroPilot(8, control_size=16, n_layers=1, n_heads=4, maximum_evidence_items=2)
    with torch.no_grad():
        pilot.action.weight.fill_(-1000)
        pilot.pointer.weight.fill_(-1000)
    certificate = torch.zeros((1, 17))
    certificate[0, -5] = 1
    action_logits, pointers = pilot(
        torch.ones((1, 8)),
        torch.ones((1, 2, 8)),
        torch.tensor([[True, True]]),
        torch.tensor([[0.95, 0.4]]),
        certificate,
    )
    assert action_logits.argmax(-1).item() == 1
    assert (pointers >= 0).tolist() == [[True, False]]
