from __future__ import annotations

from pathlib import Path

import pytest

from celiums_rezero.lab.budgets import BudgetExceeded, BudgetTracker
from celiums_rezero.lab.campaign import render_campaign_report, summarize_campaign
from celiums_rezero.lab.evaluator import compare_means, compare_paired, student_t_interval
from celiums_rezero.lab.registry import Registry
from celiums_rezero.lab.report import render_run_report
from celiums_rezero.lab.schemas import (
    Budget,
    Hypothesis,
    MemoryEntry,
    MemoryRelation,
    Metric,
    MetricCurve,
    MetricPoint,
    RunManifest,
    RunResult,
    RunStage,
    RunStatus,
    Verdict,
)
from celiums_rezero.lab.search import TrialScore, successive_halving
from celiums_rezero.lab.serialization import read_json


def hypothesis() -> Hypothesis:
    return Hypothesis(
        claim="Split gates improve optimization.",
        baseline="rezero_canonical",
        candidate="crz_rms",
        context={"depth": 24, "dataset": "synthetic"},
        independent_variables=("residual_strategy",),
        dependent_variables=("loss",),
        prediction="candidate_better",
        minimum_effect=0.1,
        falsification=("No minimum effect across three seeds.",),
        budget=Budget(max_wall_seconds=60),
    )


def test_hypothesis_and_run_ids_are_content_stable() -> None:
    first = hypothesis()
    second = hypothesis()
    assert first.hypothesis_id == second.hypothesis_id
    assert first.hypothesis_id is not None
    manifest_a = RunManifest(
        hypothesis_id=first.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=3,
        config={"runner": "synthetic_v1", "training": {"seed": 3}, "x": 1},
        budget=first.budget,
    )
    manifest_b = RunManifest(
        hypothesis_id=first.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=3,
        config={"runner": "synthetic_v1", "training": {"seed": 3}, "x": 1},
        budget=first.budget,
    )
    assert manifest_a.run_id == manifest_b.run_id
    values = {
        "hypothesis_id": manifest_a.hypothesis_id,
        "stage": manifest_a.stage.value,
        "seed": manifest_a.seed,
        "config": manifest_a.config,
        "budget": {
            "max_wall_seconds": manifest_a.budget.max_wall_seconds,
            "max_device_hours": manifest_a.budget.max_device_hours,
            "max_cost_usd": manifest_a.budget.max_cost_usd,
            "max_failures": manifest_a.budget.max_failures,
            "max_artifact_bytes": manifest_a.budget.max_artifact_bytes,
        },
        "run_id": manifest_a.run_id,
        "created_at": manifest_a.created_at,
        "schema_version": 1,
    }
    assert RunManifest.from_dict(values) == manifest_a
    values["run_id"] = "R-0000000000000000"
    with pytest.raises(ValueError, match="run_id"):
        RunManifest.from_dict(values)


def test_registry_is_idempotent_but_immutable(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    item = hypothesis()
    path = registry.register_hypothesis(item)
    assert registry.register_hypothesis(item) == path
    values = read_json(path)
    values["claim"] = "changed"
    path.write_text(__import__("json").dumps(values))
    with pytest.raises(FileExistsError):
        registry.register_hypothesis(item)


def test_registry_records_result_memory_and_report(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    item = hypothesis()
    registry.register_hypothesis(item)
    assert item.hypothesis_id is not None
    manifest = RunManifest(
        hypothesis_id=item.hypothesis_id,
        stage=RunStage.MINI_PILOT,
        seed=3,
        config={"runner": "synthetic_v1", "training": {"seed": 3}, "x": 1},
        budget=item.budget,
    )
    registry.register_run(manifest)
    assert manifest.run_id is not None
    result = RunResult(
        run_id=manifest.run_id,
        status=RunStatus.COMPLETED,
        metrics=(Metric("loss", 1.0, "nats", "minimize"),),
        verdict=Verdict.INCONCLUSIVE,
        summary="Smoke completed.",
        started_at="start",
        finished_at="finish",
    )
    registry.complete_run(result)
    entry = MemoryEntry(
        statement="This configuration is finite.",
        relation=MemoryRelation.HELPS,
        conditions={"x": 1},
        evidence_run_ids=(manifest.run_id,),
        confidence=0.6,
    )
    registry.add_memory(entry)
    report = render_run_report(
        registry.runs / manifest.run_id,
        manifest=registry.run_manifest(manifest.run_id),
        result=registry.run_result(manifest.run_id) or {},
    )
    assert all(path.exists() for path in report)


def test_budget_tracker_stops_overspend() -> None:
    tracker = BudgetTracker(Budget(max_wall_seconds=1, max_failures=0))
    with pytest.raises(BudgetExceeded):
        tracker.update(wall_seconds=1.1)


def test_objective_comparison_and_halving() -> None:
    comparison = compare_means(
        [10.0, 10.0], [8.0, 8.0], direction="minimize", minimum_effect=0.1
    )
    assert comparison.verdict is Verdict.POSITIVE
    survivors = successive_halving(
        [TrialScore("a", 0.5, 0.1), TrialScore("b", 0.4, 0.3)]
    )
    assert [score.name for score in survivors] == ["a"]


def test_paired_comparison_reports_student_t_uncertainty() -> None:
    interval = student_t_interval([0.1, 0.2, 0.3])
    assert interval is not None
    assert interval.lower == pytest.approx(-0.0484138, abs=1e-6)
    assert interval.upper == pytest.approx(0.4484138, abs=1e-6)
    comparison = compare_paired(
        [5.0, 5.0, 5.0],
        [4.0, 4.0, 4.0],
        direction="minimize",
        minimum_effect=0.1,
    )
    assert comparison.verdict is Verdict.POSITIVE
    assert comparison.paired_count == 3
    assert comparison.relative_effect_confidence_interval is not None


def test_metric_curve_requires_monotonic_coordinates() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MetricCurve(
            "validation_nll",
            "nats",
            "minimize",
            (MetricPoint(2, 20, 1.0), MetricPoint(1, 10, 2.0)),
        )


def test_campaign_aggregates_equal_seed_runs_and_renders_report(tmp_path: Path) -> None:
    records: list[tuple[RunManifest, RunResult]] = []
    for strategy, values in {"pre_rms": [5.0, 5.2], "crz_rms": [4.0, 4.2]}.items():
        for seed, value in enumerate(values):
            manifest = RunManifest(
                    hypothesis_id="H-0123456789ab",
                stage=RunStage.PILOT,
                seed=seed,
                config={
                    "runner": "synthetic_v1",
                    "model": {"residual_strategy": strategy},
                    "training": {"seed": seed, "steps": 10},
                },
                budget=Budget(max_wall_seconds=60),
            )
            assert manifest.run_id is not None
            result = RunResult(
                run_id=manifest.run_id,
                status=RunStatus.COMPLETED,
                metrics=(
                    Metric("validation_nll", value, "nats", "minimize"),
                    Metric("tokens_per_second", 100.0, "tokens/s", "maximize"),
                    Metric("peak_memory_bytes", 1024.0, "bytes", "minimize"),
                ),
                verdict=Verdict.INCONCLUSIVE,
                summary="complete",
                started_at="start",
                finished_at="finish",
            )
            records.append((manifest, result))

    summary = summarize_campaign(
        records,
        baseline="pre_rms",
        candidate="crz_rms",
        minimum_effect=0.1,
    )
    assert summary.candidate_verdict is Verdict.POSITIVE
    assert summary.aggregates[0].completed_seeds == 2
    assert summary.comparisons[0].relative_effect > 0.1
    assert all(path.exists() for path in render_campaign_report(tmp_path, summary))


def test_campaign_rejects_unequal_seeds() -> None:
    records: list[tuple[RunManifest, RunResult]] = []
    for strategy, seed in (("pre_rms", 1), ("crz_rms", 2)):
        manifest = RunManifest(
                hypothesis_id="H-0123456789ab",
            stage=RunStage.PILOT,
            seed=seed,
            config={
                "runner": "synthetic_v1",
                "model": {"residual_strategy": strategy},
                "training": {"seed": seed, "steps": 10},
            },
            budget=Budget(max_wall_seconds=60),
        )
        assert manifest.run_id is not None
        records.append(
            (
                manifest,
                RunResult(
                    run_id=manifest.run_id,
                    status=RunStatus.COMPLETED,
                    metrics=(Metric("validation_nll", 1.0),),
                    verdict=Verdict.INCONCLUSIVE,
                    summary="complete",
                    started_at="start",
                    finished_at="finish",
                ),
            )
        )
    with pytest.raises(ValueError, match="same seeds"):
        summarize_campaign(
            records,
            baseline="pre_rms",
            candidate="crz_rms",
            minimum_effect=0.1,
        )


def test_campaign_rejects_unequal_budgets() -> None:
    records: list[tuple[RunManifest, RunResult]] = []
    for strategy, steps in (("pre_rms", 10), ("crz_rms", 20)):
        manifest = RunManifest(
                hypothesis_id="H-0123456789ab",
            stage=RunStage.PILOT,
            seed=1,
            config={
                "runner": "synthetic_v1",
                "model": {"residual_strategy": strategy},
                "training": {"seed": 1, "steps": steps},
            },
            budget=Budget(max_wall_seconds=60),
        )
        assert manifest.run_id is not None
        records.append(
            (
                manifest,
                RunResult(
                    run_id=manifest.run_id,
                    status=RunStatus.COMPLETED,
                    metrics=(Metric("validation_nll", 1.0),),
                    verdict=Verdict.INCONCLUSIVE,
                    summary="complete",
                    started_at="start",
                    finished_at="finish",
                ),
            )
        )
    with pytest.raises(ValueError, match="equal-budget"):
        summarize_campaign(
            records,
            baseline="pre_rms",
            candidate="crz_rms",
            minimum_effect=0.1,
        )
