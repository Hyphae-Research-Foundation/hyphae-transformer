"""Aggregate equal-budget runs into a reproducible multi-seed campaign report."""

from __future__ import annotations

import html
import json
import statistics
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from celiums_rezero.lab.evaluator import Comparison, compare_means
from celiums_rezero.lab.schemas import RunManifest, RunResult, RunStatus, Verdict
from celiums_rezero.lab.serialization import canonical_json, content_hash, to_primitive, write_json


@dataclass(frozen=True, slots=True)
class RunObservation:
    strategy: str
    seed: int
    run_id: str
    status: RunStatus
    metric_value: float | None
    tokens_per_second: float | None
    peak_memory_bytes: float | None
    failure: str | None


@dataclass(frozen=True, slots=True)
class StrategyAggregate:
    strategy: str
    observations: tuple[RunObservation, ...]
    completed_seeds: int
    failed_seed_rate: float
    metric_mean: float | None
    metric_stdev: float | None
    tokens_per_second_mean: float | None
    peak_memory_bytes_max: float | None


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    strategy: str
    baseline: str
    baseline_mean: float
    strategy_mean: float
    relative_effect: float
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class CampaignSummary:
    campaign_id: str
    baseline: str
    candidate: str
    metric: str
    minimum_effect: float
    aggregates: tuple[StrategyAggregate, ...]
    comparisons: tuple[StrategyComparison, ...]
    candidate_verdict: Verdict


def _metric(result: RunResult, name: str) -> float | None:
    return next((metric.value for metric in result.metrics if metric.name == name), None)


def summarize_campaign(
    records: list[tuple[RunManifest, RunResult]],
    *,
    baseline: str,
    candidate: str,
    metric: str = "validation_nll",
    minimum_effect: float,
) -> CampaignSummary:
    if not records:
        raise ValueError("campaign requires at least one run")
    observations: list[RunObservation] = []
    identities: set[tuple[str, int]] = set()
    run_contract: str | None = None
    for manifest, result in records:
        model = manifest.config.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("residual_strategy"), str):
            raise TypeError("run manifest is missing a residual strategy")
        strategy = model["residual_strategy"]
        condition = (strategy, manifest.seed)
        if condition in identities:
            raise ValueError(f"duplicate campaign condition: {strategy}, seed {manifest.seed}")
        identities.add(condition)
        contract = _run_contract(manifest)
        if run_contract is None:
            run_contract = contract
        elif contract != run_contract:
            raise ValueError("campaign runs must share an equal-budget configuration")
        observations.append(
            RunObservation(
                strategy=strategy,
                seed=manifest.seed,
                run_id=result.run_id,
                status=result.status,
                metric_value=_metric(result, metric),
                tokens_per_second=_metric(result, "tokens_per_second"),
                peak_memory_bytes=_metric(result, "peak_memory_bytes"),
                failure=result.failure,
            )
        )

    aggregates: list[StrategyAggregate] = []
    for strategy in sorted({observation.strategy for observation in observations}):
        items = tuple(
            sorted(
                (item for item in observations if item.strategy == strategy),
                key=lambda item: item.seed,
            )
        )
        completed = [
            item
            for item in items
            if item.status is RunStatus.COMPLETED and item.metric_value is not None
        ]
        values = [item.metric_value for item in completed if item.metric_value is not None]
        throughputs = [
            item.tokens_per_second
            for item in completed
            if item.tokens_per_second is not None
        ]
        peaks = [
            item.peak_memory_bytes
            for item in completed
            if item.peak_memory_bytes is not None
        ]
        aggregates.append(
            StrategyAggregate(
                strategy=strategy,
                observations=items,
                completed_seeds=len(completed),
                failed_seed_rate=1 - len(completed) / len(items),
                metric_mean=statistics.fmean(values) if values else None,
                metric_stdev=(
                    statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
                ),
                tokens_per_second_mean=statistics.fmean(throughputs) if throughputs else None,
                peak_memory_bytes_max=max(peaks) if peaks else None,
            )
        )

    by_strategy = {aggregate.strategy: aggregate for aggregate in aggregates}
    if baseline not in by_strategy or candidate not in by_strategy:
        raise ValueError("baseline and candidate must both be present in the campaign")
    baseline_seeds = {item.seed for item in by_strategy[baseline].observations}
    if any(
        {item.seed for item in aggregate.observations} != baseline_seeds
        for aggregate in aggregates
    ):
        raise ValueError("all campaign strategies must use the same seeds")
    baseline_values = _completed_values(by_strategy[baseline])
    comparisons: list[StrategyComparison] = []
    for aggregate in aggregates:
        values = _completed_values(aggregate)
        if aggregate.strategy == baseline or not baseline_values or not values:
            continue
        comparison: Comparison = compare_means(
            baseline_values,
            values,
            direction="minimize",
            minimum_effect=minimum_effect,
        )
        comparisons.append(
            StrategyComparison(
                strategy=aggregate.strategy,
                baseline=baseline,
                baseline_mean=comparison.baseline_mean,
                strategy_mean=comparison.candidate_mean,
                relative_effect=comparison.relative_effect,
                verdict=comparison.verdict,
            )
        )
    candidate_comparison = next(
        (comparison for comparison in comparisons if comparison.strategy == candidate),
        None,
    )
    candidate_verdict = Verdict.INVALID
    if (
        candidate_comparison is not None
        and by_strategy[baseline].failed_seed_rate == 0
        and by_strategy[candidate].failed_seed_rate == 0
    ):
        candidate_verdict = candidate_comparison.verdict
    campaign_identity = {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "minimum_effect": minimum_effect,
        "run_ids": sorted(observation.run_id for observation in observations),
    }
    return CampaignSummary(
        campaign_id=f"C-{content_hash(campaign_identity)}",
        baseline=baseline,
        candidate=candidate,
        metric=metric,
        minimum_effect=minimum_effect,
        aggregates=tuple(aggregates),
        comparisons=tuple(comparisons),
        candidate_verdict=candidate_verdict,
    )


def _completed_values(aggregate: StrategyAggregate) -> list[float]:
    return [
        observation.metric_value
        for observation in aggregate.observations
        if observation.status is RunStatus.COMPLETED
        and observation.metric_value is not None
    ]


def _run_contract(manifest: RunManifest) -> str:
    config = deepcopy(manifest.config)
    model = config.get("model")
    training = config.get("training")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise TypeError("campaign run is missing model or training configuration")
    model.pop("residual_strategy", None)
    training.pop("seed", None)
    return canonical_json(
        {
            "stage": manifest.stage,
            "config": config,
            "budget": manifest.budget,
            "code_revision": manifest.code_revision,
            "data_revision": manifest.data_revision,
        }
    )


def render_campaign_report(
    output_root: Path,
    summary: CampaignSummary,
) -> tuple[Path, Path]:
    output_directory = output_root / "campaigns" / summary.campaign_id
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "campaign.json"
    html_path = output_directory / "report.html"
    write_json(json_path, summary)

    comparison_by_strategy = {
        comparison.strategy: comparison for comparison in summary.comparisons
    }
    rows = []
    for aggregate in summary.aggregates:
        comparison = comparison_by_strategy.get(aggregate.strategy)
        effect = None if comparison is None else comparison.relative_effect
        verdict = "baseline" if comparison is None else comparison.verdict.value
        rows.append(
            "<tr>"
            f"<td>{html.escape(aggregate.strategy)}</td>"
            f"<td>{aggregate.completed_seeds}/{len(aggregate.observations)}</td>"
            f"<td>{_format_number(aggregate.metric_mean)}</td>"
            f"<td>{_format_number(aggregate.metric_stdev)}</td>"
            f"<td>{_format_percent(effect)}</td>"
            f"<td>{html.escape(verdict)}</td>"
            "</tr>"
        )
    payload = json.dumps(to_primitive(summary), ensure_ascii=True, sort_keys=True, indent=2)
    document = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Celiums ReZero Campaign {html.escape(summary.campaign_id)}</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1080px; margin: 3rem auto;
        padding: 0 1rem; color: #16211d; }}
h1, h2 {{ color: #074f3b; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #b9c7c1; padding: .55rem; text-align: left; }}
pre {{ background: #eef4f1; padding: 1rem; overflow: auto; }}
</style>
<h1>Celiums ReZero Campaign</h1>
<p><strong>Campaign:</strong> {html.escape(summary.campaign_id)}</p>
<p><strong>Candidate verdict:</strong> {html.escape(summary.candidate_verdict.value)}</p>
<p>Metric: {html.escape(summary.metric)}; baseline: {html.escape(summary.baseline)};
minimum relative effect: {summary.minimum_effect:.1%}.</p>
<h2>Aggregate Results</h2>
<table><thead><tr><th>Strategy</th><th>Completed</th><th>Mean</th><th>Stdev</th>
<th>Effect vs baseline</th><th>Verdict</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Evidence</h2>
<pre>{html.escape(payload)}</pre>
</html>
"""
    html_path.write_text(document)
    return json_path, html_path


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"
