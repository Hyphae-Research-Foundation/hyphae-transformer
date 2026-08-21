"""Objective paired and unpaired rules for preregistered architecture hypotheses."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from celiums_rezero.lab.schemas import Verdict


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    level: float
    lower: float
    upper: float
    method: str = "student_t"


@dataclass(frozen=True, slots=True)
class Comparison:
    baseline_mean: float
    candidate_mean: float
    relative_effect: float
    minimum_effect: float
    direction: str
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class PairedComparison(Comparison):
    paired_count: int = 0
    signed_effect_mean: float = 0.0
    relative_effect_confidence_interval: ConfidenceInterval | None = None


def compare_means(
    baseline: list[float],
    candidate: list[float],
    *,
    direction: str,
    minimum_effect: float,
) -> Comparison:
    if not baseline or not candidate:
        raise ValueError("both conditions require observations")
    _validate_comparison(direction, minimum_effect)
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    signed = _signed_effect(baseline_mean, candidate_mean, direction)
    relative_effect = signed / max(abs(baseline_mean), 1e-12)
    verdict = _point_verdict(relative_effect, minimum_effect)
    return Comparison(
        baseline_mean,
        candidate_mean,
        relative_effect,
        minimum_effect,
        direction,
        verdict,
    )


def compare_paired(
    baseline: list[float],
    candidate: list[float],
    *,
    direction: str,
    minimum_effect: float,
) -> PairedComparison:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired conditions require equal non-empty observations")
    _validate_comparison(direction, minimum_effect)
    baseline_mean = statistics.fmean(baseline)
    candidate_mean = statistics.fmean(candidate)
    effects = [
        _signed_effect(base, item, direction)
        for base, item in zip(baseline, candidate, strict=True)
    ]
    effect_mean = statistics.fmean(effects)
    denominator = max(abs(baseline_mean), 1e-12)
    relative_effect = effect_mean / denominator
    effect_interval = student_t_interval(effects)
    relative_interval = (
        None
        if effect_interval is None
        else ConfidenceInterval(
            level=effect_interval.level,
            lower=effect_interval.lower / denominator,
            upper=effect_interval.upper / denominator,
        )
    )
    verdict = Verdict.INCONCLUSIVE
    if relative_interval is not None:
        if relative_interval.lower >= minimum_effect:
            verdict = Verdict.POSITIVE
        elif relative_interval.upper <= -minimum_effect:
            verdict = Verdict.NEGATIVE
    return PairedComparison(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        relative_effect=relative_effect,
        minimum_effect=minimum_effect,
        direction=direction,
        verdict=verdict,
        paired_count=len(effects),
        signed_effect_mean=effect_mean,
        relative_effect_confidence_interval=relative_interval,
    )


def student_t_interval(values: list[float]) -> ConfidenceInterval | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    if deviation == 0:
        return ConfidenceInterval(0.95, mean, mean)
    critical = _t_critical(len(values) - 1)
    margin = critical * deviation / math.sqrt(len(values))
    return ConfidenceInterval(0.95, mean - margin, mean + margin)


def _t_critical(degrees_of_freedom: int) -> float:
    table = (
        12.706204736,
        4.30265273,
        3.182446305,
        2.776445105,
        2.570581836,
        2.446911851,
        2.364624252,
        2.306004135,
        2.262157163,
        2.228138852,
        2.20098516,
        2.17881283,
        2.160368656,
        2.144786688,
        2.131449546,
        2.119905299,
        2.109815578,
        2.10092204,
        2.093024054,
        2.085963447,
        2.079613845,
        2.073873068,
        2.06865761,
        2.063898562,
        2.059538553,
        2.055529439,
        2.051830516,
        2.048407142,
        2.045229642,
        2.042272456,
    )
    return table[degrees_of_freedom - 1] if degrees_of_freedom <= 30 else 1.959963985


def _validate_comparison(direction: str, minimum_effect: float) -> None:
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    if minimum_effect < 0:
        raise ValueError("minimum_effect cannot be negative")


def _signed_effect(baseline: float, candidate: float, direction: str) -> float:
    return baseline - candidate if direction == "minimize" else candidate - baseline


def _point_verdict(relative_effect: float, minimum_effect: float) -> Verdict:
    if relative_effect >= minimum_effect:
        return Verdict.POSITIVE
    if relative_effect <= -minimum_effect:
        return Verdict.NEGATIVE
    return Verdict.INCONCLUSIVE
