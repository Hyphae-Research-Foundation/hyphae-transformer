"""Objective comparison rules for preregistered architecture hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

from celiums_rezero.lab.schemas import Verdict


@dataclass(frozen=True, slots=True)
class Comparison:
    baseline_mean: float
    candidate_mean: float
    relative_effect: float
    minimum_effect: float
    direction: str
    verdict: Verdict


def compare_means(
    baseline: list[float],
    candidate: list[float],
    *,
    direction: str,
    minimum_effect: float,
) -> Comparison:
    if not baseline or not candidate:
        raise ValueError("both conditions require observations")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    if minimum_effect < 0:
        raise ValueError("minimum_effect cannot be negative")
    baseline_mean = sum(baseline) / len(baseline)
    candidate_mean = sum(candidate) / len(candidate)
    denominator = max(abs(baseline_mean), 1e-12)
    signed = (
        baseline_mean - candidate_mean
        if direction == "minimize"
        else candidate_mean - baseline_mean
    )
    relative_effect = signed / denominator
    if relative_effect >= minimum_effect:
        verdict = Verdict.POSITIVE
    elif relative_effect <= -minimum_effect:
        verdict = Verdict.NEGATIVE
    else:
        verdict = Verdict.INCONCLUSIVE
    return Comparison(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        relative_effect=relative_effect,
        minimum_effect=minimum_effect,
        direction=direction,
        verdict=verdict,
    )
