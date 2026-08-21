"""Bounded search utilities for staged experiment promotion."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt


@dataclass(frozen=True, slots=True)
class TrialScore:
    name: str
    value: float
    cost: float

    @property
    def information_value(self) -> float:
        return self.value - self.cost


def successive_halving(
    scores: list[TrialScore], *, keep_fraction: float = 0.5
) -> list[TrialScore]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if not scores:
        return []
    keep = max(1, round(len(scores) * keep_fraction))
    return sorted(scores, key=lambda score: score.information_value, reverse=True)[:keep]


def upper_confidence_bound(
    *, value_sum: float, visits: int, parent_visits: int, exploration: float
) -> float:
    if visits < 1 or parent_visits < visits:
        raise ValueError("invalid tree visit counts")
    if exploration < 0:
        raise ValueError("exploration cannot be negative")
    return value_sum / visits + exploration * sqrt(2 * log(parent_visits) / visits)
