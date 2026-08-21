"""Preflight and observed budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass

from celiums_rezero.lab.schemas import Budget


class BudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class BudgetTracker:
    budget: Budget
    wall_seconds: float = 0.0
    device_hours: float = 0.0
    cost_usd: float = 0.0
    failures: int = 0
    artifact_bytes: int = 0

    def update(
        self,
        *,
        wall_seconds: float = 0.0,
        device_hours: float = 0.0,
        cost_usd: float = 0.0,
        failures: int = 0,
        artifact_bytes: int = 0,
    ) -> None:
        values = (wall_seconds, device_hours, cost_usd, failures, artifact_bytes)
        if any(value < 0 for value in values):
            raise ValueError("budget usage increments cannot be negative")
        self.wall_seconds += wall_seconds
        self.device_hours += device_hours
        self.cost_usd += cost_usd
        self.failures += failures
        self.artifact_bytes += artifact_bytes
        self.enforce()

    def enforce(self) -> None:
        checks = {
            "wall seconds": (self.wall_seconds, self.budget.max_wall_seconds),
            "failures": (self.failures, self.budget.max_failures),
            "artifact bytes": (self.artifact_bytes, self.budget.max_artifact_bytes),
        }
        if self.budget.max_device_hours > 0:
            checks["device hours"] = (
                self.device_hours,
                self.budget.max_device_hours,
            )
        if self.budget.max_cost_usd > 0:
            checks["cost"] = (self.cost_usd, self.budget.max_cost_usd)
        for name, (observed, maximum) in checks.items():
            if observed > maximum:
                raise BudgetExceeded(f"{name} budget exceeded: {observed} > {maximum}")
