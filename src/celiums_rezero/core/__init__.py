"""Core residual primitives and diagnostics."""

from celiums_rezero.core.gates import ReZeroGate
from celiums_rezero.core.optim import build_optimizer_groups

__all__ = ["ReZeroGate", "build_optimizer_groups"]
