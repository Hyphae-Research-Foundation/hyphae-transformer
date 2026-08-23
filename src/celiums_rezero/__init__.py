"""Hyphae Transformer public API (legacy import namespace)."""

from celiums_rezero.core.gates import ReZeroGate
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.model import ReZeroLM

__all__ = ["ModelConfig", "ReZeroGate", "ReZeroLM", "ResidualStrategy"]
__version__ = "0.2.0"
