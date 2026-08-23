"""Governed frozen-backbone control-head training."""

from celiums_rezero.governed.backbone import FixtureBackboneV1, PinnedBackboneIdentity
from celiums_rezero.governed.model import GovernedControlHead
from celiums_rezero.governed.schemas import ControlAction, ControlTarget, TrajectoryStep
from celiums_rezero.governed.trainer import ControlTrainConfig, train_control_head

__all__ = [
    "ControlAction",
    "ControlTarget",
    "ControlTrainConfig",
    "FixtureBackboneV1",
    "GovernedControlHead",
    "PinnedBackboneIdentity",
    "TrajectoryStep",
    "train_control_head",
]
