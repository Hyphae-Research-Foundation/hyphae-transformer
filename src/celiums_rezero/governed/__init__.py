"""Governed frozen-backbone control-head training."""

from celiums_rezero.governed.backbone import FixtureBackboneV1, PinnedBackboneIdentity
from celiums_rezero.governed.deployment import (
    GovernedShadowController,
    ShadowControlResult,
    build_deployment_bundle,
    inspect_deployment_bundle,
    load_deployment_bundle,
)
from celiums_rezero.governed.model import GovernedControlHead
from celiums_rezero.governed.schemas import ControlAction, ControlTarget, TrajectoryStep
from celiums_rezero.governed.trainer import ControlTrainConfig, train_control_head

__all__ = [
    "ControlAction",
    "ControlTarget",
    "ControlTrainConfig",
    "FixtureBackboneV1",
    "GovernedControlHead",
    "GovernedShadowController",
    "PinnedBackboneIdentity",
    "ShadowControlResult",
    "TrajectoryStep",
    "build_deployment_bundle",
    "inspect_deployment_bundle",
    "load_deployment_bundle",
    "train_control_head",
]
