"""Governed frozen-backbone control-head training."""

from celiums_rezero.governed.backbone import FixtureBackboneV1, PinnedBackboneIdentity
from celiums_rezero.governed.deployment import (
    AuditedShadowObserver,
    GovernedShadowController,
    ReZeroDeploymentBundleManifest,
    ShadowControlResult,
    build_deployment_bundle,
    build_rezero_deployment_bundle,
    inspect_deployment_bundle,
    inspect_rezero_deployment_bundle,
    load_deployment_bundle,
    load_rezero_deployment_bundle,
)
from celiums_rezero.governed.hyphaelm import ReZeroNeuroPilot
from celiums_rezero.governed.model import GovernedControlHead, ReZeroSequenceControlHead
from celiums_rezero.governed.navigation import (
    NAVIGATION_ACTIONS,
    NavigationStep,
    derive_navigation_dataset,
)
from celiums_rezero.governed.schemas import ControlAction, ControlTarget, TrajectoryStep
from celiums_rezero.governed.trainer import ControlTrainConfig, train_control_head

__all__ = [
    "NAVIGATION_ACTIONS",
    "AuditedShadowObserver",
    "ControlAction",
    "ControlTarget",
    "ControlTrainConfig",
    "FixtureBackboneV1",
    "GovernedControlHead",
    "GovernedShadowController",
    "NavigationStep",
    "PinnedBackboneIdentity",
    "ReZeroDeploymentBundleManifest",
    "ReZeroNeuroPilot",
    "ReZeroSequenceControlHead",
    "ShadowControlResult",
    "TrajectoryStep",
    "build_deployment_bundle",
    "build_rezero_deployment_bundle",
    "derive_navigation_dataset",
    "inspect_deployment_bundle",
    "inspect_rezero_deployment_bundle",
    "load_deployment_bundle",
    "load_rezero_deployment_bundle",
    "train_control_head",
]
