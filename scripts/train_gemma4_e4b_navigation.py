#!/usr/bin/env python3
"""Train the preregistered ReZero neuropilot over fixture navigation trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from celiums_rezero.governed.backbone import FixtureBackboneV1
from celiums_rezero.governed.data import load_governed_dataset
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.navigation import derive_navigation_dataset
from celiums_rezero.governed.navigation_experiment import run_navigation_experiment
from celiums_rezero.governed.schemas import (
    DatasetSplit,
    GovernedDatasetManifest,
)
from celiums_rezero.knowledge.schemas import SufficiencyPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    preregistration_bytes = arguments.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes)
    manifest_values = json.loads((arguments.dataset / "manifest.json").read_text())
    dataset = load_governed_dataset(
        arguments.dataset,
        GovernedDatasetManifest(
            splits=tuple(
                (name, DatasetSplit(**item))
                for name, item in sorted(manifest_values["splits"].items())
            ),
            policy=SufficiencyPolicy(**manifest_values["policy"]),
            maximum_evidence_items=manifest_values["maximum_evidence_items"],
        ),
    )
    if arguments.model is not None:
        backbone = Gemma4E4BFrozenBackbone(arguments.model, device=arguments.device)
        device = torch.device(arguments.device)
    else:
        backbone = FixtureBackboneV1()
        device = torch.device("cpu")
    trajectories = derive_navigation_dataset(
        dataset,
        SufficiencyPolicy(**manifest_values["policy"]),
        provenance=preregistration["dataset"]["provenance"],
    )
    report = run_navigation_experiment(
        backbone=backbone,
        preregistration=preregistration,
        preregistration_sha256=hashlib.sha256(preregistration_bytes).hexdigest(),
        dataset=dataset,
        trajectories=trajectories,
        output=arguments.out,
        device=device,
        maximum_evidence_items=dataset.manifest.maximum_evidence_items,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
