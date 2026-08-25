#!/usr/bin/env python3
"""Run the preregistered frozen-Gemma ReZero sequence-control experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.rezero_experiment import (
    load_governed_dataset_directory,
    run_rezero_sequence_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-batch-size", type=int, required=True)
    arguments = parser.parse_args()
    preregistration_bytes = arguments.preregistration.read_bytes()
    preregistration = json.loads(preregistration_bytes)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("Gemma ReZero training requires ROCm")
    dataset = load_governed_dataset_directory(
        arguments.dataset,
        expected_dataset_id=str(preregistration["dataset"]["governed_dataset_id"]),
    )
    report = run_rezero_sequence_experiment(
        backbone=Gemma4E4BFrozenBackbone(arguments.model, device=str(device)),
        dataset=dataset,
        preregistration=preregistration,
        preregistration_sha256=hashlib.sha256(preregistration_bytes).hexdigest(),
        output=arguments.out,
        device=device,
        feature_batch_size=arguments.feature_batch_size,
        scope="gemma",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
