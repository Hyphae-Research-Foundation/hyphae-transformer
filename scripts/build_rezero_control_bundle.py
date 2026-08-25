#!/usr/bin/env python3
"""Build a deterministic ReZero sequence-control deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from celiums_rezero.governed.deployment import build_rezero_deployment_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = build_rezero_deployment_bundle(
        output=arguments.out,
        checkpoint=arguments.checkpoint,
        training_report=arguments.training_report,
        preregistration=arguments.preregistration,
        dataset_manifest=arguments.dataset_manifest,
        source_revision=arguments.source_revision,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "bundle_id": manifest.bundle_id,
                "path": str(arguments.out),
                "bytes": arguments.out.stat().st_size,
                "sha256": hashlib.sha256(arguments.out.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
