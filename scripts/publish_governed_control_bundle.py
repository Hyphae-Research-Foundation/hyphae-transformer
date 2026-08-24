#!/usr/bin/env python3
"""Publish and independently verify a governed-control bundle in durable object storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from celiums_rezero.governed.deployment import inspect_deployment_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--profile", required=True)
    arguments = parser.parse_args()
    if not arguments.destination.startswith("s3://"):
        raise ValueError("durable destination must be an explicit s3:// URI")
    manifest = inspect_deployment_bundle(arguments.bundle)
    local_digest = hashlib.sha256(arguments.bundle.read_bytes()).hexdigest()
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(arguments.bundle),
            arguments.destination,
            "--profile",
            arguments.profile,
            "--only-show-errors",
        ],
        check=True,
    )
    with tempfile.TemporaryDirectory() as temporary:
        downloaded = Path(temporary) / arguments.bundle.name
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                arguments.destination,
                str(downloaded),
                "--profile",
                arguments.profile,
                "--only-show-errors",
            ],
            check=True,
        )
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != local_digest:
            raise RuntimeError("durable bundle verification failed")
        inspect_deployment_bundle(downloaded)
    print(
        json.dumps(
            {
                "bundle_id": manifest.bundle_id,
                "destination": arguments.destination,
                "sha256": local_digest,
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
