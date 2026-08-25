#!/usr/bin/env python3
"""Build a deterministic ReZero navigation pilot bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint.read_bytes()
    torch = __import__("torch")
    loaded = torch.load(
        __import__("io").BytesIO(checkpoint), map_location="cpu", weights_only=False
    )
    payload = json.loads(loaded["backbone"])
    report = json.loads(arguments.training_report.read_text())
    seed_report = next(item for item in report["final"] if item["seed"] == 17)
    if (
        report.get("completed") is not True
        or report.get("passed") is not True
        or seed_report["validation"].get("passed") is not True
        or hashlib.sha256(checkpoint).hexdigest() != seed_report["training"]["checkpoint_sha256"]
    ):
        raise SystemExit("navigation bundle requires the passing canonical seed")
    expected_schema = (
        "hyphae-transformer.gemma4-e4b-rezero-navigation-experiment/v2"
        if arguments.version == "v2"
        else "hyphae-transformer.gemma4-e4b-rezero-navigation-experiment/v1"
    )
    if report.get("schema") != expected_schema:
        raise SystemExit("navigation bundle report schema differs")
    calibration = report.get("calibration", {})
    if arguments.version == "v2" and (
        calibration.get("scheme") != "hyphae-2.1.0-exact-filtered-v1"
        or calibration.get("score_scale") != 0.03278688524590164
    ):
        raise SystemExit("navigation v2 bundle requires pinned calibration")
    manifest = {
        "schema": f"hyphae-transformer.rezero-navigation-pilot-bundle/{arguments.version}",
        "backbone": payload,
        "selected_learning_rate": report["selected_learning_rate"],
        "experiment_schema": expected_schema,
        "calibration": calibration,
        "seed": 17,
        "artifacts": [
            {
                "path": "navigation-control.pt",
                "bytes": len(checkpoint),
                "sha256": hashlib.sha256(checkpoint).hexdigest(),
            },
            {
                "path": "training-report.json",
                "bytes": arguments.training_report.stat().st_size,
                "sha256": hashlib.sha256(arguments.training_report.read_bytes()).hexdigest(),
            },
            {
                "path": "preregistration.json",
                "bytes": arguments.preregistration.stat().st_size,
                "sha256": hashlib.sha256(arguments.preregistration.read_bytes()).hexdigest(),
            },
        ],
    }
    sources = {
        "navigation-control.pt": checkpoint,
        "training-report.json": arguments.training_report.read_bytes(),
        "preregistration.json": arguments.preregistration.read_bytes(),
        "manifest.json": (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=arguments.out.parent) as temporary_name:
        root = Path(temporary_name)
        for name, content in sources.items():
            (root / name).write_bytes(content)
        with (
            arguments.out.open("wb") as raw_output,
            gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0, filename="") as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                info = archive.gettarinfo(str(path), arcname=path.name)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as file_source:
                    archive.addfile(info, file_source)
    print(
        json.dumps(
            {
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
