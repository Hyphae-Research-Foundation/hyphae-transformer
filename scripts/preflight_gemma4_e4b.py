#!/usr/bin/env python3
"""Fail-closed metadata and environment preflight before E4B download/training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REQUIRED_ARTIFACTS = {
    "chat_template.jinja": (
        None,
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    ),
    "config.json": (
        None,
        "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
    ),
    "generation_config.json": (
        None,
        "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    ),
    "model.safetensors": (
        15_992_595_884,
        "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503",
    ),
    "processor_config.json": (
        None,
        "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    ),
    "tokenizer.json": (
        32_169_626,
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    "tokenizer_config.json": (
        None,
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    arguments = parser.parse_args()
    result = run_preflight(arguments.model, require_gpu=arguments.require_gpu)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


def run_preflight(model: Path | None, *, require_gpu: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "hyphae-transformer.gemma4-e4b-preflight/v1",
        "model_id": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "required_transformers": "5.14.1",
        "required_artifacts": REQUIRED_ARTIFACTS,
        "disk_free_bytes": shutil.disk_usage(_existing_path(model)).free,
    }
    blockers: list[str] = []
    if model is None or not model.is_dir():
        blockers.append("model_checkpoint_absent")
    else:
        artifact_results = {}
        for name, (expected_bytes, expected_digest) in REQUIRED_ARTIFACTS.items():
            path = model / name
            observed_bytes = path.stat().st_size if path.is_file() else None
            observed_digest = (
                _sha256(path) if path.is_file() else None
            )
            valid = (
                not path.is_symlink()
                and observed_digest == expected_digest
                and (expected_bytes is None or observed_bytes == expected_bytes)
            )
            artifact_results[name] = {
                "bytes": observed_bytes,
                "sha256": observed_digest,
                "valid": valid,
            }
            if not valid:
                blockers.append(f"artifact_mismatch:{name}")
        result["artifacts"] = artifact_results
    try:
        import torch

        result["torch"] = torch.__version__
        result["hip"] = torch.version.hip
        result["gpu_available"] = torch.cuda.is_available()
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if require_gpu:
            if not torch.cuda.is_available():
                blockers.append("gpu_unavailable")
            if torch.version.hip is None:
                blockers.append("rocm_unavailable")
            if torch.cuda.device_count() != 1:
                blockers.append("gpu_count_mismatch")
            elif "MI355" not in torch.cuda.get_device_name(0):
                blockers.append("gpu_model_mismatch")
    except ModuleNotFoundError:
        blockers.append("torch_absent")
    try:
        import transformers

        result["transformers"] = transformers.__version__
        if transformers.__version__ != result["required_transformers"]:
            blockers.append("transformers_version_mismatch")
    except ModuleNotFoundError:
        blockers.append("transformers_absent")
    result["blockers"] = blockers
    result["ready"] = not blockers
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_path(path: Path | None) -> Path:
    candidate = Path.cwd() if path is None else path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
