#!/usr/bin/env python3
"""Run one exact governed Gemma request through the supervised runtime boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from celiums_rezero.governed.deployment import inspect_deployment_bundle
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.runtime import (
    QUOTED_RUNTIME_VERSION,
    quoted_runtime_manifest_sha256,
)
from celiums_rezero.knowledge.model_runtime import (
    SupervisedFrozenGemmaRuntime,
    SupervisedFrozenRuntimeConfig,
)
from celiums_rezero.knowledge.orchestration import (
    FrozenModelIdentity,
    GovernedModelRequest,
)
from celiums_rezero.knowledge.schemas import EvidenceHit

BUNDLE_SHA256 = "93db742ead71c12fa46c62661b12108fdb0a815d3b5fcf180821538dcfc8b9be"
QUERY = "What is the approved maintenance window?"
GENERATION = "generation_quoted_runtime_canary_v1"
HANDLE = "passage_0000000000000000"
PASSAGE = "Service policy: approved maintenance window is 02:00-04:00 UTC."
WRAPPER = """#!/usr/bin/env python3
import sys

sys.path[:0] = ["/workspace/src", "/python"]

from celiums_rezero.governed.runtime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-patch-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-timeout-seconds", type=float, default=180)
    arguments = parser.parse_args()
    if arguments.bundle_sha256 != BUNDLE_SHA256:
        raise ValueError("quoted runtime canary bundle differs from its pin")
    if not _digest(arguments.source_revision, 40) or not _digest(
        arguments.source_patch_sha256, 64
    ):
        raise ValueError("quoted runtime source identity is invalid")
    arguments.out.mkdir(parents=True, exist_ok=True)
    executable = arguments.out / "hyphae-governed-runtime"
    executable.write_text(WRAPPER, encoding="ascii")
    executable.chmod(0o500)
    executable_digest = _sha256(executable)
    bundle = inspect_deployment_bundle(arguments.bundle)
    runtime_manifest = quoted_runtime_manifest_sha256(arguments.bundle_sha256)
    identity = FrozenModelIdentity(
        Gemma4E4BFrozenBackbone.model_id,
        Gemma4E4BFrozenBackbone.revision,
        runtime_manifest,
        QUOTED_RUNTIME_VERSION,
    )
    hit = EvidenceHit(
        HANDLE,
        "quoted_runtime_canary",
        "v1",
        PASSAGE,
        0.97,
        hashlib.sha256(PASSAGE.encode()).hexdigest(),
    )
    runtime = SupervisedFrozenGemmaRuntime(
        SupervisedFrozenRuntimeConfig(
            executable=executable,
            executable_sha256=executable_digest,
            identity=identity,
            arguments=(
                "--model",
                str(arguments.model),
                "--bundle",
                str(arguments.bundle),
                "--bundle-sha256",
                arguments.bundle_sha256,
                "--runtime-manifest-sha256",
                runtime_manifest,
                "--device",
                "cuda:0",
            ),
        )
    )
    exchange = runtime.infer_exchange(
        GovernedModelRequest(QUERY, GENERATION, (hit,), maximum_output_bytes=1024),
        timeout_seconds=arguments.runtime_timeout_seconds,
    )
    result = exchange.result
    passed = (
        result.identity == identity
        and result.decision == "answer"
        and len(result.claims) == 1
        and result.claims[0].handle == HANDLE
        and result.claims[0].quote == PASSAGE
    )
    (arguments.out / "protocol-request.json").write_bytes(exchange.request_payload)
    (arguments.out / "protocol-response.json").write_bytes(exchange.response_payload)
    report = {
        "schema": "hyphae-transformer.gemma4-e4b-quoted-runtime-canary/v1",
        "completed": True,
        "passed": passed,
        "request_count": 1,
        "source_revision": arguments.source_revision,
        "source_patch_sha256": arguments.source_patch_sha256,
        "model_id": identity.model_id,
        "model_revision": identity.revision,
        "model_artifact_manifest_sha256": (
            Gemma4E4BFrozenBackbone.artifact_manifest_digest()
        ),
        "model_safetensors_sha256": Gemma4E4BFrozenBackbone.model_safetensors_sha256,
        "bundle_sha256": arguments.bundle_sha256,
        "bundle_id": bundle.bundle_id,
        "runtime_manifest_sha256": runtime_manifest,
        "runtime_version": identity.runtime_version,
        "executable_sha256": executable_digest,
        "request_id": exchange.request_id,
        "request_sha256": hashlib.sha256(exchange.request_payload).hexdigest(),
        "response_sha256": hashlib.sha256(exchange.response_payload).hexdigest(),
        "decision": result.decision,
        "evidence_handles": [claim.handle for claim in result.claims],
        "answer": "\n\n".join(claim.quote for claim in result.claims),
    }
    report_path = arguments.out / "quoted-runtime-canary-report.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(report_path)
    return 0 if passed else 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
