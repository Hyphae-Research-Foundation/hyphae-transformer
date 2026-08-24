"""Pinned control-bundle runtime that can only return supplied quotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import torch

from celiums_rezero.governed.deployment import load_deployment_bundle
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.schemas import ControlAction
from celiums_rezero.knowledge.model_runtime import REQUEST_SCHEMA, RESPONSE_SCHEMA
from celiums_rezero.knowledge.orchestration import FrozenModelIdentity
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyDecision,
    TenantId,
)
from celiums_rezero.lab.serialization import canonical_json

QUOTED_RUNTIME_VERSION = "gemma4-e4b-control-v3-quoted-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    request = _request(sys.stdin.buffer.read(1_000_001))
    device = torch.device(arguments.device)
    backbone = Gemma4E4BFrozenBackbone(arguments.model, device=arguments.device)
    controller = load_deployment_bundle(
        arguments.bundle,
        expected_bundle_sha256=arguments.bundle_sha256,
        backbone=backbone,
        device=device,
    )
    expected_identity = FrozenModelIdentity(
        backbone.model_id,
        backbone.revision,
        quoted_runtime_manifest_sha256(arguments.bundle_sha256),
        QUOTED_RUNTIME_VERSION,
    )
    if expected_identity.manifest_digest != arguments.runtime_manifest_sha256:
        raise ValueError("runtime manifest does not match loaded artifact identities")
    if _identity(request["identity"]) != expected_identity:
        raise ValueError("runtime request identity does not match loaded artifacts")
    hits = tuple(_hit(value) for value in cast(list[object], request["passages"]))
    query = cast(str, request["query"])
    generation = cast(str, request["generation_id"])
    request_id = cast(str, request["request_id"])
    bundle = EvidenceBundle(
        tenant=TenantId("runtime_fixture"),
        query_digest=hashlib.sha256(query.encode()).hexdigest(),
        corpus_generation=generation,
        hits=hits,
    )
    observed = controller.observe(
        query=query,
        evidence=bundle,
        host_decision=SufficiencyDecision.SUPPORTED,
    )
    selected = set(observed.selected_handles)
    claims = [
        {"handle": hit.handle, "quote": hit.text}
        for hit in hits
        if hit.handle in selected
    ]
    maximum = cast(int, request["maximum_output_bytes"])
    answer = "\n\n".join(item["quote"] for item in claims)
    if observed.predicted_action is not ControlAction.ANSWER or len(answer.encode()) > maximum:
        decision = "insufficient"
        claims = []
    else:
        decision = "answer"
    print(
        canonical_json(
            {
                "schema": RESPONSE_SCHEMA,
                "request_id": request_id,
                "identity": expected_identity,
                "decision": decision,
                "claims": claims,
            }
        )
    )
    return 0


def _request(payload: bytes) -> dict[str, object]:
    if len(payload) > 1_000_000:
        raise ValueError("runtime request exceeds its byte bound")
    value = json.loads(payload, object_pairs_hook=_unique_object)
    fields = {
        "schema",
        "request_id",
        "identity",
        "query",
        "generation_id",
        "maximum_output_bytes",
        "passages",
    }
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != REQUEST_SCHEMA:
        raise ValueError("runtime request fields are invalid")
    if not isinstance(value["passages"], list) or not value["passages"]:
        raise ValueError("runtime request passages are invalid")
    return cast(dict[str, object], value)


def _identity(value: object) -> FrozenModelIdentity:
    fields = {"model_id", "revision", "manifest_digest", "runtime_version"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("runtime identity fields are invalid")
    return FrozenModelIdentity(
        cast(str, value["model_id"]),
        cast(str, value["revision"]),
        cast(str, value["manifest_digest"]),
        cast(str, value["runtime_version"]),
    )


def _hit(value: object) -> EvidenceHit:
    fields = {
        "handle",
        "source_id",
        "source_version",
        "text",
        "score",
        "content_digest",
        "trusted",
        "active",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("runtime passage fields are invalid")
    return EvidenceHit(
        handle=cast(str, value["handle"]),
        source_id=cast(str, value["source_id"]),
        source_version=cast(str, value["source_version"]),
        text=cast(str, value["text"]),
        score=float(cast(float, value["score"])),
        content_digest=cast(str, value["content_digest"]),
        trusted=cast(bool, value["trusted"]),
        active=cast(bool, value["active"]),
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("runtime request has duplicate keys")
        value[key] = item
    return value


def quoted_runtime_manifest_sha256(bundle_sha256: str) -> str:
    if len(bundle_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_sha256
    ):
        raise ValueError("governed bundle digest is invalid")
    return hashlib.sha256(
        canonical_json(
            {
                "schema": "hyphae-gemma-quoted-runtime-manifest/v1",
                "model_id": Gemma4E4BFrozenBackbone.model_id,
                "model_revision": Gemma4E4BFrozenBackbone.revision,
                "model_artifact_manifest_sha256": (
                    Gemma4E4BFrozenBackbone.artifact_manifest_digest()
                ),
                "bundle_sha256": bundle_sha256,
                "runtime_version": QUOTED_RUNTIME_VERSION,
                "response_schema": RESPONSE_SCHEMA,
            }
        ).encode()
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
