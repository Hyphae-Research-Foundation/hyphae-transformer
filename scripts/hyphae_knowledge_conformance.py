#!/usr/bin/env python3
"""Exercise the live knowledge publication gate against an isolated Hyphae endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from celiums_rezero.knowledge import (
    AcquisitionPolicy,
    AcquisitionWorker,
    DurablePublicationAuthorizer,
    EvidenceBundle,
    HyphaeRetrievalGateway,
    HyphaeShadowIngestor,
    InMemorySourceConnector,
    InMemoryTenantStore,
    JobStatus,
    KnowledgeCoordinator,
    PublicationReceiptStore,
    RetrievalConfig,
    StrictArtifactValidator,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.schemas import SourceArtifact, SourcePolicy


class FixtureEmbedder:
    profile = "hyphae-conformance-v1"

    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode()).digest()
        return (digest[0] / 255, digest[1] / 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--collection", type=int, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument(
        "--backend-id",
        required=True,
        help="SHA-256 identity bound to the persistent Hyphae directory lineage",
    )
    parser.add_argument("--tenant", default="tenant_conformance")
    parser.add_argument("--expected-sdk-version", default="2.1.0")
    parser.add_argument("--expected-runtime-version", default="2.1.0")
    parser.add_argument("--score-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if not arguments.endpoint.is_socket():
        raise SystemExit("Hyphae endpoint must be an existing Unix socket")
    if arguments.receipts.exists() and arguments.receipts.is_symlink():
        raise SystemExit("receipt directory cannot be a symlink")
    key = arguments.api_key_file.read_text(encoding="ascii").strip()
    if not key:
        raise SystemExit("Hyphae API key file is empty")

    import hyphae_sdk
    from hyphae_sdk.v2 import HyphaeClient
    from hyphae_sdk.v2.protocol import PROTOCOL_MAJOR, PROTOCOL_MINOR

    if hyphae_sdk.__version__ != arguments.expected_sdk_version:
        raise RuntimeError("Hyphae SDK version differs from certification pin")

    tenant = TenantId(arguments.tenant)
    backend_id = arguments.backend_id
    query = "How does integrated filtering bind evidence?"
    normalized = normalize_query(query)
    body = (
        b"Integrated filtering binds evidence to the active corpus generation "
        b"before retrieval."
    )
    source = SourcePolicy(
        source_id="official_docs",
        allowed_hosts=("docs.example.com",),
        allowed_mime_types=("text/plain",),
        allowed_license_ids=("Apache-2.0",),
        max_download_bytes=4096,
    )
    artifact = SourceArtifact(
        tenant=tenant,
        source_id=source.source_id,
        source_version="conformance-v1",
        body=body,
        content_type="text/plain",
        license_id="Apache-2.0",
        content_digest=hashlib.sha256(body).hexdigest(),
    )
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(),
        acquisition=AcquisitionPolicy(version="conformance-policy-v1", sources=(source,)),
        store=InMemoryTenantStore(),
        embedding_profile=FixtureEmbedder.profile,
    )
    pending = coordinator.answer_or_enqueue(
        tenant=tenant,
        query=query,
        evidence=EvidenceBundle(
            tenant=tenant,
            query_digest=hashlib.sha256(normalized.encode()).hexdigest(),
            corpus_generation="generation-conformance-v1",
            hits=(),
        ),
        source_id=source.source_id,
    )
    assert pending.job_id is not None
    receipts = PublicationReceiptStore(arguments.receipts)
    with HyphaeClient.local_authenticated(str(arguments.endpoint), key) as client:
        capabilities = client.capabilities()
        if capabilities.kind != "capabilities":
            raise RuntimeError("Hyphae capability response is invalid")
        ingestor = HyphaeShadowIngestor(
            tenant=tenant,
            client=client,
            collection=arguments.collection,
            vector_target="semantic",
            backend_id=backend_id,
            publish=True,
            receipt_store=receipts,
        )
        worker = AcquisitionWorker(
            coordinator=coordinator,
            connector=InMemorySourceConnector({(tenant.value, source.source_id): artifact}),
            embedder=FixtureEmbedder(),
            ingestor=ingestor,
            verifier=ingestor,
            validator=StrictArtifactValidator(),
            authorizer=DurablePublicationAuthorizer(
                tenant=tenant,
                store=receipts,
                authority="hyphae-conformance-operator",
                enabled=True,
            ),
        )
        outcome = worker.run(tenant, pending.job_id)
        if outcome.job.status is not JobStatus.READY or outcome.receipt is None:
            raise RuntimeError(f"publication did not become ready: {outcome.job.failure}")

        recovered = PublicationReceiptStore(arguments.receipts)
        durable = recovered.load_ingest(tenant, outcome.receipt.idempotency_key)
        if durable != outcome.receipt:
            raise RuntimeError("durable ingest receipt did not survive restart")
        gateway = HyphaeRetrievalGateway(
            tenant=tenant,
            client=client,
            config=RetrievalConfig(
                collection=arguments.collection,
                corpus_generation="generation-conformance-v1",
                backend_id=backend_id,
                vector_target="semantic",
                score_scale=arguments.score_scale,
                require_exact_vector_receipt=True,
            ),
            embedder=FixtureEmbedder(),
        )
        evidence = gateway.retrieve(tenant, query)
        if not evidence.hits or all(hit.text != body.decode() for hit in evidence.hits):
            raise RuntimeError("integrated retrieval did not hydrate the published body")

    print(
        json.dumps(
            {
                "schema": "celiums-rezero-hyphae-conformance-v1",
                "status": "passed",
                "tenant": tenant.value,
                "collection": arguments.collection,
                "sdk_version": hyphae_sdk.__version__,
                "expected_runtime_version": arguments.expected_runtime_version,
                "protocol": {"major": PROTOCOL_MAJOR, "minor": PROTOCOL_MINOR},
                "capabilities": _json_value(capabilities.value),
                "ingest_receipt": _json_value(asdict(outcome.receipt)),
                "snapshot_fingerprint": evidence.snapshot_fingerprint,
                "evidence_handles": [hit.handle for hit in evidence.hits],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
