#!/usr/bin/env python3
"""Exercise the live knowledge publication gate against an isolated Hyphae endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from celiums_rezero.knowledge import (
    HYPHAE_210_RETRIEVAL_PROFILE,
    AcquisitionPolicy,
    DurableAcquisitionWorker,
    DurableFinalizationWorker,
    DurablePublicationAuthorizer,
    EvidenceBundle,
    FinalizationPolicy,
    GenerationAuthority,
    GenerationRoutedEvidenceProvider,
    GenerationRoutedRetriever,
    HyphaeShadowIngestor,
    InMemorySourceConnector,
    JobStatus,
    KnowledgeCoordinator,
    PublicationReceiptStore,
    SQLiteMailboxConfig,
    SQLiteMailboxNotificationSink,
    SQLiteTenantStore,
    StrictArtifactValidator,
    SufficiencyPolicy,
    SupervisedFrozenGemmaRuntime,
    SupervisedFrozenRuntimeConfig,
    TenantId,
)
from celiums_rezero.knowledge.acquisition import (
    ChunkingPolicy,
    chunk_validated_artifact,
    ingest_idempotency_key,
)
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.orchestration import (
    FrozenModelIdentity,
    HostGemmaAnswerer,
)
from celiums_rezero.knowledge.schemas import (
    EmbeddedChunk,
    GenerationManifest,
    SourceArtifact,
    SourcePolicy,
)


class FixtureEmbedder:
    profile = "hyphae-conformance-v1"
    dimensions = 2

    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode()).digest()
        return (digest[0] / 255, digest[1] / 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--collection", type=int, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--routing-database", type=Path, required=True)
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
    from hyphae_sdk.v2 import HyphaeClient, RequestOptions
    from hyphae_sdk.v2.protocol import PROTOCOL_MAJOR, PROTOCOL_MINOR

    if hyphae_sdk.__version__ != arguments.expected_sdk_version:
        raise RuntimeError("Hyphae SDK version differs from certification pin")

    tenant = TenantId(arguments.tenant)
    backend_id = arguments.backend_id
    if arguments.score_scale != HYPHAE_210_RETRIEVAL_PROFILE.score_scale:
        raise RuntimeError("retrieval score scale differs from the Hyphae 2.1.0 profile")
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
    routing_store = SQLiteTenantStore(arguments.routing_database, tenant=tenant)
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(),
        acquisition=AcquisitionPolicy(version="conformance-policy-v1", sources=(source,)),
        store=routing_store,
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
        validator = StrictArtifactValidator()
        job = coordinator.job_status(tenant, pending.job_id)
        if job is None or ingestor.target is None:
            raise RuntimeError("canary job or publication target is absent")
        validated = validator.validate(artifact)
        embedder = FixtureEmbedder()
        embedded = tuple(
            EmbeddedChunk(chunk, embedder.profile, embedder.embed(chunk.text))
            for chunk in chunk_validated_artifact(validated, ChunkingPolicy())
        )
        manifest = GenerationManifest(
            tenant=tenant,
            generation_id="generation-conformance-v1",
            target=ingestor.target,
            parent_generation_id=None,
            chunk_ids=tuple(item.chunk.chunk_id for item in embedded),
            ingest_idempotency_keys=(
                ingest_idempotency_key(
                    job,
                    embedded,
                    FixtureEmbedder.profile,
                    target=ingestor.target,
                ),
            ),
            ingest_receipt_digests=(),
        )
        authority = GenerationAuthority(routing_store, receipts=receipts)
        authority.register(manifest)
        worker = DurableAcquisitionWorker(
            worker_id="hyphae-conformance-acquisition",
            lease_seconds=30,
            coordinator=coordinator,
            connector=InMemorySourceConnector({(tenant.value, source.source_id): artifact}),
            embedder=embedder,
            ingestor=ingestor,
            verifier=ingestor,
            validator=validator,
            authorizer=DurablePublicationAuthorizer(
                tenant=tenant,
                store=receipts,
                authority="hyphae-conformance-operator",
                enabled=True,
            ),
        )
        outcome = worker.run_next(job_id=pending.job_id)
        if outcome is None:
            raise RuntimeError("durable publication job was not claimable")
        if outcome.job.status is not JobStatus.READY or outcome.receipt is None:
            raise RuntimeError(f"publication did not become ready: {outcome.job.failure}")

        recovered = PublicationReceiptStore(arguments.receipts)
        durable = recovered.load_ingest(tenant, outcome.receipt.idempotency_key)
        if durable != outcome.receipt:
            raise RuntimeError("durable ingest receipt did not survive restart")
        authority = GenerationAuthority(
            routing_store,
            receipts=recovered,
        )
        authority.register(manifest)
        authority.verify_candidate(manifest, (outcome.receipt,))
        activation = authority.activate(
            manifest.generation_id,
            expected_revision=0,
            actor="hyphae-conformance-operator",
            reason="isolated native canary",
        )
        router = GenerationRoutedRetriever(
            tenant=tenant,
            authority=authority,
            client=client,
            profile=HYPHAE_210_RETRIEVAL_PROFILE,
            embedder=embedder,
            request_options_factory=lambda timeout: RequestOptions(
                deadline_micros=int((time.time() + timeout) * 1_000_000)
            ),
        )
        evidence = router.retrieve(tenant, query)
        if not evidence.hits or all(hit.text != body.decode() for hit in evidence.hits):
            raise RuntimeError("integrated retrieval did not hydrate the published body")
        runtime_executable = (
            Path(sys.executable).parent / "hyphae-quoted-runtime-canary"
        ).resolve(strict=True)
        runtime_identity = FrozenModelIdentity(
            "canary-quoted-subprocess",
            "v1",
            hashlib.sha256(b"hyphae-completion-canary-subprocess-v1").hexdigest(),
            "supervised-json-v1",
        )
        runtime = SupervisedFrozenGemmaRuntime(
            SupervisedFrozenRuntimeConfig(
                executable=runtime_executable,
                executable_sha256=_sha256(runtime_executable),
                identity=runtime_identity,
            )
        )
        mailbox_config = SQLiteMailboxConfig(
            tenant_id=tenant.value,
            path=arguments.routing_database.with_name("mailbox.sqlite3"),
            mailbox_id="hyphae-conformance-mailbox-v1",
        )
        sink = SQLiteMailboxNotificationSink(mailbox_config)
        finalizer = DurableFinalizationWorker(
            store=routing_store,
            worker_id="hyphae-conformance-finalizer",
            lease_seconds=10,
            answerer=HostGemmaAnswerer(
                runtime=runtime,
                evidence=GenerationRoutedEvidenceProvider(retriever=router),
                expected_identity=runtime_identity,
            ),
            sink=sink,
            policy=FinalizationPolicy(
                answer_timeout_seconds=5,
                notification_timeout_seconds=5,
                lease_safety_seconds=1,
            ),
        )
        completed = finalizer.run_next(job_id=pending.job_id)
        if completed is None or completed.status is not JobStatus.COMPLETED:
            raise RuntimeError("routed canary did not complete durable finalization")
        notification = routing_store.prepared_notification(tenant, pending.job_id)
        if notification is None:
            raise RuntimeError("routed canary did not persist its notification command")
        recovered_mailbox = SQLiteMailboxNotificationSink(mailbox_config)
        replay_receipt = recovered_mailbox.deliver(notification, timeout_seconds=5)
        if recovered_mailbox.accepted_count() != 1:
            raise RuntimeError("durable mailbox did not deduplicate notification replay")

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
                "generation_activation": _json_value(asdict(activation)),
                "generation_snapshot": _json_value(asdict(authority.snapshot())),
                "finalization": {
                    "job_id": completed.job_id,
                    "status": completed.status,
                    "notification_id": notification.notification_id,
                    "notification_attempts": routing_store.notification_attempts(
                        tenant, pending.job_id
                    ),
                    "answer": notification.answer,
                    "evidence_handles": notification.evidence_handles,
                    "mailbox_sink_id": sink.sink_id,
                    "mailbox_provider_receipt": replay_receipt.provider_receipt,
                    "mailbox_accepted": recovered_mailbox.accepted_count(),
                },
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
