from __future__ import annotations

import hashlib
import socket
import struct
import threading
from pathlib import Path

from celiums_rezero.knowledge import SQLiteTenantStore, TenantId
from celiums_rezero.knowledge.generation import GenerationAuthority
from celiums_rezero.knowledge.operations import AuditChain, render_prometheus
from celiums_rezero.knowledge.orchestration import (
    FrozenModelIdentity,
    GovernedModelResult,
    HostGemmaAnswerer,
    QuotedClaim,
)
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EvidenceBundle,
    EvidenceHit,
    FinalizationQueueSnapshot,
    GenerationManifest,
    PublicationTarget,
)
from celiums_rezero.knowledge.security import ClamDScanner, ExternalDlpScanner
from celiums_rezero.knowledge.supervisor import run_supervised


class Clock:
    def __call__(self) -> int:
        return 1_800_000_000_000_000


def store(tmp_path: Path, tenant: str) -> SQLiteTenantStore:
    root = tmp_path / tenant
    root.mkdir(mode=0o700)
    return SQLiteTenantStore(root / "jobs.sqlite3", tenant=TenantId(tenant), clock_us=Clock())


def manifest(tenant: TenantId, suffix: str, parent: str | None = None) -> GenerationManifest:
    return GenerationManifest(
        tenant=tenant,
        generation_id=f"generation_{suffix}",
        target=PublicationTarget(
            "a" * 64,
            10 + int(hashlib.sha256(suffix.encode()).hexdigest()[:6], 16),
            "semantic",
        ),
        parent_generation_id=parent,
        chunk_ids=(f"chunk_{suffix:0<16}"[:22],),
        ingest_idempotency_keys=(hashlib.sha256((suffix + "key").encode()).hexdigest(),),
        ingest_receipt_digests=(hashlib.sha256(suffix.encode()).hexdigest(),),
    )


def test_generation_cutover_and_rollback_are_tenant_local(tmp_path: Path) -> None:
    tenant_a = TenantId("tenant_a")
    tenant_b = TenantId("tenant_b")
    authority_a = GenerationAuthority(store(tmp_path, tenant_a.value))
    authority_b = GenerationAuthority(store(tmp_path, tenant_b.value))
    first = manifest(tenant_a, "one")
    second = manifest(tenant_a, "two", first.generation_id)
    other = manifest(tenant_b, "other")
    for authority, item in ((authority_a, first), (authority_a, second), (authority_b, other)):
        authority.register(item)
        verified = GenerationManifest(
            tenant=item.tenant,
            generation_id=item.generation_id,
            target=item.target,
            parent_generation_id=item.parent_generation_id,
            chunk_ids=item.chunk_ids,
            ingest_idempotency_keys=item.ingest_idempotency_keys,
            ingest_receipt_digests=item.ingest_receipt_digests,
            verification_token=hashlib.sha256(
                b"generation-verification-v1\0"
                + (item.manifest_digest or "").encode()
                + b"".join(value.encode() for value in item.ingest_receipt_digests)
            ).hexdigest(),
        )
        authority.store._complete_generation_manifest(item, verified)
    authority_a.activate(first.generation_id, expected_revision=0, actor="test", reason="bootstrap")
    authority_b.activate(other.generation_id, expected_revision=0, actor="test", reason="bootstrap")
    authority_a.pause(expected_revision=1, actor="test")
    authority_a.activate(second.generation_id, expected_revision=2, actor="test", reason="promote")
    rolled = authority_a.rollback(expected_revision=3, actor="test", reason="incident")
    assert rolled.to_generation_id == first.generation_id
    assert authority_a.snapshot().generation_id == first.generation_id
    assert authority_b.snapshot().generation_id == other.generation_id


def test_generation_cutover_compare_and_swap_rejects_stale_revision(tmp_path: Path) -> None:
    tenant = TenantId("tenant_a")
    authority = GenerationAuthority(store(tmp_path, tenant.value))
    first = manifest(tenant, "one")
    second = manifest(tenant, "two", first.generation_id)
    for item in (first, second):
        authority.register(item)
        verified = GenerationManifest(
            tenant=item.tenant,
            generation_id=item.generation_id,
            target=item.target,
            parent_generation_id=item.parent_generation_id,
            chunk_ids=item.chunk_ids,
            ingest_idempotency_keys=item.ingest_idempotency_keys,
            ingest_receipt_digests=item.ingest_receipt_digests,
            verification_token=hashlib.sha256(
                b"generation-verification-v1\0"
                + (item.manifest_digest or "").encode()
                + b"".join(value.encode() for value in item.ingest_receipt_digests)
            ).hexdigest(),
        )
        authority.store._complete_generation_manifest(item, verified)
    authority.activate(first.generation_id, expected_revision=0, actor="test", reason="first")
    try:
        authority.activate(second.generation_id, expected_revision=0, actor="test", reason="stale")
    except PermissionError:
        pass
    else:
        raise AssertionError("stale generation revision was accepted")


def test_clamd_and_dlp_adapters_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "clamd.sock"
    received = bytearray()

    def server() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(1)
            connection, _ = listener.accept()
            with connection:
                assert connection.recv(10) == b"zINSTREAM\0"
                while True:
                    size = struct.unpack(">I", connection.recv(4))[0]
                    if size == 0:
                        break
                    received.extend(connection.recv(size))
                connection.sendall(b"stream: OK\0")

    thread = threading.Thread(target=server)
    thread.start()
    while not path.exists():
        pass
    scanner = ClamDScanner(path, "clamav-test")
    assert scanner.findings(b"safe bytes") == ()
    thread.join()
    assert bytes(received) == b"safe bytes"

    dlp = ExternalDlpScanner(
        name="pii",
        target="parsed",
        version="dlp-v1",
        policy_revision="policy-v1",
        scan_request=lambda request: {
            "schema": "celiums-dlp-response-v1",
            "content_digest": request["content_digest"],
            "policy_revision": "policy-v1",
            "findings": [],
        },
    )
    assert dlp.findings(b"safe text") == ()


def test_audit_metrics_supervisor_and_gemma_boundary(tmp_path: Path) -> None:
    snapshot = FinalizationQueueSnapshot(
        tenant=TenantId("tenant_a"),
        observed_at_us=1,
        ready=1,
        answering_due=0,
        answering_deferred=0,
        notifying_due=0,
        notifying_deferred=0,
        leased=0,
        dead_lettered=0,
        notification_attempts=0,
        oldest_claimable_age_seconds=1.5,
    )
    assert 'state="ready"} 1' in render_prometheus(snapshot)
    chain = AuditChain(tmp_path / "audit" / "events.jsonl")
    chain.append(
        occurred_at_us=1,
        event_type="test",
        subject_digest="a" * 64,
        outcome="ok",
        detail={"revision": 1},
    )
    assert len(chain.verify()) == 1
    result = run_supervised(("/bin/sh", "-c", "printf ok"), timeout_seconds=2)
    assert result.stdout == b"ok" and not result.timed_out

    text = "Verified quotation"
    hit = EvidenceHit(
        handle="passage_0123456789abcdef",
        source_id="docs",
        source_version="v1",
        text=text,
        score=1.0,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
    )
    job = AcquisitionJob(
        tenant=TenantId("tenant_a"),
        query="question",
        query_digest=hashlib.sha256(b"question").hexdigest(),
        corpus_generation="generation_one",
        policy_version="policy-v1",
        embedding_profile="fixture-v1",
        source_id="docs",
    )

    class Evidence:
        def retrieve_for_job(self, item: AcquisitionJob) -> EvidenceBundle:
            return EvidenceBundle(item.tenant, item.query_digest, item.corpus_generation, (hit,))

    class Runtime:
        identity = FrozenModelIdentity("gemma", "rev", "a" * 64, "fixture")

        def infer(self, request, *, timeout_seconds):
            del timeout_seconds
            return GovernedModelResult(
                self.identity,
                "answer",
                (QuotedClaim(request.passages[0].handle, text),),
            )

    answer = HostGemmaAnswerer(runtime=Runtime(), evidence=Evidence()).answer(
        job, timeout_seconds=5
    )
    assert answer is not None and answer.answer == text
