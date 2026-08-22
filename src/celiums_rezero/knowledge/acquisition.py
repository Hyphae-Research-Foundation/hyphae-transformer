"""Deterministic Phase 2 acquisition simulator with no network or Hyphae writes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from celiums_rezero.knowledge.coordinator import KnowledgeCoordinator
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EmbeddedChunk,
    IngestReceipt,
    JobStatus,
    KnowledgeChunk,
    SourceArtifact,
    TenantId,
)


class AcquisitionError(RuntimeError):
    """Acquisition could not complete under the declared policy."""


class SourceConnector(Protocol):
    def acquire(self, tenant: TenantId, source_id: str, query: str) -> SourceArtifact: ...


class ChunkEmbedder(Protocol):
    @property
    def profile(self) -> str: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


class KnowledgeIngestor(Protocol):
    def ingest(
        self,
        tenant: TenantId,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        corpus_generation: str,
        idempotency_key: str,
    ) -> IngestReceipt: ...


class IngestVerifier(Protocol):
    def verify(self, receipt: IngestReceipt, expected: tuple[EmbeddedChunk, ...]) -> bool: ...


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    max_chunk_bytes: int = 1024
    overlap_bytes: int = 128
    max_chunks: int = 10_000

    def __post_init__(self) -> None:
        if self.max_chunk_bytes < 64:
            raise ValueError("chunk size must be at least 64 bytes")
        if not 0 <= self.overlap_bytes < self.max_chunk_bytes:
            raise ValueError("chunk overlap is invalid")
        if self.max_chunks < 1:
            raise ValueError("maximum chunk count must be positive")


@dataclass(frozen=True, slots=True)
class AcquisitionOutcome:
    job: AcquisitionJob
    artifact_digest: str
    receipt: IngestReceipt | None = None


class AcquisitionWorker:
    """Runs one queued job through the full deterministic simulated lifecycle."""

    def __init__(
        self,
        *,
        coordinator: KnowledgeCoordinator,
        connector: SourceConnector,
        embedder: ChunkEmbedder,
        ingestor: KnowledgeIngestor,
        verifier: IngestVerifier,
        chunking: ChunkingPolicy | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.connector = connector
        self.embedder = embedder
        self.ingestor = ingestor
        self.verifier = verifier
        self.chunking = ChunkingPolicy() if chunking is None else chunking

    def run(self, tenant: TenantId, job_id: str) -> AcquisitionOutcome:
        job = self.coordinator.job_status(tenant, job_id)
        if job is None:
            raise KeyError("job is not visible in this tenant")
        if job.status is not JobStatus.QUEUED:
            raise ValueError("only queued jobs can be acquired")
        source_policy = self.coordinator.acquisition.source(job.source_id)
        if source_policy is None:
            denied = self.coordinator.store.transition(
                tenant, job_id, JobStatus.POLICY_DENIED, failure="source is not allowlisted"
            )
            return AcquisitionOutcome(denied, "")
        self.coordinator.store.transition(tenant, job_id, JobStatus.ACQUIRING)
        try:
            artifact = self.connector.acquire(tenant, job.source_id, job.query)
            if artifact.tenant != tenant or artifact.source_id != job.source_id:
                raise AcquisitionError("source connector crossed its tenant or source binding")
            if len(artifact.body) > source_policy.max_download_bytes:
                raise AcquisitionError("source artifact exceeds its byte budget")
            if artifact.content_type not in source_policy.allowed_mime_types:
                raise AcquisitionError("source artifact type is not allowed")
            self.coordinator.store.transition(tenant, job_id, JobStatus.QUARANTINED)
            self.coordinator.store.transition(tenant, job_id, JobStatus.VALIDATING)
            if artifact.license_id not in source_policy.allowed_license_ids:
                terminal = self.coordinator.store.transition(
                    tenant,
                    job_id,
                    JobStatus.LICENSE_UNKNOWN,
                    failure=f"license is not allowed: {artifact.license_id}",
                )
                return AcquisitionOutcome(terminal, artifact.content_digest)
            self.coordinator.store.transition(tenant, job_id, JobStatus.CHUNKING)
            chunks = chunk_artifact(artifact, self.chunking)
            self.coordinator.store.transition(tenant, job_id, JobStatus.EMBEDDING)
            embedded = tuple(
                EmbeddedChunk(chunk, self.embedder.profile, self.embedder.embed(chunk.text))
                for chunk in chunks
            )
            self.coordinator.store.transition(tenant, job_id, JobStatus.INGESTING)
            idempotency_key = ingest_idempotency_key(job, embedded, self.embedder.profile)
            receipt = self.ingestor.ingest(
                tenant,
                embedded,
                corpus_generation=job.corpus_generation,
                idempotency_key=idempotency_key,
            )
            self.coordinator.store.transition(tenant, job_id, JobStatus.VERIFYING)
            if not self.verifier.verify(receipt, embedded):
                raise AcquisitionError("ingested evidence failed deterministic verification")
            ready = self.coordinator.store.transition(tenant, job_id, JobStatus.READY)
            return AcquisitionOutcome(ready, artifact.content_digest, receipt)
        except Exception as error:
            current = self.coordinator.job_status(tenant, job_id)
            assert current is not None
            if current.status.terminal:
                return AcquisitionOutcome(current, "")
            failed = self.coordinator.store.transition(
                tenant,
                job_id,
                JobStatus.FAILED,
                failure=f"{type(error).__name__}: {error}",
            )
            return AcquisitionOutcome(failed, "")

    def finalize(
        self,
        tenant: TenantId,
        job_id: str,
        *,
        evidence_sufficient: bool,
    ) -> AcquisitionJob:
        job = self.coordinator.job_status(tenant, job_id)
        if job is None:
            raise KeyError("job is not visible in this tenant")
        if job.status is not JobStatus.READY:
            raise ValueError("only ready jobs can be finalized")
        answering = self.coordinator.store.transition(tenant, job_id, JobStatus.ANSWERING)
        if not evidence_sufficient:
            return self.coordinator.store.transition(
                tenant,
                job_id,
                JobStatus.INSUFFICIENT_AFTER_INGEST,
                failure="newly ingested evidence remains insufficient",
            )
        assert answering.status is JobStatus.ANSWERING
        self.coordinator.store.transition(tenant, job_id, JobStatus.NOTIFYING)
        return self.coordinator.store.transition(tenant, job_id, JobStatus.COMPLETED)


def chunk_artifact(
    artifact: SourceArtifact, policy: ChunkingPolicy
) -> tuple[KnowledgeChunk, ...]:
    text = artifact.body.decode("utf-8", errors="strict")
    encoded = text.encode()
    step = policy.max_chunk_bytes - policy.overlap_bytes
    chunks: list[KnowledgeChunk] = []
    start = 0
    ordinal = 0
    while start < len(encoded):
        if len(chunks) >= policy.max_chunks:
            raise AcquisitionError("source artifact exceeds the maximum chunk count")
        end = min(start + policy.max_chunk_bytes, len(encoded))
        while end > start:
            try:
                piece = encoded[start:end].decode()
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            raise AcquisitionError("source artifact cannot be split into UTF-8 chunks")
        digest = hashlib.sha256(piece.encode()).hexdigest()
        identity = hashlib.sha256(
            b"knowledge-chunk-v1\0"
            + artifact.content_digest.encode()
            + start.to_bytes(8, "big")
            + end.to_bytes(8, "big")
        ).hexdigest()[:16]
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"chunk_{identity}",
                source_id=artifact.source_id,
                source_version=artifact.source_version,
                ordinal=ordinal,
                byte_start=start,
                byte_end=end,
                text=piece,
                content_digest=digest,
            )
        )
        if end == len(encoded):
            break
        start = _utf8_boundary(encoded, start + step)
        ordinal += 1
    return tuple(chunks)


def _utf8_boundary(encoded: bytes, target: int) -> int:
    bounded = min(max(target, 0), len(encoded))
    while bounded < len(encoded) and encoded[bounded] & 0b1100_0000 == 0b1000_0000:
        bounded += 1
    return bounded


def ingest_idempotency_key(
    job: AcquisitionJob,
    chunks: tuple[EmbeddedChunk, ...],
    embedding_profile: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"knowledge-ingest-v1\0")
    digest.update(job.tenant.value.encode())
    digest.update(job.source_id.encode())
    digest.update(job.corpus_generation.encode())
    digest.update(embedding_profile.encode())
    for item in chunks:
        digest.update(item.chunk.chunk_id.encode())
        digest.update(item.chunk.content_digest.encode())
        for value in item.values:
            digest.update(float(value).hex().encode())
    return digest.hexdigest()


class InMemorySourceConnector:
    """Explicitly provisioned source artifacts; no URL or network surface exists."""

    def __init__(self, artifacts: dict[tuple[str, str], SourceArtifact]) -> None:
        self._artifacts = dict(artifacts)

    def acquire(self, tenant: TenantId, source_id: str, query: str) -> SourceArtifact:
        del query
        artifact = self._artifacts.get((tenant.value, source_id))
        if artifact is None:
            raise AcquisitionError("source connector has no tenant-bound artifact")
        return artifact


class InMemoryKnowledgeIndex:
    """Idempotent tenant-local simulated Hyphae ingest and verification authority."""

    def __init__(self) -> None:
        self._receipts: dict[tuple[str, str], IngestReceipt] = {}
        self._chunks: dict[str, dict[str, EmbeddedChunk]] = {}

    def ingest(
        self,
        tenant: TenantId,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        corpus_generation: str,
        idempotency_key: str,
    ) -> IngestReceipt:
        key = (tenant.value, idempotency_key)
        existing = self._receipts.get(key)
        if existing is not None:
            return IngestReceipt(
                tenant=existing.tenant,
                source_id=existing.source_id,
                source_version=existing.source_version,
                corpus_generation=existing.corpus_generation,
                chunk_ids=existing.chunk_ids,
                idempotency_key=existing.idempotency_key,
                replayed=True,
            )
        if not chunks:
            raise AcquisitionError("ingest requires at least one embedded chunk")
        source_id = chunks[0].chunk.source_id
        source_version = chunks[0].chunk.source_version
        if any(
            item.chunk.source_id != source_id or item.chunk.source_version != source_version
            for item in chunks
        ):
            raise AcquisitionError("ingest batch mixes source identities")
        tenant_chunks = self._chunks.setdefault(tenant.value, {})
        for item in chunks:
            prior = tenant_chunks.get(item.chunk.chunk_id)
            if prior is not None and prior != item:
                raise AcquisitionError("chunk identity conflicts with existing content")
        tenant_chunks.update((item.chunk.chunk_id, item) for item in chunks)
        receipt = IngestReceipt(
            tenant=tenant,
            source_id=source_id,
            source_version=source_version,
            corpus_generation=corpus_generation,
            chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            idempotency_key=idempotency_key,
            replayed=False,
        )
        self._receipts[key] = receipt
        return receipt

    def verify(self, receipt: IngestReceipt, expected: tuple[EmbeddedChunk, ...]) -> bool:
        if receipt.tenant.value not in self._chunks:
            return False
        stored = self._chunks[receipt.tenant.value]
        expected_ids = tuple(item.chunk.chunk_id for item in expected)
        return receipt.chunk_ids == expected_ids and all(
            stored.get(item.chunk.chunk_id) == item for item in expected
        )

    def tenant_chunk_count(self, tenant: TenantId) -> int:
        return len(self._chunks.get(tenant.value, {}))

    def chunks_for_receipt(self, receipt: IngestReceipt) -> tuple[EmbeddedChunk, ...]:
        stored = self._chunks.get(receipt.tenant.value, {})
        return tuple(stored[chunk_id] for chunk_id in receipt.chunk_ids)
