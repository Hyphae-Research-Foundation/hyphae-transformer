"""Versioned knowledge, evidence, source, and asynchronous job contracts."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from urllib.parse import urlparse

from celiums_rezero.lab.serialization import content_hash

TENANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
HANDLE_PATTERN = re.compile(r"^[a-z]+_[0-9a-f]{16}$")
HOST_PATTERN = re.compile(r"^[a-z0-9.-]+$")
NO_KNOWLEDGE_MESSAGE = "No poseo este conocimiento, descargando..."


@dataclass(frozen=True, slots=True)
class TenantId:
    value: str

    def __post_init__(self) -> None:
        if not TENANT_PATTERN.fullmatch(self.value):
            raise ValueError("tenant ID must be a normalized opaque identifier")


class SufficiencyDecision(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    ABSENT = "absent"
    BLOCKED = "blocked"
    PENDING = "pending"


class JobStatus(StrEnum):
    QUEUED = "queued"
    ACQUIRING = "acquiring"
    QUARANTINED = "quarantined"
    VALIDATING = "validating"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INGESTING = "ingesting"
    VERIFYING = "verifying"
    READY = "ready"
    ANSWERING = "answering"
    NOTIFYING = "notifying"
    COMPLETED = "completed"
    POLICY_DENIED = "policy_denied"
    LICENSE_UNKNOWN = "license_unknown"
    SECURITY_REJECTED = "security_rejected"
    FAILED = "failed"
    INSUFFICIENT_AFTER_INGEST = "insufficient_after_ingest"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.POLICY_DENIED,
            JobStatus.LICENSE_UNKNOWN,
            JobStatus.SECURITY_REJECTED,
            JobStatus.FAILED,
            JobStatus.INSUFFICIENT_AFTER_INGEST,
            JobStatus.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class EvidenceHit:
    handle: str
    source_id: str
    source_version: str
    text: str
    score: float
    content_digest: str
    trusted: bool = True
    active: bool = True

    def __post_init__(self) -> None:
        if not HANDLE_PATTERN.fullmatch(self.handle):
            raise ValueError("evidence handle is invalid")
        if not self.source_id or not self.source_version or not self.text.strip():
            raise ValueError("evidence source, version, and text are required")
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("evidence score must be finite and in [0, 1]")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_digest):
            raise ValueError("evidence digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    tenant: TenantId
    query_digest: str
    corpus_generation: str
    hits: tuple[EvidenceHit, ...]
    snapshot_fingerprint: str | None = None
    approximate: bool = False
    conflicting: bool = False
    blocked: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.query_digest):
            raise ValueError("query digest must be lowercase SHA-256")
        if not self.corpus_generation:
            raise ValueError("corpus generation is required")
        handles = [hit.handle for hit in self.hits]
        if len(handles) != len(set(handles)):
            raise ValueError("evidence handles must be unique")
        if self.snapshot_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.snapshot_fingerprint
        ):
            raise ValueError("snapshot fingerprint must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SufficiencyPolicy:
    minimum_score: float = 0.72
    minimum_margin: float = 0.08
    minimum_trusted_hits: int = 1
    allow_approximate: bool = False

    def __post_init__(self) -> None:
        values = (self.minimum_score, self.minimum_margin)
        if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("sufficiency thresholds must be finite and in [0, 1]")
        if self.minimum_trusted_hits < 1:
            raise ValueError("minimum_trusted_hits must be positive")

    def decide(self, bundle: EvidenceBundle) -> SufficiencyDecision:
        if bundle.blocked:
            return SufficiencyDecision.BLOCKED
        if bundle.conflicting:
            return SufficiencyDecision.CONFLICT
        active = tuple(hit for hit in bundle.hits if hit.active and hit.trusted)
        if not active:
            return SufficiencyDecision.ABSENT
        if bundle.approximate and not self.allow_approximate:
            return SufficiencyDecision.PARTIAL
        ordered = sorted(active, key=lambda hit: (-hit.score, hit.handle))
        if ordered[0].score < self.minimum_score:
            return SufficiencyDecision.ABSENT
        if len(ordered) > 1 and ordered[0].score - ordered[1].score < self.minimum_margin:
            return SufficiencyDecision.PARTIAL
        if len(ordered) < self.minimum_trusted_hits:
            return SufficiencyDecision.PARTIAL
        return SufficiencyDecision.SUPPORTED


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    source_id: str
    allowed_hosts: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...] = ("/",)
    allowed_mime_types: tuple[str, ...] = ("text/plain", "text/html", "application/pdf")
    allowed_license_ids: tuple[str, ...] = ()
    max_download_bytes: int = 20_000_000
    max_redirects: int = 0

    def __post_init__(self) -> None:
        if not self.source_id or not self.allowed_hosts:
            raise ValueError("source ID and hosts are required")
        hosts = tuple(host.lower().rstrip(".") for host in self.allowed_hosts)
        if any(not HOST_PATTERN.fullmatch(host) or _host_is_forbidden(host) for host in hosts):
            raise ValueError("source hosts must be public normalized DNS names")
        if any(
            not path.startswith("/") or ".." in path.split("/")
            for path in self.allowed_path_prefixes
        ):
            raise ValueError("source path prefixes are invalid")
        if self.max_download_bytes < 1 or not 0 <= self.max_redirects <= 5:
            raise ValueError("source download limits are invalid")
        object.__setattr__(self, "allowed_hosts", hosts)

    def permits_url(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment:
            return False
        if parsed.port not in {None, 443} or host not in self.allowed_hosts:
            return False
        if _host_is_forbidden(host):
            return False
        path = parsed.path or "/"
        return any(path.startswith(prefix) for prefix in self.allowed_path_prefixes)


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    version: str
    sources: tuple[SourcePolicy, ...]
    automatic: bool = True
    max_active_jobs: int = 4
    max_jobs_per_day: int = 100

    def __post_init__(self) -> None:
        if not self.version or not self.sources:
            raise ValueError("acquisition policy version and sources are required")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        if min(self.max_active_jobs, self.max_jobs_per_day) < 1:
            raise ValueError("acquisition job limits must be positive")

    def source(self, source_id: str) -> SourcePolicy | None:
        return next((source for source in self.sources if source.source_id == source_id), None)


@dataclass(frozen=True, slots=True)
class AcquisitionJob:
    tenant: TenantId
    query: str
    query_digest: str
    corpus_generation: str
    policy_version: str
    embedding_profile: str
    source_id: str
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    failure: str | None = None
    job_id: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.source_id or not self.policy_version:
            raise ValueError("job query, source, and policy are required")
        if self.attempts < 0:
            raise ValueError("job attempts cannot be negative")
        identity = {
            "tenant": self.tenant.value,
            "query_digest": self.query_digest,
            "corpus_generation": self.corpus_generation,
            "policy_version": self.policy_version,
            "embedding_profile": self.embedding_profile,
            "source_id": self.source_id,
        }
        expected = f"job_{content_hash(identity)}"
        if self.job_id is None:
            object.__setattr__(self, "job_id", expected)
        elif self.job_id != expected:
            raise ValueError("job ID does not match its immutable identity")


@dataclass(frozen=True, slots=True)
class KnowledgeResponse:
    status: str
    answer: str
    decision: SufficiencyDecision
    job_id: str | None = None
    deduplicated: bool = False
    evidence_handles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    tenant: TenantId
    source_id: str
    source_version: str
    body: bytes
    content_type: str
    license_id: str
    content_digest: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_version or not self.body:
            raise ValueError("source artifact identity and body are required")
        if not self.content_type or not self.license_id:
            raise ValueError("source artifact type and license are required")
        if self.content_digest != hashlib.sha256(self.body).hexdigest():
            raise ValueError("source artifact digest does not match its bytes")


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    source_version: str
    ordinal: int
    byte_start: int
    byte_end: int
    text: str
    content_digest: str

    def __post_init__(self) -> None:
        if not HANDLE_PATTERN.fullmatch(self.chunk_id) or not self.text:
            raise ValueError("knowledge chunk identity and text are required")
        if self.ordinal < 0 or not 0 <= self.byte_start < self.byte_end:
            raise ValueError("knowledge chunk coordinates are invalid")
        if self.content_digest != hashlib.sha256(self.text.encode()).hexdigest():
            raise ValueError("knowledge chunk digest does not match its text")


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: KnowledgeChunk
    embedding_profile: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.embedding_profile or not self.values:
            raise ValueError("embedding profile and values are required")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("embedding values must be finite")


@dataclass(frozen=True, slots=True)
class IngestReceipt:
    tenant: TenantId
    source_id: str
    source_version: str
    corpus_generation: str
    chunk_ids: tuple[str, ...]
    idempotency_key: str
    replayed: bool

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_version or not self.corpus_generation:
            raise ValueError("ingest receipt source and generation are required")
        if not self.chunk_ids or len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("ingest receipt requires unique chunks")
        if not re.fullmatch(r"[0-9a-f]{64}", self.idempotency_key):
            raise ValueError("ingest idempotency key must be lowercase SHA-256")


def _host_is_forbidden(host: str) -> bool:
    if host in {"localhost", "metadata.google.internal"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global
