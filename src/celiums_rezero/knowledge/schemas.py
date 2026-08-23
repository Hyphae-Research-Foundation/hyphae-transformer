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
    SHADOW_VALIDATED = "shadow_validated"
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
            JobStatus.SHADOW_VALIDATED,
            JobStatus.POLICY_DENIED,
            JobStatus.LICENSE_UNKNOWN,
            JobStatus.SECURITY_REJECTED,
            JobStatus.FAILED,
            JobStatus.INSUFFICIENT_AFTER_INGEST,
            JobStatus.CANCELLED,
        }


class IngestMode(StrEnum):
    SIMULATED = "simulated"
    SHADOW = "shadow"
    LIVE = "live"


class FinalizationPhase(StrEnum):
    ANSWERING = "answering"
    NOTIFYING = "notifying"


class DeadLetterReason(StrEnum):
    PERMANENT = "permanent"
    RETRIES_EXHAUSTED = "retries_exhausted"


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
    resource_url: str | None = None
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
        if self.resource_url is not None and not self.permits_url(self.resource_url):
            raise ValueError("source resource URL is not permitted by its own policy")

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
class JobLease:
    tenant: TenantId
    job_id: str
    owner_id: str
    fence: int
    expires_at_us: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"job_[0-9a-f]{16}", self.job_id):
            raise ValueError("lease job ID is invalid")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.owner_id):
            raise ValueError("lease owner ID is invalid")
        if self.fence < 1 or self.expires_at_us < 1:
            raise ValueError("lease fence and expiry must be positive")


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
class SecurityScanReceipt:
    scanner: str
    version: str
    target: str
    content_digest: str
    findings: int = 0

    def __post_init__(self) -> None:
        if not self.scanner or not self.version or self.target not in {"raw", "parsed"}:
            raise ValueError("security scan identity is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_digest):
            raise ValueError("security scan digest must be lowercase SHA-256")
        if self.findings < 0:
            raise ValueError("security scan finding count cannot be negative")


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    artifact: SourceArtifact
    body: bytes
    content_digest: str
    parser: str
    parser_version: str
    scans: tuple[SecurityScanReceipt, ...]

    def __post_init__(self) -> None:
        if not self.body or not self.parser or not self.parser_version:
            raise ValueError("validated artifact parser and body are required")
        if self.content_digest != hashlib.sha256(self.body).hexdigest():
            raise ValueError("validated artifact digest does not match its bytes")
        scanner_names = [scan.scanner for scan in self.scans]
        if len(scanner_names) != len(set(scanner_names)):
            raise ValueError("validated artifact scanners must be unique")
        if any(scan.findings for scan in self.scans):
            raise ValueError("validated artifact cannot contain security findings")


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
class PublicationTarget:
    backend_id: str
    collection: int
    vector_target: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.backend_id):
            raise ValueError("publication backend ID must be lowercase SHA-256")
        if self.collection < 1 or not self.vector_target or len(self.vector_target) > 128:
            raise ValueError("publication target is invalid")


@dataclass(frozen=True, slots=True)
class PublicationAuthorization:
    tenant: TenantId
    source_id: str
    source_version: str
    corpus_generation: str
    policy_version: str
    raw_digest: str
    parsed_digest: str
    parser: str
    parser_version: str
    scans: tuple[SecurityScanReceipt, ...]
    chunk_ids: tuple[str, ...]
    chunk_digests: tuple[str, ...]
    chunk_coordinates: tuple[tuple[int, int, int], ...]
    embedding_profile: str
    idempotency_key: str
    authority: str
    target: PublicationTarget
    embedding_digests: tuple[str, ...]
    authorization_id: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.source_id,
            self.source_version,
            self.corpus_generation,
            self.policy_version,
            self.parser,
            self.parser_version,
            self.embedding_profile,
            self.authority,
        )
        if any(not value for value in required):
            raise ValueError("publication authorization identity is incomplete")
        digests = (self.raw_digest, self.parsed_digest, self.idempotency_key)
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in digests):
            raise ValueError("publication authorization digests must be lowercase SHA-256")
        if not self.chunk_ids or len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("publication authorization requires unique chunks")
        if len(self.chunk_ids) != len(self.chunk_digests) or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in self.chunk_digests
        ):
            raise ValueError("publication authorization chunk digests are invalid")
        if len(self.chunk_ids) != len(self.chunk_coordinates) or any(
            ordinal != index or not 0 <= start < end
            for index, (ordinal, start, end) in enumerate(self.chunk_coordinates)
        ):
            raise ValueError("publication authorization chunk coordinates are invalid")
        if len(self.chunk_ids) != len(self.embedding_digests) or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in self.embedding_digests
        ):
            raise ValueError("publication authorization embedding digests are invalid")
        if not self.scans or any(scan.findings for scan in self.scans):
            raise ValueError("publication authorization requires clean security scans")
        scanner_targets = {scan.scanner: scan.target for scan in self.scans}
        required_scanners = {
            "malware": "raw",
            "pii": "parsed",
            "secrets": "parsed",
            "prompt-injection": "parsed",
        }
        if scanner_targets != required_scanners:
            raise ValueError("publication authorization is missing mandatory security scans")
        if any(
            scan.content_digest
            != (self.raw_digest if scan.target == "raw" else self.parsed_digest)
            for scan in self.scans
        ):
            raise ValueError("publication authorization scan digests do not match the content")
        identity = {
            "schema": "knowledge-publication-authorization-v1",
            "tenant": self.tenant.value,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "corpus_generation": self.corpus_generation,
            "policy_version": self.policy_version,
            "raw_digest": self.raw_digest,
            "parsed_digest": self.parsed_digest,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "scans": self.scans,
            "chunk_ids": self.chunk_ids,
            "chunk_digests": self.chunk_digests,
            "chunk_coordinates": self.chunk_coordinates,
            "embedding_profile": self.embedding_profile,
            "idempotency_key": self.idempotency_key,
            "authority": self.authority,
            "target": self.target,
            "embedding_digests": self.embedding_digests,
        }
        expected = f"authorization_{content_hash(identity, length=64)}"
        if self.authorization_id is None:
            object.__setattr__(self, "authorization_id", expected)
        elif self.authorization_id != expected:
            raise ValueError("publication authorization ID does not match its contents")


@dataclass(frozen=True, slots=True)
class PreparedIngest:
    tenant: TenantId
    job_id: str
    corpus_generation: str
    idempotency_key: str
    mode: IngestMode
    chunks: tuple[EmbeddedChunk, ...]
    authorization: PublicationAuthorization | None = None
    target: PublicationTarget | None = None
    command_digest: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"job_[0-9a-f]{16}", self.job_id):
            raise ValueError("prepared ingest job ID is invalid")
        if not self.corpus_generation or not re.fullmatch(
            r"[0-9a-f]{64}", self.idempotency_key
        ):
            raise ValueError("prepared ingest generation or idempotency key is invalid")
        if not isinstance(self.mode, IngestMode) or not self.chunks:
            raise ValueError("prepared ingest mode and chunks are required")
        chunk_ids = tuple(item.chunk.chunk_id for item in self.chunks)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("prepared ingest chunk IDs must be unique")
        if self.mode is IngestMode.LIVE:
            if self.authorization is None or self.target is None:
                raise ValueError("live prepared ingest requires authorization and target")
            if (
                self.authorization.tenant != self.tenant
                or self.authorization.target != self.target
                or self.authorization.idempotency_key != self.idempotency_key
                or self.authorization.chunk_ids != chunk_ids
            ):
                raise ValueError("prepared ingest does not match its authorization")
        elif self.authorization is not None or self.target is not None:
            raise ValueError("non-live prepared ingest cannot carry live authority")
        identity = {
            "schema": "knowledge-prepared-ingest-v1",
            "tenant": self.tenant.value,
            "job_id": self.job_id,
            "corpus_generation": self.corpus_generation,
            "idempotency_key": self.idempotency_key,
            "mode": self.mode,
            "chunks": self.chunks,
            "authorization": self.authorization,
            "target": self.target,
        }
        expected = content_hash(identity, length=64)
        if self.command_digest is None:
            object.__setattr__(self, "command_digest", expected)
        elif self.command_digest != expected:
            raise ValueError("prepared ingest digest does not match its contents")


@dataclass(frozen=True, slots=True)
class PreparedNotification:
    tenant: TenantId
    job_id: str
    sink_id: str
    answer: str
    evidence_handles: tuple[str, ...]
    corpus_generation: str
    query_digest: str
    notification_id: str | None = None
    command_digest: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"job_[0-9a-f]{16}", self.job_id):
            raise ValueError("notification job ID is invalid")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.sink_id):
            raise ValueError("notification sink ID is invalid")
        if not self.answer.strip() or len(self.answer.encode()) > 1_000_000:
            raise ValueError("notification answer is empty or exceeds its byte bound")
        if (
            len(self.evidence_handles) > 1024
            or sum(len(handle) for handle in self.evidence_handles) > 128_000
            or len(self.evidence_handles) != len(set(self.evidence_handles))
            or any(not HANDLE_PATTERN.fullmatch(handle) for handle in self.evidence_handles)
        ):
            raise ValueError("notification evidence handles are invalid")
        if not self.corpus_generation or not re.fullmatch(r"[0-9a-f]{64}", self.query_digest):
            raise ValueError("notification generation or query digest is invalid")
        identity = {
            "schema": "knowledge-notification-v1",
            "tenant": self.tenant.value,
            "job_id": self.job_id,
            "sink_id": self.sink_id,
            "answer": self.answer,
            "evidence_handles": self.evidence_handles,
            "corpus_generation": self.corpus_generation,
            "query_digest": self.query_digest,
        }
        expected_notification = f"notification_{content_hash(identity, length=64)}"
        if self.notification_id is None:
            object.__setattr__(self, "notification_id", expected_notification)
        elif self.notification_id != expected_notification:
            raise ValueError("notification ID does not match its contents")
        expected_digest = content_hash(
            {**identity, "notification_id": expected_notification}, length=64
        )
        if self.command_digest is None:
            object.__setattr__(self, "command_digest", expected_digest)
        elif self.command_digest != expected_digest:
            raise ValueError("notification command digest does not match its contents")


@dataclass(frozen=True, slots=True)
class NotificationReceipt:
    tenant: TenantId
    job_id: str
    notification_id: str
    sink_id: str
    command_digest: str
    provider_receipt: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"job_[0-9a-f]{16}", self.job_id):
            raise ValueError("notification receipt job ID is invalid")
        if not re.fullmatch(r"notification_[0-9a-f]{64}", self.notification_id):
            raise ValueError("notification receipt ID is invalid")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", self.sink_id):
            raise ValueError("notification receipt sink is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.command_digest):
            raise ValueError("notification receipt command digest is invalid")
        if not self.provider_receipt or len(self.provider_receipt.encode()) > 4096:
            raise ValueError("notification provider receipt is invalid")


@dataclass(frozen=True, slots=True)
class FinalizationDeadLetter:
    tenant: TenantId
    job_id: str
    phase: FinalizationPhase
    reason: DeadLetterReason
    failures: int
    error: str
    dead_lettered_at_us: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"job_[0-9a-f]{16}", self.job_id):
            raise ValueError("dead-letter job ID is invalid")
        if self.failures < 1 or self.dead_lettered_at_us < 1:
            raise ValueError("dead-letter count and time must be positive")
        if not self.error or len(self.error.encode()) > 4096:
            raise ValueError("dead-letter error is invalid")


@dataclass(frozen=True, slots=True)
class FinalizationQueueSnapshot:
    tenant: TenantId
    observed_at_us: int
    ready: int
    answering_due: int
    answering_deferred: int
    notifying_due: int
    notifying_deferred: int
    leased: int
    dead_lettered: int
    notification_attempts: int
    oldest_claimable_age_seconds: float

    def __post_init__(self) -> None:
        counts = (
            self.ready,
            self.answering_due,
            self.answering_deferred,
            self.notifying_due,
            self.notifying_deferred,
            self.leased,
            self.dead_lettered,
            self.notification_attempts,
        )
        if self.observed_at_us < 1 or any(value < 0 for value in counts):
            raise ValueError("finalization queue snapshot counters are invalid")
        if not isfinite(self.oldest_claimable_age_seconds) or (
            self.oldest_claimable_age_seconds < 0
        ):
            raise ValueError("finalization queue snapshot age is invalid")


@dataclass(frozen=True, slots=True)
class IngestReceipt:
    tenant: TenantId
    source_id: str
    source_version: str
    corpus_generation: str
    chunk_ids: tuple[str, ...]
    idempotency_key: str
    replayed: bool
    published: bool = False
    mode: IngestMode = IngestMode.SHADOW
    target: PublicationTarget | None = None
    authorization_id: str | None = None
    backend_receipt_digest: str | None = None
    backend_receipt_json: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, IngestMode):
            raise ValueError("ingest receipt mode must be typed")
        if not isinstance(self.replayed, bool) or not isinstance(self.published, bool):
            raise ValueError("ingest receipt flags must be booleans")
        if not self.source_id or not self.source_version or not self.corpus_generation:
            raise ValueError("ingest receipt source and generation are required")
        if not self.chunk_ids or len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("ingest receipt requires unique chunks")
        if not re.fullmatch(r"[0-9a-f]{64}", self.idempotency_key):
            raise ValueError("ingest idempotency key must be lowercase SHA-256")
        if self.backend_receipt_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.backend_receipt_digest
        ):
            raise ValueError("backend receipt digest must be lowercase SHA-256")
        if self.authorization_id is not None and not re.fullmatch(
            r"authorization_[0-9a-f]{64}", self.authorization_id
        ):
            raise ValueError("publication authorization ID is invalid")
        if self.mode is IngestMode.LIVE:
            if not self.published or any(
                value is None
                for value in (
                    self.target,
                    self.authorization_id,
                    self.backend_receipt_digest,
                    self.backend_receipt_json,
                )
            ):
                raise ValueError("live ingest receipt evidence must be complete")
            assert self.backend_receipt_json is not None
            if len(self.backend_receipt_json.encode()) > 1_000_000:
                raise ValueError("backend ingest receipt exceeds its byte bound")
        elif self.authorization_id is not None or any(
            value is not None
            for value in (self.target, self.backend_receipt_digest, self.backend_receipt_json)
        ):
            raise ValueError("non-live ingest receipt cannot carry live publication evidence")
        if self.mode is IngestMode.SHADOW and self.published:
            raise ValueError("shadow ingest receipt cannot be published")
        if self.mode is IngestMode.SIMULATED and not self.published:
            raise ValueError("simulated ingest receipt must represent stored fixture evidence")


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    tenant: TenantId
    source_id: str
    source_version: str
    source_url: str
    policy_version: str
    content_type: str
    license_id: str
    raw_bytes: int
    raw_digest: str
    chunk_ids: tuple[str, ...]
    embedding_profile: str
    ingest_idempotency_key: str
    published: bool

    def __post_init__(self) -> None:
        if not self.source_url.startswith("https://") or self.raw_bytes < 1:
            raise ValueError("acquisition source URL and byte count are invalid")
        digests = (self.raw_digest, self.ingest_idempotency_key)
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in digests):
            raise ValueError("acquisition receipt digests must be lowercase SHA-256")
        if not self.chunk_ids or not self.embedding_profile:
            raise ValueError("acquisition receipt chunks and embedding profile are required")


def _host_is_forbidden(host: str) -> bool:
    if host in {"localhost", "metadata.google.internal"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global
