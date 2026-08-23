"""Tenant-isolated job stores, lease fencing, and durable ingest outboxes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EmbeddedChunk,
    IngestMode,
    JobLease,
    JobStatus,
    KnowledgeChunk,
    PreparedIngest,
    PreparedNotification,
    PublicationAuthorization,
    PublicationTarget,
    SecurityScanReceipt,
    TenantId,
)
from celiums_rezero.lab.serialization import canonical_json, content_hash

_SCHEMA_VERSION = 2
_MAX_OUTBOX_BYTES = 32 * 1024 * 1024
_LEASE_OWNER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class JobStore(Protocol):
    def enqueue(self, job: AcquisitionJob) -> tuple[AcquisitionJob, bool]: ...

    def enqueue_bounded(
        self,
        job: AcquisitionJob,
        *,
        max_active_jobs: int,
        max_jobs_per_day: int,
    ) -> tuple[AcquisitionJob, bool] | None: ...

    def get(self, tenant: TenantId, job_id: str) -> AcquisitionJob | None: ...

    def transition(
        self,
        tenant: TenantId,
        job_id: str,
        status: JobStatus,
        *,
        failure: str | None = None,
    ) -> AcquisitionJob: ...

    def active_count(self, tenant: TenantId) -> int: ...


class InMemoryTenantStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, AcquisitionJob]] = {}

    def enqueue(self, job: AcquisitionJob) -> tuple[AcquisitionJob, bool]:
        assert job.job_id is not None
        tenant_jobs = self._jobs.setdefault(job.tenant.value, {})
        existing = tenant_jobs.get(job.job_id)
        if existing is not None:
            return existing, True
        tenant_jobs[job.job_id] = job
        return job, False

    def enqueue_bounded(
        self,
        job: AcquisitionJob,
        *,
        max_active_jobs: int,
        max_jobs_per_day: int,
    ) -> tuple[AcquisitionJob, bool] | None:
        assert job.job_id is not None
        existing = self.get(job.tenant, job.job_id)
        if existing is not None:
            return existing, True
        jobs = tuple(self._jobs.get(job.tenant.value, {}).values())
        cutoff = _now_us() - 86_400_000_000
        daily = sum(_iso_to_us(item.created_at) >= cutoff for item in jobs)
        if self.active_count(job.tenant) >= max_active_jobs or daily >= max_jobs_per_day:
            return None
        return self.enqueue(job)

    def get(self, tenant: TenantId, job_id: str) -> AcquisitionJob | None:
        return self._jobs.get(tenant.value, {}).get(job_id)

    def transition(
        self,
        tenant: TenantId,
        job_id: str,
        status: JobStatus,
        *,
        failure: str | None = None,
    ) -> AcquisitionJob:
        job = self.get(tenant, job_id)
        if job is None:
            raise KeyError("job is not visible in this tenant")
        _validate_transition(job.status, status)
        updated = replace(
            job,
            status=status,
            attempts=job.attempts + (1 if status is JobStatus.ACQUIRING else 0),
            updated_at=datetime.now(UTC).isoformat(),
            failure=failure,
        )
        self._jobs[tenant.value][job_id] = updated
        return updated

    def active_count(self, tenant: TenantId) -> int:
        return sum(
            job.status in _ACQUISITION_ACTIVE
            for job in self._jobs.get(tenant.value, {}).values()
        )


class SQLiteTenantStore:
    """One WAL-backed SQLite authority bound to exactly one tenant."""

    def __init__(
        self,
        path: Path | str,
        *,
        tenant: TenantId,
        timeout_seconds: float = 5.0,
        clock_us: Callable[[], int] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("SQLite timeout must be positive")
        self.path = Path(path)
        self.tenant = tenant
        self.timeout_seconds = timeout_seconds
        self._clock_us = _now_us if clock_us is None else clock_us
        created = self._prepare_path()
        self._initializing = created
        self._migrating = False
        if not created:
            self._migrating = self._verify_existing_database() == 1
        with self._connect() as connection:
            if created:
                self._initialize_new(connection)
            else:
                self._configure_existing(connection)
            self._check_integrity(connection)
        self._initializing = False

    def enqueue(self, job: AcquisitionJob) -> tuple[AcquisitionJob, bool]:
        result = self.enqueue_bounded(
            job, max_active_jobs=2**31 - 1, max_jobs_per_day=2**31 - 1
        )
        assert result is not None
        return result

    def enqueue_bounded(
        self,
        job: AcquisitionJob,
        *,
        max_active_jobs: int,
        max_jobs_per_day: int,
    ) -> tuple[AcquisitionJob, bool] | None:
        self._require_tenant(job.tenant)
        if min(max_active_jobs, max_jobs_per_day) < 1:
            raise ValueError("job quotas must be positive")
        if (
            job.status is not JobStatus.QUEUED
            or job.attempts != 0
            or job.failure is not None
        ):
            raise ValueError("new durable jobs must be pristine and queued")
        assert job.job_id is not None
        identity_digest = _job_identity_digest(job)
        now = self._clock_us()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)
            ).fetchone()
            if row is not None:
                existing = _row_job(row, self.tenant)
                if row["identity_digest"] != identity_digest or not _same_identity(existing, job):
                    raise RuntimeError("durable job identity collision")
                return existing, True
            active = connection.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status IN ({_acquisition_placeholders()})",
                tuple(status.value for status in _ACQUISITION_ACTIVE),
            ).fetchone()[0]
            daily = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE created_at_us >= ?", (now - 86_400_000_000,)
            ).fetchone()[0]
            if active >= max_active_jobs or daily >= max_jobs_per_day:
                return None
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, identity_digest, query, query_digest, corpus_generation,
                    policy_version, embedding_profile, source_id, status, attempts,
                    created_at_us, updated_at_us, failure
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    identity_digest,
                    job.query,
                    job.query_digest,
                    job.corpus_generation,
                    job.policy_version,
                    job.embedding_profile,
                    job.source_id,
                    job.status.value,
                    job.attempts,
                    now,
                    now,
                    job.failure,
                ),
            )
            connection.execute(
                "INSERT INTO leases (job_id, fence) VALUES (?, 0)", (job.job_id,)
            )
        stored = self.get(job.tenant, job.job_id)
        assert stored is not None
        return stored, False

    def get(self, tenant: TenantId, job_id: str) -> AcquisitionJob | None:
        self._require_tenant(tenant)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return None if row is None else _row_job(row, tenant)

    def transition(
        self,
        tenant: TenantId,
        job_id: str,
        status: JobStatus,
        *,
        failure: str | None = None,
    ) -> AcquisitionJob:
        self._require_tenant(tenant)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError("job is not visible in this tenant")
            job = _row_job(row, tenant)
            lease = connection.execute(
                "SELECT owner_id FROM leases WHERE job_id = ?", (job_id,)
            ).fetchone()
            if lease is None or lease["owner_id"] is not None:
                raise PermissionError("durable job mutation requires an unleased control state")
            if (job.status, status) not in {
                (JobStatus.QUEUED, JobStatus.CANCELLED),
                (JobStatus.QUEUED, JobStatus.POLICY_DENIED),
            }:
                raise PermissionError("durable job transitions require a fenced lease")
            _validate_transition(job.status, status)
            attempts = job.attempts + (1 if status is JobStatus.ACQUIRING else 0)
            connection.execute(
                "UPDATE jobs SET status = ?, attempts = ?, updated_at_us = ?, failure = ? "
                "WHERE job_id = ? AND status = ?",
                (status.value, attempts, self._clock_us(), failure, job_id, job.status.value),
            )
        updated = self.get(tenant, job_id)
        assert updated is not None
        return updated

    def active_count(self, tenant: TenantId) -> int:
        self._require_tenant(tenant)
        with self._connect() as connection:
            count = connection.execute(
                f"SELECT COUNT(*) FROM jobs WHERE status IN ({_acquisition_placeholders()})",
                tuple(status.value for status in _ACQUISITION_ACTIVE),
            ).fetchone()[0]
        return int(count)

    def claim(
        self,
        *,
        owner_id: str,
        lease_seconds: float,
        job_id: str | None = None,
    ) -> tuple[AcquisitionJob, JobLease] | None:
        if not _LEASE_OWNER_PATTERN.fullmatch(owner_id) or lease_seconds <= 0:
            raise ValueError("lease owner and duration are required")
        with self._transaction() as connection:
            now = self._clock_us()
            expires = now + int(lease_seconds * 1_000_000)
            parameters: list[object] = [now]
            predicate = ""
            if job_id is not None:
                predicate = "AND jobs.job_id = ?"
                parameters.append(job_id)
            row = connection.execute(
                f"""
                SELECT jobs.*, leases.fence
                FROM jobs JOIN leases USING (job_id)
                WHERE jobs.status IN ('queued', 'ingesting', 'verifying')
                  AND (leases.owner_id IS NULL OR leases.expires_at_us <= ?)
                  {predicate}
                ORDER BY jobs.created_at_us, jobs.job_id
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            fence = int(row["fence"]) + 1
            connection.execute(
                "UPDATE leases SET owner_id = ?, fence = ?, acquired_at_us = ?, "
                "expires_at_us = ? WHERE job_id = ?",
                (owner_id, fence, now, expires, row["job_id"]),
            )
            job = _row_job(row, self.tenant)
            if job.status is JobStatus.QUEUED:
                connection.execute(
                    "UPDATE jobs SET status = 'acquiring', attempts = attempts + 1, "
                    "updated_at_us = ? WHERE job_id = ? AND status = 'queued'",
                    (now, job.job_id),
                )
                job = replace(
                    job,
                    status=JobStatus.ACQUIRING,
                    attempts=job.attempts + 1,
                    updated_at=_us_to_iso(now),
                )
        return job, JobLease(self.tenant, cast(str, job.job_id), owner_id, fence, expires)

    def recover_expired(
        self, *, job_id: str | None = None, limit: int = 100
    ) -> int:
        """Requeue expired pre-outbox work; exact ingest phases resume in place."""
        if limit < 1 or limit > 10_000:
            raise ValueError("recovery limit is invalid")
        now = self._clock_us()
        parameters: list[object] = [now]
        predicate = ""
        if job_id is not None:
            predicate = "AND jobs.job_id = ?"
            parameters.append(job_id)
        with self._transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT jobs.job_id
                FROM jobs JOIN leases USING (job_id)
                WHERE jobs.status IN (
                    'acquiring', 'quarantined', 'validating', 'chunking', 'embedding'
                )
                  AND leases.owner_id IS NOT NULL
                  AND leases.expires_at_us <= ?
                  {predicate}
                ORDER BY jobs.updated_at_us, jobs.job_id
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
            identities = tuple(cast(str, row["job_id"]) for row in rows)
            for identity in identities:
                if connection.execute(
                    "SELECT 1 FROM ingest_outbox WHERE job_id = ?", (identity,)
                ).fetchone() is not None:
                    raise RuntimeError("pre-ingest job unexpectedly owns a durable outbox")
                connection.execute(
                    "UPDATE jobs SET status = 'queued', updated_at_us = ?, failure = NULL "
                    "WHERE job_id = ?",
                    (now, identity),
                )
                connection.execute(
                    "UPDATE leases SET owner_id = NULL, acquired_at_us = NULL, "
                    "expires_at_us = NULL WHERE job_id = ?",
                    (identity,),
                )
        return len(identities)

    def renew(self, lease: JobLease, *, lease_seconds: float) -> JobLease:
        self._require_lease(lease)
        if lease_seconds <= 0:
            raise ValueError("lease duration must be positive")
        with self._transaction() as connection:
            now = self._clock_us()
            expires = now + int(lease_seconds * 1_000_000)
            cursor = connection.execute(
                "UPDATE leases SET expires_at_us = ? WHERE job_id = ? AND owner_id = ? "
                "AND fence = ? AND expires_at_us > ?",
                (expires, lease.job_id, lease.owner_id, lease.fence, now),
            )
            if cursor.rowcount != 1:
                raise PermissionError("job lease is absent, stale, or expired")
        return replace(lease, expires_at_us=expires)

    def transition_leased(
        self,
        lease: JobLease,
        status: JobStatus,
        *,
        failure: str | None = None,
    ) -> AcquisitionJob:
        self._require_lease(lease)
        now = self._clock_us()
        with self._transaction() as connection:
            job = self._leased_job(connection, lease, now)
            if (job.status, status) in {
                (JobStatus.EMBEDDING, JobStatus.INGESTING),
                (JobStatus.INGESTING, JobStatus.VERIFYING),
            }:
                raise PermissionError("ingest transitions require the durable outbox APIs")
            if job.status is JobStatus.VERIFYING and status in {
                JobStatus.READY,
                JobStatus.SHADOW_VALIDATED,
            }:
                raise PermissionError(
                    "verification success requires the durable completion API"
                )
            if job.status in {JobStatus.READY, JobStatus.ANSWERING, JobStatus.NOTIFYING}:
                raise PermissionError("durable finalization requires a notification outbox")
            _validate_transition(job.status, status)
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, updated_at_us = ?, failure = ? "
                "WHERE job_id = ? AND status = ?",
                (status.value, now, failure, lease.job_id, job.status.value),
            )
            if cursor.rowcount != 1:
                raise PermissionError("job changed while its lease was held")
        updated = self.get(self.tenant, lease.job_id)
        assert updated is not None
        return updated

    def complete_verification(
        self,
        lease: JobLease,
        *,
        receipt_json: str,
        status: JobStatus,
    ) -> AcquisitionJob:
        from celiums_rezero.knowledge.publication import decode_ingest_receipt

        if status not in {JobStatus.READY, JobStatus.SHADOW_VALIDATED}:
            raise ValueError("verification completion status is invalid")
        self._require_lease(lease)
        now = self._clock_us()
        with self._transaction() as connection:
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.VERIFYING:
                raise ValueError("only verifying jobs can complete verification")
            evidence = connection.execute(
                "SELECT payload_json, command_digest, receipt_json, receipt_digest "
                "FROM ingest_outbox WHERE job_id = ?",
                (lease.job_id,),
            ).fetchone()
            if evidence is None or evidence["receipt_json"] != receipt_json:
                raise PermissionError("verification receipt does not match durable evidence")
            if hashlib.sha256(receipt_json.encode()).hexdigest() != evidence["receipt_digest"]:
                raise RuntimeError("durable verification receipt integrity failed")
            command = decode_prepared_ingest(cast(str, evidence["payload_json"]))
            if command.command_digest != evidence["command_digest"]:
                raise RuntimeError("durable verification command integrity failed")
            receipt = decode_ingest_receipt(receipt_json)
            expected_authorization = (
                None
                if command.authorization is None
                else command.authorization.authorization_id
            )
            first = command.chunks[0].chunk
            if (
                receipt.tenant != command.tenant
                or receipt.source_id != first.source_id
                or receipt.source_version != first.source_version
                or receipt.corpus_generation != command.corpus_generation
                or receipt.chunk_ids
                != tuple(item.chunk.chunk_id for item in command.chunks)
                or receipt.idempotency_key != command.idempotency_key
                or receipt.mode is not command.mode
                or receipt.target != command.target
                or receipt.authorization_id != expected_authorization
                or (receipt.published and status is not JobStatus.READY)
                or (not receipt.published and status is not JobStatus.SHADOW_VALIDATED)
            ):
                raise PermissionError("verification receipt does not match its command")
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, updated_at_us = ?, failure = NULL "
                "WHERE job_id = ? AND status = 'verifying'",
                (status.value, now, lease.job_id),
            )
            if cursor.rowcount != 1:
                raise PermissionError("verifying job changed before completion")
            connection.execute(
                "UPDATE leases SET owner_id = NULL, acquired_at_us = NULL, "
                "expires_at_us = NULL WHERE job_id = ? AND owner_id = ? AND fence = ?",
                (lease.job_id, lease.owner_id, lease.fence),
            )
        updated = self.get(self.tenant, lease.job_id)
        assert updated is not None
        return updated

    def claim_finalization(
        self,
        *,
        owner_id: str,
        lease_seconds: float,
        job_id: str | None = None,
    ) -> tuple[AcquisitionJob, JobLease] | None:
        if not _LEASE_OWNER_PATTERN.fullmatch(owner_id) or lease_seconds <= 0:
            raise ValueError("finalization lease owner and duration are required")
        with self._transaction() as connection:
            now = self._clock_us()
            expires = now + int(lease_seconds * 1_000_000)
            parameters: list[object] = [now, now]
            predicate = ""
            if job_id is not None:
                predicate = "AND jobs.job_id = ?"
                parameters.append(job_id)
            row = connection.execute(
                f"""
                SELECT jobs.*, leases.fence
                FROM jobs JOIN leases USING (job_id)
                LEFT JOIN notification_outbox USING (job_id)
                WHERE jobs.status IN ('ready', 'answering', 'notifying')
                  AND (leases.owner_id IS NULL OR leases.expires_at_us <= ?)
                  AND (
                    jobs.status != 'notifying'
                    OR notification_outbox.next_attempt_at_us <= ?
                  )
                  {predicate}
                ORDER BY jobs.updated_at_us, jobs.job_id
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            fence = int(row["fence"]) + 1
            connection.execute(
                "UPDATE leases SET owner_id = ?, fence = ?, acquired_at_us = ?, "
                "expires_at_us = ? WHERE job_id = ?",
                (owner_id, fence, now, expires, row["job_id"]),
            )
            job = _row_job(row, self.tenant)
            if job.status is JobStatus.READY:
                connection.execute(
                    "UPDATE jobs SET status = 'answering', updated_at_us = ? "
                    "WHERE job_id = ? AND status = 'ready'",
                    (now, job.job_id),
                )
                job = replace(job, status=JobStatus.ANSWERING, updated_at=_us_to_iso(now))
        return job, JobLease(self.tenant, cast(str, job.job_id), owner_id, fence, expires)

    def complete_insufficient(self, lease: JobLease, *, failure: str) -> AcquisitionJob:
        self._require_lease(lease)
        if not failure:
            raise ValueError("insufficient finalization failure is required")
        with self._transaction() as connection:
            now = self._clock_us()
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.ANSWERING:
                raise ValueError("only answering jobs can become insufficient")
            connection.execute(
                "UPDATE jobs SET status = 'insufficient_after_ingest', updated_at_us = ?, "
                "failure = ? WHERE job_id = ? AND status = 'answering'",
                (now, failure, lease.job_id),
            )
            self._clear_lease(connection, lease)
        updated = self.get(self.tenant, lease.job_id)
        assert updated is not None
        return updated

    def stage_notification(
        self, lease: JobLease, command: PreparedNotification
    ) -> None:
        self._require_lease(lease)
        if command.tenant != self.tenant or command.job_id != lease.job_id:
            raise PermissionError("notification crossed its tenant or job binding")
        payload = encode_prepared_notification(command)
        with self._transaction() as connection:
            now = self._clock_us()
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.ANSWERING:
                raise ValueError("only answering jobs can stage notification")
            if (
                command.query_digest != job.query_digest
                or command.corpus_generation != job.corpus_generation
            ):
                raise ValueError("notification does not match its durable job")
            existing = connection.execute(
                "SELECT command_digest, payload_json FROM notification_outbox "
                "WHERE job_id = ?",
                (lease.job_id,),
            ).fetchone()
            if existing is not None:
                if existing["command_digest"] != command.command_digest or existing[
                    "payload_json"
                ] != payload:
                    raise FileExistsError("immutable notification outbox differs")
            else:
                connection.execute(
                    "INSERT INTO notification_outbox "
                    "(job_id, notification_id, sink_id, command_digest, payload_json, "
                    "created_at_us, attempts, next_attempt_at_us) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        lease.job_id,
                        command.notification_id,
                        command.sink_id,
                        command.command_digest,
                        payload,
                        now,
                        now,
                    ),
                )
            connection.execute(
                "UPDATE jobs SET status = 'notifying', updated_at_us = ? "
                "WHERE job_id = ? AND status = 'answering'",
                (now, lease.job_id),
            )

    def prepared_notification(
        self, tenant: TenantId, job_id: str
    ) -> PreparedNotification | None:
        self._require_tenant(tenant)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT notification_outbox.*, jobs.query_digest, jobs.corpus_generation "
                "FROM notification_outbox JOIN jobs USING (job_id) WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        command = decode_prepared_notification(cast(str, row["payload_json"]))
        if (
            command.tenant != self.tenant
            or command.command_digest != row["command_digest"]
            or command.job_id != job_id
            or command.notification_id != row["notification_id"]
            or command.sink_id != row["sink_id"]
            or command.query_digest != row["query_digest"]
            or command.corpus_generation != row["corpus_generation"]
        ):
            raise RuntimeError("durable notification outbox integrity failed")
        return command

    def notification_attempts(self, tenant: TenantId, job_id: str) -> int:
        self._require_tenant(tenant)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM notification_outbox WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError("notification outbox is absent")
        return int(row["attempts"])

    def defer_notification(
        self,
        lease: JobLease,
        *,
        error: str,
        delay_seconds: float,
    ) -> None:
        self._require_lease(lease)
        if not error or delay_seconds <= 0:
            raise ValueError("notification retry evidence is invalid")
        with self._transaction() as connection:
            now = self._clock_us()
            next_attempt = now + int(delay_seconds * 1_000_000)
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.NOTIFYING:
                raise ValueError("only notifying jobs can be deferred")
            cursor = connection.execute(
                "UPDATE notification_outbox SET attempts = attempts + 1, "
                "next_attempt_at_us = ?, last_error = ? WHERE job_id = ? "
                "AND receipt_json IS NULL",
                (next_attempt, error[:4096], lease.job_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("notification retry could not be recorded")
            self._clear_lease(connection, lease)

    def complete_notification(
        self, lease: JobLease, receipt_json: str
    ) -> AcquisitionJob:
        from celiums_rezero.knowledge.finalization import decode_notification_receipt

        self._require_lease(lease)
        receipt_digest = hashlib.sha256(receipt_json.encode()).hexdigest()
        with self._transaction() as connection:
            now = self._clock_us()
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.NOTIFYING:
                raise ValueError("only notifying jobs can complete notification")
            outbox = connection.execute(
                "SELECT * FROM notification_outbox WHERE job_id = ?", (lease.job_id,)
            ).fetchone()
            if outbox is None:
                raise RuntimeError("notifying job has no durable outbox")
            command = decode_prepared_notification(cast(str, outbox["payload_json"]))
            receipt = decode_notification_receipt(receipt_json)
            if (
                command.tenant != self.tenant
                or command.job_id != lease.job_id
                or command.query_digest != job.query_digest
                or command.corpus_generation != job.corpus_generation
                or command.notification_id != outbox["notification_id"]
                or command.sink_id != outbox["sink_id"]
                or command.command_digest != outbox["command_digest"]
                or receipt.tenant != command.tenant
                or receipt.job_id != command.job_id
                or receipt.notification_id != command.notification_id
                or receipt.sink_id != command.sink_id
                or receipt.command_digest != command.command_digest
            ):
                raise PermissionError("notification receipt does not match its command")
            if outbox["receipt_json"] is not None and (
                outbox["receipt_json"] != receipt_json
                or outbox["receipt_digest"] != receipt_digest
            ):
                raise FileExistsError("immutable notification receipt differs")
            connection.execute(
                "UPDATE notification_outbox SET receipt_json = ?, receipt_digest = ?, "
                "delivered_at_us = ?, attempts = attempts + 1, last_error = NULL "
                "WHERE job_id = ?",
                (receipt_json, receipt_digest, now, lease.job_id),
            )
            connection.execute(
                "UPDATE jobs SET status = 'completed', updated_at_us = ?, failure = NULL "
                "WHERE job_id = ? AND status = 'notifying'",
                (now, lease.job_id),
            )
            self._clear_lease(connection, lease)
        updated = self.get(self.tenant, lease.job_id)
        assert updated is not None
        return updated

    @staticmethod
    def _clear_lease(connection: sqlite3.Connection, lease: JobLease) -> None:
        cursor = connection.execute(
            "UPDATE leases SET owner_id = NULL, acquired_at_us = NULL, "
            "expires_at_us = NULL WHERE job_id = ? AND owner_id = ? AND fence = ?",
            (lease.job_id, lease.owner_id, lease.fence),
        )
        if cursor.rowcount != 1:
            raise PermissionError("lease changed before durable completion")

    def release(self, lease: JobLease) -> None:
        self._require_lease(lease)
        now = self._clock_us()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE leases SET owner_id = NULL, acquired_at_us = NULL, expires_at_us = NULL "
                "WHERE job_id = ? AND owner_id = ? AND fence = ? AND expires_at_us > ?",
                (lease.job_id, lease.owner_id, lease.fence, now),
            )
            if cursor.rowcount != 1:
                raise PermissionError("job lease is absent, stale, or expired")

    def stage_ingest(self, lease: JobLease, command: PreparedIngest) -> None:
        self._require_lease(lease)
        if command.tenant != self.tenant or command.job_id != lease.job_id:
            raise PermissionError("prepared ingest crossed its tenant or job binding")
        payload = encode_prepared_ingest(command)
        if len(payload.encode()) > _MAX_OUTBOX_BYTES:
            raise ValueError("prepared ingest exceeds its durable byte bound")
        now = self._clock_us()
        with self._transaction() as connection:
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.EMBEDDING:
                raise ValueError("only embedding jobs can stage ingest")
            if (
                command.corpus_generation != job.corpus_generation
                or any(
                    item.embedding_profile != job.embedding_profile
                    or item.chunk.source_id != job.source_id
                    for item in command.chunks
                )
            ):
                raise ValueError("prepared ingest does not match its durable job")
            from celiums_rezero.knowledge.acquisition import validate_embedded_chunks

            validate_embedded_chunks(
                job,
                command.chunks,
                job.embedding_profile,
                command.idempotency_key,
                target=command.target,
            )
            if command.authorization is not None and (
                command.authorization.policy_version != job.policy_version
                or command.authorization.source_id != job.source_id
                or command.authorization.corpus_generation != job.corpus_generation
                or command.authorization.embedding_profile != job.embedding_profile
            ):
                raise ValueError("prepared authorization does not match its durable job")
            existing = connection.execute(
                "SELECT command_digest, payload_json FROM ingest_outbox WHERE job_id = ?",
                (lease.job_id,),
            ).fetchone()
            if existing is not None:
                if existing["command_digest"] != command.command_digest or existing[
                    "payload_json"
                ] != payload:
                    raise FileExistsError("immutable ingest outbox differs")
            else:
                connection.execute(
                    "INSERT INTO ingest_outbox "
                    "(job_id, command_digest, payload_json, created_at_us) "
                    "VALUES (?, ?, ?, ?)",
                    (lease.job_id, command.command_digest, payload, now),
                )
            connection.execute(
                "UPDATE jobs SET status = 'ingesting', updated_at_us = ? "
                "WHERE job_id = ? AND status = 'embedding'",
                (now, lease.job_id),
            )

    def prepared_ingest(self, tenant: TenantId, job_id: str) -> PreparedIngest | None:
        self._require_tenant(tenant)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, command_digest FROM ingest_outbox WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        command = decode_prepared_ingest(cast(str, row["payload_json"]))
        if command.command_digest != row["command_digest"] or command.job_id != job_id:
            raise RuntimeError("durable ingest outbox integrity failed")
        return command

    def record_ingest_receipt(self, lease: JobLease, receipt_json: str) -> None:
        self._require_lease(lease)
        if not receipt_json or len(receipt_json.encode()) > 4_000_000:
            raise ValueError("ingest receipt payload is invalid")
        now = self._clock_us()
        digest = hashlib.sha256(receipt_json.encode()).hexdigest()
        with self._transaction() as connection:
            job = self._leased_job(connection, lease, now)
            if job.status is not JobStatus.INGESTING:
                raise ValueError("only ingesting jobs can record a receipt")
            existing = connection.execute(
                "SELECT receipt_json, receipt_digest FROM ingest_outbox WHERE job_id = ?",
                (lease.job_id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("ingesting job has no durable outbox")
            if existing["receipt_json"] is not None:
                if existing["receipt_json"] != receipt_json or existing["receipt_digest"] != digest:
                    raise FileExistsError("immutable ingest receipt differs")
            else:
                cursor = connection.execute(
                "UPDATE ingest_outbox SET receipt_json = ?, receipt_digest = ? "
                "WHERE job_id = ? AND receipt_json IS NULL",
                (receipt_json, digest, lease.job_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("ingest receipt could not be persisted")
            connection.execute(
                "UPDATE jobs SET status = 'verifying', updated_at_us = ? "
                "WHERE job_id = ? AND status = 'ingesting'",
                (now, lease.job_id),
            )

    def ingest_receipt_json(self, tenant: TenantId, job_id: str) -> str | None:
        self._require_tenant(tenant)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT receipt_json, receipt_digest FROM ingest_outbox WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None or row["receipt_json"] is None:
            return None
        payload = cast(str, row["receipt_json"])
        if hashlib.sha256(payload.encode()).hexdigest() != row["receipt_digest"]:
            raise RuntimeError("durable ingest receipt integrity failed")
        return payload

    def pragmas(self) -> dict[str, object]:
        with self._connect() as connection:
            return {
                "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
                "foreign_keys": connection.execute("PRAGMA foreign_keys").fetchone()[0],
            }

    def _prepare_path(self) -> bool:
        requested = self.path.absolute()
        if any(part in {".", ".."} for part in self.path.parts):
            raise ValueError("SQLite tenant path cannot contain traversal components")
        existing = requested.parent
        missing: list[str] = []
        while not os.path.lexists(existing):
            missing.append(existing.name)
            existing = existing.parent
        if existing.is_symlink() or not existing.is_dir():
            raise ValueError("SQLite tenant path ancestors must be real directories")
        resolved_existing = existing.resolve(strict=True)
        if existing != resolved_existing:
            raise ValueError("SQLite tenant path ancestors cannot contain symlinks")
        current = resolved_existing
        for component in reversed(missing):
            current = current / component
            current.mkdir(mode=0o700)
            _fsync_directory(current.parent)
        resolved_parent = requested.parent.resolve(strict=True)
        if requested.parent != resolved_parent:
            raise ValueError("SQLite tenant directory cannot contain symlinks")
        if resolved_parent.is_symlink() or not resolved_parent.is_dir():
            raise ValueError("SQLite tenant directory must be a real directory")
        parent_metadata = resolved_parent.stat()
        if hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid():
            raise PermissionError("SQLite tenant directory has another owner")
        if stat.S_IMODE(parent_metadata.st_mode) & 0o077:
            raise PermissionError("SQLite tenant directory permissions are too broad")
        self.path = resolved_parent / requested.name
        self._validate_sidecars()
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            descriptor = os.open(
                self.path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            created = False
        else:
            created = True
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("SQLite tenant database must be a regular file")
            if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
                raise PermissionError("SQLite tenant database has another owner")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise PermissionError("SQLite tenant database permissions are too broad")
            self._database_identity = (metadata.st_dev, metadata.st_ino)
            if created:
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            _fsync_directory(resolved_parent)
        return created

    def _validate_sidecars(self) -> None:
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            if os.path.lexists(sidecar):
                metadata = sidecar.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("SQLite sidecar must be a regular file")
                if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
                    raise PermissionError("SQLite sidecar has another owner")
                if stat.S_IMODE(metadata.st_mode) & 0o077:
                    raise PermissionError("SQLite sidecar permissions are too broad")

    def _validate_database_identity(self) -> None:
        metadata = self.path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or (
            metadata.st_dev,
            metadata.st_ino,
        ) != self._database_identity:
            raise PermissionError("SQLite tenant database identity changed")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise PermissionError("SQLite tenant database has another owner")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PermissionError("SQLite tenant database permissions are too broad")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._validate_database_identity()
        self._validate_sidecars()
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            self._validate_database_identity()
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(f"PRAGMA busy_timeout={int(self.timeout_seconds * 1000)}")
            connection.execute("PRAGMA trusted_schema=OFF")
            if (
                not self._initializing
                and not self._migrating
                and connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
            ):
                raise RuntimeError("SQLite WAL durability changed")
            if not self._initializing and not self._migrating:
                objects = {
                    (cast(str, row[0]), cast(str, row[1]))
                    for row in connection.execute(
                        "SELECT type, name FROM sqlite_master "
                        "WHERE name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
                if objects != _EXPECTED_SCHEMA_OBJECTS:
                    raise PermissionError("SQLite tenant schema objects changed")
                forbidden = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type IN ('trigger', 'view') AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if forbidden:
                    raise PermissionError("SQLite tenant schema contains executable extensions")
                schema_rows = connection.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE name IN "
                    "('tenant_meta','jobs','leases','ingest_outbox','notification_outbox',"
                    "'jobs_status_created','notification_due') "
                    "ORDER BY type, name"
                ).fetchall()
                schema_digest = hashlib.sha256(
                    canonical_json([tuple(row) for row in schema_rows]).encode()
                ).hexdigest()
                if schema_digest != _EXPECTED_SCHEMA_DIGEST:
                    raise PermissionError("SQLite tenant schema changed")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def _initialize_new(self, connection: sqlite3.Connection) -> None:
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise RuntimeError("SQLite WAL durability is unavailable")
        connection.executescript(_SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO tenant_meta VALUES (1, ?, ?, ?)",
                (self.tenant.value, _SCHEMA_VERSION, self._clock_us()),
            )
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _verify_existing_database(self) -> int:
        uri = f"{self.path.as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        try:
            tables = {
                cast(str, row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "tenant_meta" not in tables:
                raise PermissionError("SQLite database is not a recognized tenant authority")
            row = connection.execute(
                "SELECT tenant_id, schema_version FROM tenant_meta WHERE singleton = 1"
            ).fetchone()
            if row is None or row["tenant_id"] != self.tenant.value:
                raise PermissionError("SQLite database belongs to another tenant")
            version = int(row["schema_version"])
            if version not in {1, _SCHEMA_VERSION}:
                raise PermissionError("SQLite database schema version is unsupported")
            required = {
                "tenant_meta",
                "jobs",
                "leases",
                "ingest_outbox",
            }
            if version == _SCHEMA_VERSION:
                required.add("notification_outbox")
            if tables != required:
                raise PermissionError("SQLite database is not a recognized tenant authority")
            forbidden = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('trigger', 'view') AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if forbidden:
                raise PermissionError("SQLite tenant schema contains executable extensions")
            expected_columns: dict[str, set[str]] = {
                "tenant_meta": {"singleton", "tenant_id", "schema_version", "created_at_us"},
                "jobs": {
                    "job_id",
                    "identity_digest",
                    "query",
                    "query_digest",
                    "corpus_generation",
                    "policy_version",
                    "embedding_profile",
                    "source_id",
                    "status",
                    "attempts",
                    "created_at_us",
                    "updated_at_us",
                    "failure",
                },
                "leases": {"job_id", "owner_id", "fence", "acquired_at_us", "expires_at_us"},
                "ingest_outbox": {
                    "job_id",
                    "command_digest",
                    "payload_json",
                    "created_at_us",
                    "receipt_json",
                    "receipt_digest",
                },
            }
            if version == _SCHEMA_VERSION:
                expected_columns["notification_outbox"] = {
                    "job_id",
                    "notification_id",
                    "sink_id",
                    "command_digest",
                    "payload_json",
                    "created_at_us",
                    "attempts",
                    "next_attempt_at_us",
                    "last_error",
                    "receipt_json",
                    "receipt_digest",
                    "delivered_at_us",
                }
            for table, columns in expected_columns.items():
                info = connection.execute(f"PRAGMA table_info({table})").fetchall()
                observed = {cast(str, row[1]) for row in info}
                if observed != columns:
                    raise PermissionError("SQLite tenant schema columns are incompatible")
            foreign_keys = connection.execute("PRAGMA foreign_key_list(leases)").fetchall()
            outbox_keys = connection.execute(
                "PRAGMA foreign_key_list(ingest_outbox)"
            ).fetchall()
            notification_keys = (
                connection.execute("PRAGMA foreign_key_list(notification_outbox)").fetchall()
                if version == _SCHEMA_VERSION
                else (object(),)
            )
            if not foreign_keys or not outbox_keys or not notification_keys:
                raise PermissionError("SQLite tenant schema foreign keys are incompatible")
            indexes = {
                cast(str, row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            expected_indexes = {"jobs_status_created"}
            if version == _SCHEMA_VERSION:
                expected_indexes.add("notification_due")
            if indexes != expected_indexes:
                raise PermissionError("SQLite tenant schema indexes are incompatible")
            names = (
                "('tenant_meta','jobs','leases','ingest_outbox','notification_outbox',"
                "'jobs_status_created','notification_due')"
                if version == _SCHEMA_VERSION
                else "('tenant_meta','jobs','leases','ingest_outbox','jobs_status_created')"
            )
            schema_rows = connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                f"WHERE name IN {names} ORDER BY type, name"
            ).fetchall()
            schema_digest = hashlib.sha256(
                canonical_json([tuple(row) for row in schema_rows]).encode()
            ).hexdigest()
            expected_digest = (
                _EXPECTED_SCHEMA_DIGEST
                if version == _SCHEMA_VERSION
                else _EXPECTED_SCHEMA_V1_DIGEST
            )
            if schema_digest != expected_digest:
                raise PermissionError("SQLite tenant schema definition is incompatible")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite quick_check failed")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("SQLite foreign key check failed")
            return version
        finally:
            connection.close()

    def _configure_existing(self, connection: sqlite3.Connection) -> None:
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise RuntimeError("SQLite WAL durability is unavailable")
        if self._migrating:
            try:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute(
                    "SELECT schema_version FROM tenant_meta WHERE singleton = 1"
                ).fetchone()[0]
                if version == 1:
                    objects = {
                        (cast(str, row[0]), cast(str, row[1]))
                        for row in connection.execute(
                            "SELECT type, name FROM sqlite_master "
                            "WHERE name NOT LIKE 'sqlite_%'"
                        ).fetchall()
                    }
                    if objects != _EXPECTED_SCHEMA_V1_OBJECTS:
                        raise RuntimeError("SQLite v1 schema changed before migration")
                    rows = connection.execute(
                        "SELECT type, name, sql FROM sqlite_master "
                        "WHERE name IN "
                        "('tenant_meta','jobs','leases','ingest_outbox','jobs_status_created') "
                        "ORDER BY type, name"
                    ).fetchall()
                    digest = hashlib.sha256(
                        canonical_json([tuple(row) for row in rows]).encode()
                    ).hexdigest()
                    if digest != _EXPECTED_SCHEMA_V1_DIGEST:
                        raise RuntimeError("SQLite v1 schema definition changed before migration")
                    connection.execute(_NOTIFICATION_TABLE_SQL)
                    connection.execute(_NOTIFICATION_INDEX_SQL)
                    connection.execute(
                        "UPDATE tenant_meta SET schema_version = ? "
                        "WHERE singleton = 1 AND schema_version = 1",
                        (_SCHEMA_VERSION,),
                    )
                elif version != _SCHEMA_VERSION:
                    raise RuntimeError("SQLite tenant schema migration version changed")
                if connection.in_transaction:
                    connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            self._migrating = False
            objects = {
                (cast(str, row[0]), cast(str, row[1]))
                for row in connection.execute(
                    "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if objects != _EXPECTED_SCHEMA_OBJECTS:
                raise RuntimeError("SQLite tenant schema migration is incomplete")
            rows = connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name IN "
                "('tenant_meta','jobs','leases','ingest_outbox','notification_outbox',"
                "'jobs_status_created','notification_due') ORDER BY type, name"
            ).fetchall()
            digest = hashlib.sha256(
                canonical_json([tuple(row) for row in rows]).encode()
            ).hexdigest()
            if digest != _EXPECTED_SCHEMA_DIGEST:
                raise RuntimeError("SQLite v2 schema migration digest is invalid")

    @staticmethod
    def _check_integrity(connection: sqlite3.Connection) -> None:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite quick_check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("SQLite foreign key check failed")

    def _leased_job(
        self, connection: sqlite3.Connection, lease: JobLease, now: int
    ) -> AcquisitionJob:
        row = connection.execute(
            """
            SELECT jobs.* FROM jobs JOIN leases USING (job_id)
            WHERE jobs.job_id = ? AND leases.owner_id = ? AND leases.fence = ?
              AND leases.expires_at_us > ?
            """,
            (lease.job_id, lease.owner_id, lease.fence, now),
        ).fetchone()
        if row is None:
            raise PermissionError("job lease is absent, stale, or expired")
        return _row_job(row, self.tenant)

    def _require_tenant(self, tenant: TenantId) -> None:
        if tenant != self.tenant:
            raise PermissionError("SQLite store is bound to another tenant")

    def _require_lease(self, lease: JobLease) -> None:
        self._require_tenant(lease.tenant)


def encode_prepared_ingest(command: PreparedIngest) -> str:
    value = json.loads(canonical_json(command))
    assert isinstance(value, dict)
    chunks = value["chunks"]
    assert isinstance(chunks, list)
    for item in chunks:
        assert isinstance(item, dict)
        item["values"] = [float(number).hex() for number in item["values"]]
    return canonical_json({"schema": "knowledge-prepared-ingest-v1", "value": value})


def encode_prepared_notification(command: PreparedNotification) -> str:
    return canonical_json(
        {"schema": "knowledge-prepared-notification-v1", "value": command}
    )


def decode_prepared_notification(payload: str) -> PreparedNotification:
    parsed = json.loads(payload, object_pairs_hook=_unique_object)
    if canonical_json(parsed) != payload or not isinstance(parsed, dict):
        raise ValueError("prepared notification is not canonical")
    if set(parsed) != {"schema", "value"} or parsed["schema"] != (
        "knowledge-prepared-notification-v1"
    ):
        raise ValueError("prepared notification envelope is invalid")
    value = parsed["value"]
    fields = {
        "tenant",
        "job_id",
        "sink_id",
        "answer",
        "evidence_handles",
        "corpus_generation",
        "query_digest",
        "notification_id",
        "command_digest",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("prepared notification fields are invalid")
    tenant = value["tenant"]
    handles = value["evidence_handles"]
    if (
        not isinstance(tenant, dict)
        or set(tenant) != {"value"}
        or not isinstance(handles, list)
    ):
        raise ValueError("prepared notification bindings are invalid")
    return PreparedNotification(
        tenant=TenantId(_string(tenant["value"])),
        job_id=_string(value["job_id"]),
        sink_id=_string(value["sink_id"]),
        answer=_string(value["answer"]),
        evidence_handles=tuple(_string(handle) for handle in handles),
        corpus_generation=_string(value["corpus_generation"]),
        query_digest=_string(value["query_digest"]),
        notification_id=_string(value["notification_id"]),
        command_digest=_string(value["command_digest"]),
    )


def decode_prepared_ingest(payload: str) -> PreparedIngest:
    parsed = json.loads(payload, object_pairs_hook=_unique_object)
    if canonical_json(parsed) != payload:
        raise ValueError("prepared ingest JSON is not canonical")
    if not isinstance(parsed, dict) or set(parsed) != {"schema", "value"}:
        raise ValueError("prepared ingest envelope is invalid")
    if parsed["schema"] != "knowledge-prepared-ingest-v1" or not isinstance(
        parsed["value"], dict
    ):
        raise ValueError("prepared ingest schema is invalid")
    value = cast(dict[str, object], parsed["value"])
    fields = {
        "tenant",
        "job_id",
        "corpus_generation",
        "idempotency_key",
        "mode",
        "chunks",
        "authorization",
        "target",
        "command_digest",
    }
    if set(value) != fields:
        raise ValueError("prepared ingest fields are invalid")
    tenant = value["tenant"]
    chunks = value["chunks"]
    if not isinstance(tenant, dict) or set(tenant) != {"value"} or not isinstance(chunks, list):
        raise ValueError("prepared ingest tenant or chunks are invalid")
    authorization = value["authorization"]
    target = value["target"]
    return PreparedIngest(
        tenant=TenantId(_string(tenant["value"])),
        job_id=_string(value["job_id"]),
        corpus_generation=_string(value["corpus_generation"]),
        idempotency_key=_string(value["idempotency_key"]),
        mode=IngestMode(_string(value["mode"])),
        chunks=tuple(_embedded_chunk(item) for item in chunks),
        authorization=None if authorization is None else _authorization(authorization),
        target=None if target is None else _target(target),
        command_digest=_string(value["command_digest"]),
    )


def _embedded_chunk(value: object) -> EmbeddedChunk:
    if not isinstance(value, dict) or set(value) != {"chunk", "embedding_profile", "values"}:
        raise ValueError("embedded chunk is invalid")
    chunk = value["chunk"]
    values = value["values"]
    if not isinstance(chunk, dict) or not isinstance(values, list):
        raise ValueError("embedded chunk payload is invalid")
    fields = {
        "chunk_id",
        "source_id",
        "source_version",
        "ordinal",
        "byte_start",
        "byte_end",
        "text",
        "content_digest",
    }
    if set(chunk) != fields or any(isinstance(item, bool) or not isinstance(item, int) for item in (
        chunk["ordinal"], chunk["byte_start"], chunk["byte_end"]
    )):
        raise ValueError("knowledge chunk payload is invalid")
    decoded_values = tuple(float.fromhex(_string(item)) for item in values)
    if any(
        original != decoded.hex() for original, decoded in zip(values, decoded_values, strict=True)
    ):
        raise ValueError("embedding float encoding is not canonical")
    return EmbeddedChunk(
        chunk=KnowledgeChunk(
            chunk_id=_string(chunk["chunk_id"]),
            source_id=_string(chunk["source_id"]),
            source_version=_string(chunk["source_version"]),
            ordinal=cast(int, chunk["ordinal"]),
            byte_start=cast(int, chunk["byte_start"]),
            byte_end=cast(int, chunk["byte_end"]),
            text=_string(chunk["text"]),
            content_digest=_string(chunk["content_digest"]),
        ),
        embedding_profile=_string(value["embedding_profile"]),
        values=decoded_values,
    )


def _authorization(value: object) -> PublicationAuthorization:
    fields = {
        "tenant",
        "source_id",
        "source_version",
        "corpus_generation",
        "policy_version",
        "raw_digest",
        "parsed_digest",
        "parser",
        "parser_version",
        "scans",
        "chunk_ids",
        "chunk_digests",
        "chunk_coordinates",
        "embedding_profile",
        "idempotency_key",
        "authority",
        "target",
        "embedding_digests",
        "authorization_id",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("prepared authorization is invalid")
    tenant = value.get("tenant")
    scans = value.get("scans")
    target = value.get("target")
    list_fields = (
        value.get("chunk_ids"),
        value.get("chunk_digests"),
        value.get("chunk_coordinates"),
        value.get("embedding_digests"),
    )
    if (
        not isinstance(tenant, dict)
        or set(tenant) != {"value"}
        or not isinstance(scans, list)
        or not isinstance(target, dict)
        or any(not isinstance(item, list) for item in list_fields)
    ):
        raise ValueError("prepared authorization bindings are invalid")
    return PublicationAuthorization(
        tenant=TenantId(_string(tenant["value"])),
        source_id=_string(value["source_id"]),
        source_version=_string(value["source_version"]),
        corpus_generation=_string(value["corpus_generation"]),
        policy_version=_string(value["policy_version"]),
        raw_digest=_string(value["raw_digest"]),
        parsed_digest=_string(value["parsed_digest"]),
        parser=_string(value["parser"]),
        parser_version=_string(value["parser_version"]),
        scans=tuple(_scan(item) for item in scans),
        chunk_ids=tuple(_string(item) for item in cast(list[object], value["chunk_ids"])),
        chunk_digests=tuple(
            _string(item) for item in cast(list[object], value["chunk_digests"])
        ),
        chunk_coordinates=tuple(
            _coordinate(item) for item in cast(list[object], value["chunk_coordinates"])
        ),
        embedding_profile=_string(value["embedding_profile"]),
        idempotency_key=_string(value["idempotency_key"]),
        authority=_string(value["authority"]),
        target=_target(target),
        embedding_digests=tuple(
            _string(item) for item in cast(list[object], value["embedding_digests"])
        ),
        authorization_id=_string(value["authorization_id"]),
    )


def _scan(value: object) -> SecurityScanReceipt:
    fields = {"scanner", "version", "target", "content_digest", "findings"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("prepared security scan is invalid")
    findings = value["findings"]
    if isinstance(findings, bool) or not isinstance(findings, int):
        raise ValueError("prepared security scan finding count is invalid")
    return SecurityScanReceipt(
        scanner=_string(value["scanner"]),
        version=_string(value["version"]),
        target=_string(value["target"]),
        content_digest=_string(value["content_digest"]),
        findings=findings,
    )


def _coordinate(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError("prepared chunk coordinate is invalid")
    return cast(tuple[int, int, int], tuple(value))


def _target(value: object) -> PublicationTarget:
    if not isinstance(value, dict) or set(value) != {
        "backend_id",
        "collection",
        "vector_target",
    }:
        raise ValueError("publication target payload is invalid")
    collection = value["collection"]
    if isinstance(collection, bool) or not isinstance(collection, int):
        raise ValueError("publication target collection is invalid")
    return PublicationTarget(
        _string(value["backend_id"]), collection, _string(value["vector_target"])
    )


def _row_job(row: sqlite3.Row, tenant: TenantId) -> AcquisitionJob:
    return AcquisitionJob(
        tenant=tenant,
        query=cast(str, row["query"]),
        query_digest=cast(str, row["query_digest"]),
        corpus_generation=cast(str, row["corpus_generation"]),
        policy_version=cast(str, row["policy_version"]),
        embedding_profile=cast(str, row["embedding_profile"]),
        source_id=cast(str, row["source_id"]),
        status=JobStatus(row["status"]),
        attempts=cast(int, row["attempts"]),
        created_at=_us_to_iso(cast(int, row["created_at_us"])),
        updated_at=_us_to_iso(cast(int, row["updated_at_us"])),
        failure=cast(str | None, row["failure"]),
        job_id=cast(str, row["job_id"]),
    )


def _job_identity_digest(job: AcquisitionJob) -> str:
    return content_hash(
        {
            "schema": "knowledge-job-identity-v1",
            "tenant": job.tenant.value,
            "query_digest": job.query_digest,
            "corpus_generation": job.corpus_generation,
            "policy_version": job.policy_version,
            "embedding_profile": job.embedding_profile,
            "source_id": job.source_id,
        },
        length=64,
    )


def _same_identity(left: AcquisitionJob, right: AcquisitionJob) -> bool:
    fields = (
        "tenant",
        "query_digest",
        "corpus_generation",
        "policy_version",
        "embedding_profile",
        "source_id",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _validate_transition(current: JobStatus, status: JobStatus) -> None:
    if current.terminal:
        raise ValueError("terminal jobs cannot transition")
    if status not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid job transition: {current} -> {status}")


def _acquisition_placeholders() -> str:
    return ",".join("?" for _ in _ACQUISITION_ACTIVE)


def _now_us() -> int:
    return time.time_ns() // 1000


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _iso_to_us(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("job timestamp must include a timezone")
    return int(parsed.timestamp() * 1_000_000)


def _us_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000, UTC).isoformat()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("prepared ingest contains duplicate JSON keys")
        result[key] = value
    return result


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("prepared ingest field must be a string")
    return value


_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {JobStatus.ACQUIRING, JobStatus.POLICY_DENIED, JobStatus.CANCELLED}
    ),
    JobStatus.ACQUIRING: frozenset(
        {
            JobStatus.QUARANTINED,
            JobStatus.POLICY_DENIED,
            JobStatus.FAILED,
            JobStatus.SECURITY_REJECTED,
        }
    ),
    JobStatus.QUARANTINED: frozenset(
        {JobStatus.VALIDATING, JobStatus.SECURITY_REJECTED}
    ),
    JobStatus.VALIDATING: frozenset(
        {
            JobStatus.CHUNKING,
            JobStatus.LICENSE_UNKNOWN,
            JobStatus.SECURITY_REJECTED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.CHUNKING: frozenset({JobStatus.EMBEDDING, JobStatus.FAILED}),
    JobStatus.EMBEDDING: frozenset({JobStatus.INGESTING, JobStatus.FAILED}),
    JobStatus.INGESTING: frozenset({JobStatus.VERIFYING, JobStatus.FAILED}),
    JobStatus.VERIFYING: frozenset(
        {
            JobStatus.READY,
            JobStatus.SHADOW_VALIDATED,
            JobStatus.INSUFFICIENT_AFTER_INGEST,
            JobStatus.FAILED,
        }
    ),
    JobStatus.READY: frozenset({JobStatus.ANSWERING, JobStatus.FAILED}),
    JobStatus.ANSWERING: frozenset(
        {JobStatus.NOTIFYING, JobStatus.INSUFFICIENT_AFTER_INGEST, JobStatus.FAILED}
    ),
    JobStatus.NOTIFYING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED}),
    **{status: frozenset() for status in JobStatus if status.terminal},
}

_ACQUISITION_ACTIVE = (
    JobStatus.QUEUED,
    JobStatus.ACQUIRING,
    JobStatus.QUARANTINED,
    JobStatus.VALIDATING,
    JobStatus.CHUNKING,
    JobStatus.EMBEDDING,
    JobStatus.INGESTING,
    JobStatus.VERIFYING,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenant_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    tenant_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at_us INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    identity_digest TEXT NOT NULL UNIQUE,
    query TEXT NOT NULL,
    query_digest TEXT NOT NULL,
    corpus_generation TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    embedding_profile TEXT NOT NULL,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    failure TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at_us, job_id);
CREATE TABLE IF NOT EXISTS leases (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    owner_id TEXT,
    fence INTEGER NOT NULL CHECK (fence >= 0),
    acquired_at_us INTEGER,
    expires_at_us INTEGER,
    CHECK ((owner_id IS NULL AND acquired_at_us IS NULL AND expires_at_us IS NULL)
        OR (owner_id IS NOT NULL AND acquired_at_us IS NOT NULL
            AND expires_at_us IS NOT NULL AND expires_at_us > acquired_at_us))
);
CREATE TABLE IF NOT EXISTS ingest_outbox (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    command_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    receipt_json TEXT,
    receipt_digest TEXT,
    CHECK ((receipt_json IS NULL AND receipt_digest IS NULL)
        OR (receipt_json IS NOT NULL AND receipt_digest IS NOT NULL))
);
"""

_NOTIFICATION_TABLE_SQL = """
CREATE TABLE notification_outbox (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    notification_id TEXT NOT NULL UNIQUE,
    sink_id TEXT NOT NULL,
    command_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_us INTEGER NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    next_attempt_at_us INTEGER NOT NULL,
    last_error TEXT,
    receipt_json TEXT,
    receipt_digest TEXT,
    delivered_at_us INTEGER,
    CHECK ((receipt_json IS NULL AND receipt_digest IS NULL AND delivered_at_us IS NULL)
        OR (receipt_json IS NOT NULL AND receipt_digest IS NOT NULL
            AND delivered_at_us IS NOT NULL))
)
"""

_NOTIFICATION_INDEX_SQL = """
CREATE INDEX notification_due ON notification_outbox(next_attempt_at_us, job_id)
"""

_NOTIFICATION_SCHEMA = _NOTIFICATION_TABLE_SQL + ";" + _NOTIFICATION_INDEX_SQL + ";"

_SCHEMA_V1 = _SCHEMA
_SCHEMA += _NOTIFICATION_SCHEMA

def _expected_schema_digest() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_SCHEMA)
        rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name IN ('tenant_meta','jobs','leases','ingest_outbox',"
            "'notification_outbox','jobs_status_created','notification_due') "
            "ORDER BY type, name"
        ).fetchall()
        return hashlib.sha256(canonical_json([tuple(row) for row in rows]).encode()).hexdigest()
    finally:
        connection.close()


def _schema_digest(schema: str, names: str) -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema)
        rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            f"WHERE name IN {names} ORDER BY type, name"
        ).fetchall()
        return hashlib.sha256(canonical_json([tuple(row) for row in rows]).encode()).hexdigest()
    finally:
        connection.close()


_EXPECTED_SCHEMA_DIGEST = _expected_schema_digest()
_EXPECTED_SCHEMA_V1_DIGEST = _schema_digest(
    _SCHEMA_V1,
    "('tenant_meta','jobs','leases','ingest_outbox','jobs_status_created')",
)
_EXPECTED_SCHEMA_OBJECTS = {
    ("table", "tenant_meta"),
    ("table", "jobs"),
    ("table", "leases"),
    ("table", "ingest_outbox"),
    ("table", "notification_outbox"),
    ("index", "jobs_status_created"),
    ("index", "notification_due"),
}
_EXPECTED_SCHEMA_V1_OBJECTS = {
    ("table", "tenant_meta"),
    ("table", "jobs"),
    ("table", "leases"),
    ("table", "ingest_outbox"),
    ("index", "jobs_status_created"),
}
