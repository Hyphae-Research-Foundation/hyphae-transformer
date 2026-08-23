"""Durable finalization policy, sink conformance, metrics, and scheduling."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from celiums_rezero.knowledge.acquisition import DurableAcquisitionWorker
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    DeadLetterReason,
    FinalizationPhase,
    FinalizationQueueSnapshot,
    JobLease,
    JobStatus,
    NotificationReceipt,
    PreparedNotification,
    TenantId,
)
from celiums_rezero.knowledge.store import SQLiteTenantStore
from celiums_rezero.lab.serialization import canonical_json


class TransientFinalizationError(RuntimeError):
    """Callback failed in a way that policy may retry."""


class PermanentFinalizationError(RuntimeError):
    """Callback or configuration cannot succeed without intervention."""


class FinalizationTimeout(TransientFinalizationError):
    """Callback honored its deadline and timed out."""


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    answer: str
    evidence_handles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.answer.strip() or len(self.answer.encode()) > 1_000_000:
            raise ValueError("final answer is empty or exceeds its byte bound")


class FinalAnswerer(Protocol):
    def answer(
        self, job: AcquisitionJob, *, timeout_seconds: float
    ) -> FinalAnswer | None: ...


class NotificationSink(Protocol):
    @property
    def sink_id(self) -> str: ...

    def deliver(
        self, command: PreparedNotification, *, timeout_seconds: float
    ) -> NotificationReceipt: ...


@dataclass(frozen=True, slots=True)
class FinalizationPolicy:
    answer_timeout_seconds: float = 60.0
    notification_timeout_seconds: float = 30.0
    lease_safety_seconds: float = 5.0
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 300.0
    max_answer_failures: int = 5
    max_notification_failures: int = 8

    def __post_init__(self) -> None:
        seconds = (
            self.answer_timeout_seconds,
            self.notification_timeout_seconds,
            self.lease_safety_seconds,
            self.retry_base_seconds,
            self.retry_max_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isfinite(value)
            or value < 0.000001
            or value > 9_000_000_000_000
            or value * 1_000_000 != int(value * 1_000_000)
            for value in seconds
        ):
            raise ValueError("finalization policy durations must be positive")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("finalization retry maximum is below its base")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.max_answer_failures, self.max_notification_failures)
        ):
            raise ValueError("finalization failure limits must be positive")


class DurableFinalizationWorker:
    def __init__(
        self,
        *,
        store: SQLiteTenantStore,
        worker_id: str,
        lease_seconds: float,
        answerer: FinalAnswerer,
        sink: NotificationSink,
        policy: FinalizationPolicy | None = None,
    ) -> None:
        self.policy = FinalizationPolicy() if policy is None else policy
        required_lease = max(
            self.policy.answer_timeout_seconds,
            self.policy.notification_timeout_seconds,
        ) + self.policy.lease_safety_seconds
        if (
            not worker_id
            or isinstance(lease_seconds, bool)
            or not isfinite(lease_seconds)
            or lease_seconds > 9_000_000_000_000
            or lease_seconds * 1_000_000 != int(lease_seconds * 1_000_000)
            or lease_seconds < required_lease
        ):
            raise ValueError("finalization lease does not cover callback deadline and safety")
        self.store = store
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.answerer = answerer
        self.sink = sink
        self.sink_id = sink.sink_id
        if not self.sink_id:
            raise ValueError("notification sink ID is required")
        self.last_error: str | None = None

    def run_next(self, *, job_id: str | None = None) -> AcquisitionJob | None:
        claimed = self.store.claim_finalization(
            owner_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            job_id=job_id,
        )
        if claimed is None:
            return None
        job, lease = claimed
        try:
            if job.status is JobStatus.ANSWERING:
                return self._answer(job, lease)
            if job.status is JobStatus.NOTIFYING:
                return self._notify(job, lease)
            raise RuntimeError(f"finalization worker cannot resume {job.status}")
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            latest = self.store.get(job.tenant, job.job_id or "")
            assert latest is not None
            return latest

    def _answer(self, job: AcquisitionJob, lease: JobLease) -> AcquisitionJob:
        lease = self.store.renew(lease, lease_seconds=self.lease_seconds)
        try:
            result = self.answerer.answer(
                job, timeout_seconds=self.policy.answer_timeout_seconds
            )
            if result is not None and not isinstance(result, FinalAnswer):
                raise PermanentFinalizationError("answerer returned an invalid result")
            if result is not None:
                command = PreparedNotification(
                    tenant=job.tenant,
                    job_id=job.job_id or "",
                    sink_id=self.sink_id,
                    answer=result.answer,
                    evidence_handles=result.evidence_handles,
                    corpus_generation=job.corpus_generation,
                    query_digest=job.query_digest,
                )
        except Exception as error:
            return self._callback_failure(
                job, lease, FinalizationPhase.ANSWERING, error
            )
        if result is None:
            return self.store.complete_insufficient(
                lease, failure="newly ingested evidence remains insufficient"
            )
        assert result is not None
        lease = self.store.renew(lease, lease_seconds=self.lease_seconds)
        self.store.stage_notification(lease, command)
        return self._notify(job, lease)

    def _notify(self, job: AcquisitionJob, lease: JobLease) -> AcquisitionJob:
        command = self.store.prepared_notification(job.tenant, job.job_id or "")
        if command is None:
            raise RuntimeError("notifying job has no durable notification command")
        if command.sink_id != self.sink_id:
            error = PermanentFinalizationError("notification sink configuration drifted")
            return self._callback_failure(
                job, lease, FinalizationPhase.NOTIFYING, error, attempted=False
            )
        lease = self.store.renew(lease, lease_seconds=self.lease_seconds)
        try:
            receipt = self.sink.deliver(
                command, timeout_seconds=self.policy.notification_timeout_seconds
            )
            _validate_sink_receipt(command, receipt)
            receipt = NotificationReceipt(
                tenant=receipt.tenant,
                job_id=receipt.job_id,
                notification_id=receipt.notification_id,
                sink_id=receipt.sink_id,
                command_digest=receipt.command_digest,
                provider_receipt=receipt.provider_receipt,
            )
        except Exception as error:
            return self._callback_failure(
                job, lease, FinalizationPhase.NOTIFYING, error
            )
        return self.store.complete_notification(
            lease, encode_notification_receipt(receipt)
        )

    def _callback_failure(
        self,
        job: AcquisitionJob,
        lease: JobLease,
        phase: FinalizationPhase,
        error: Exception,
        *,
        attempted: bool = True,
    ) -> AcquisitionJob:
        message = f"{type(error).__name__}: {error}"
        if phase is FinalizationPhase.ANSWERING:
            failures = self.store.answer_failures(job.tenant, job.job_id or "") + 1
            maximum = self.policy.max_answer_failures
        else:
            failures = self.store.notification_attempts(job.tenant, job.job_id or "") + int(
                attempted
            )
            failures = max(failures, 1)
            maximum = self.policy.max_notification_failures
        if isinstance(error, (FinalizationTimeout, TimeoutError)):
            error = FinalizationTimeout(str(error))
        if isinstance(error, PermanentFinalizationError):
            return self.store.dead_letter_finalization(
                lease,
                reason=DeadLetterReason.PERMANENT,
                error=message,
                failures=failures,
                attempted=attempted,
            )
        if not isinstance(error, TransientFinalizationError) or failures >= maximum:
            reason = (
                DeadLetterReason.RETRIES_EXHAUSTED
                if isinstance(error, TransientFinalizationError)
                else DeadLetterReason.PERMANENT
            )
            return self.store.dead_letter_finalization(
                lease,
                reason=reason,
                error=message,
                failures=failures,
                attempted=attempted,
            )
        delay_us = retry_delay_us(
            job_id=job.job_id or "",
            phase=phase,
            failure=failures,
            base_us=int(self.policy.retry_base_seconds * 1_000_000),
            maximum_us=int(self.policy.retry_max_seconds * 1_000_000),
        )
        if phase is FinalizationPhase.ANSWERING:
            return self.store.defer_answering(
                lease, error=message, delay_us=delay_us
            )
        self.store.defer_notification(lease, error=message, delay_us=delay_us)
        latest = self.store.get(job.tenant, job.job_id or "")
        assert latest is not None
        return latest


class KnowledgeScheduler:
    """One-tenant cooperative scheduler; database fences remain authoritative."""

    def __init__(
        self,
        *,
        store: SQLiteTenantStore,
        acquisition: DurableAcquisitionWorker,
        finalization: DurableFinalizationWorker,
        poll_seconds: float = 1.0,
        batch_size: int = 16,
        sleep: Callable[[float], None] = time.sleep,
        error_history: int = 100,
    ) -> None:
        if poll_seconds <= 0 or batch_size < 1 or error_history < 1:
            raise ValueError("scheduler polling bounds are invalid")
        acquisition_store = acquisition.coordinator.store
        if (
            not isinstance(acquisition_store, SQLiteTenantStore)
            or acquisition_store.tenant != store.tenant
            or acquisition_store.path != store.path
            or finalization.store.tenant != store.tenant
            or finalization.store.path != store.path
        ):
            raise ValueError("scheduler workers must share one tenant database")
        self.store = store
        self.acquisition = acquisition
        self.finalization = finalization
        self.poll_seconds = poll_seconds
        self.batch_size = batch_size
        self.sleep = sleep
        self.last_errors: deque[str] = deque(maxlen=error_history)

    def tick(self) -> int:
        processed = self.store.recover_expired(limit=self.batch_size)
        for _ in range(self.batch_size):
            try:
                acquisition = self.acquisition.run_next()
            except Exception as error:
                self.last_errors.append(f"acquisition {type(error).__name__}: {error}")
                acquisition = None
            try:
                finalization = self.finalization.run_next()
                if self.finalization.last_error is not None:
                    self.last_errors.append(self.finalization.last_error)
                    self.finalization.last_error = None
            except Exception as error:
                self.last_errors.append(f"finalization {type(error).__name__}: {error}")
                finalization = None
            processed += int(acquisition is not None) + int(finalization is not None)
            if acquisition is None and finalization is None:
                break
        return processed

    def snapshot(self) -> FinalizationQueueSnapshot:
        return self.store.finalization_queue_snapshot(self.store.tenant)

    def run(self, *, should_stop: Callable[[], bool]) -> None:
        while not should_stop():
            if self.tick() == 0:
                self.sleep(self.poll_seconds)


def retry_delay_us(
    *,
    job_id: str,
    phase: FinalizationPhase,
    failure: int,
    base_us: int,
    maximum_us: int,
) -> int:
    if failure < 1 or base_us < 1 or maximum_us < base_us:
        raise ValueError("retry jitter inputs are invalid")
    cap = min(base_us * (1 << min(failure - 1, 30)), maximum_us)
    floor = max((cap + 1) // 2, 1)
    digest = hashlib.sha256(
        canonical_json(
            {
                "schema": "knowledge-finalization-jitter-v1",
                "job_id": job_id,
                "phase": phase,
                "failure": failure,
            }
        ).encode()
    ).digest()
    word = int.from_bytes(digest[:8], "big")
    return floor + ((word * (cap - floor + 1)) >> 64)


def encode_notification_receipt(receipt: NotificationReceipt) -> str:
    return canonical_json({"schema": "knowledge-notification-receipt-v1", "value": receipt})


def decode_notification_receipt(payload: str) -> NotificationReceipt:
    parsed = json.loads(payload, object_pairs_hook=_unique_object)
    if canonical_json(parsed) != payload or not isinstance(parsed, dict):
        raise ValueError("notification receipt is not canonical")
    if set(parsed) != {"schema", "value"} or parsed["schema"] != (
        "knowledge-notification-receipt-v1"
    ):
        raise ValueError("notification receipt envelope is invalid")
    value = parsed["value"]
    fields = {
        "tenant",
        "job_id",
        "notification_id",
        "sink_id",
        "command_digest",
        "provider_receipt",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("notification receipt fields are invalid")
    tenant = value["tenant"]
    if not isinstance(tenant, dict) or set(tenant) != {"value"}:
        raise ValueError("notification receipt tenant is invalid")
    return NotificationReceipt(
        tenant=TenantId(_string(tenant["value"])),
        job_id=_string(value["job_id"]),
        notification_id=_string(value["notification_id"]),
        sink_id=_string(value["sink_id"]),
        command_digest=_string(value["command_digest"]),
        provider_receipt=_string(value["provider_receipt"]),
    )


def _validate_sink_receipt(
    command: PreparedNotification, receipt: object
) -> None:
    if not isinstance(receipt, NotificationReceipt):
        raise PermanentFinalizationError("notification sink returned an untyped receipt")
    if (
        receipt.tenant != command.tenant
        or receipt.job_id != command.job_id
        or receipt.notification_id != command.notification_id
        or receipt.sink_id != command.sink_id
        or receipt.command_digest != command.command_digest
    ):
        raise PermanentFinalizationError(
            "notification sink receipt does not match its command"
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("notification receipt contains duplicate JSON keys")
        result[key] = value
    return result


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("notification receipt field must be a string")
    return value
