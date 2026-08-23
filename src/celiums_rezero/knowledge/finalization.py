"""Durable answer finalization, idempotent notification delivery, and scheduling."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from celiums_rezero.knowledge.acquisition import DurableAcquisitionWorker
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    JobLease,
    JobStatus,
    NotificationReceipt,
    PreparedNotification,
    TenantId,
)
from celiums_rezero.knowledge.store import SQLiteTenantStore
from celiums_rezero.lab.serialization import canonical_json


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    answer: str
    evidence_handles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.answer.strip() or len(self.answer.encode()) > 1_000_000:
            raise ValueError("final answer is empty or exceeds its byte bound")


class FinalAnswerer(Protocol):
    def answer(self, job: AcquisitionJob) -> FinalAnswer | None: ...


class NotificationSink(Protocol):
    @property
    def sink_id(self) -> str: ...

    def deliver(self, command: PreparedNotification) -> NotificationReceipt: ...


class DurableFinalizationWorker:
    def __init__(
        self,
        *,
        store: SQLiteTenantStore,
        worker_id: str,
        lease_seconds: float,
        answerer: FinalAnswerer,
        sink: NotificationSink,
        retry_base_seconds: float = 5.0,
        retry_max_seconds: float = 300.0,
    ) -> None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("finalization worker ID and lease duration are required")
        if retry_base_seconds <= 0 or retry_max_seconds < retry_base_seconds:
            raise ValueError("notification retry bounds are invalid")
        self.store = store
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.answerer = answerer
        self.sink = sink
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
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
        result = self.answerer.answer(job)
        if result is None:
            return self.store.complete_insufficient(
                lease, failure="newly ingested evidence remains insufficient"
            )
        command = PreparedNotification(
            tenant=job.tenant,
            job_id=job.job_id or "",
            sink_id=self.sink.sink_id,
            answer=result.answer,
            evidence_handles=result.evidence_handles,
            corpus_generation=job.corpus_generation,
            query_digest=job.query_digest,
        )
        lease = self.store.renew(lease, lease_seconds=self.lease_seconds)
        self.store.stage_notification(lease, command)
        return self._notify(job, lease)

    def _notify(self, job: AcquisitionJob, lease: JobLease) -> AcquisitionJob:
        command = self.store.prepared_notification(job.tenant, job.job_id or "")
        if command is None:
            raise RuntimeError("notifying job has no durable notification command")
        if command.sink_id != self.sink.sink_id:
            raise PermissionError("notification sink configuration drifted")
        lease = self.store.renew(lease, lease_seconds=self.lease_seconds)
        try:
            receipt = self.sink.deliver(command)
        except Exception as error:
            attempts = self.store.notification_attempts(job.tenant, job.job_id or "") + 1
            delay = min(
                self.retry_base_seconds * (2 ** min(attempts - 1, 30)),
                self.retry_max_seconds,
            )
            self.store.defer_notification(
                lease,
                error=f"{type(error).__name__}: {error}",
                delay_seconds=delay,
            )
            latest = self.store.get(job.tenant, job.job_id or "")
            assert latest is not None
            return latest
        return self.store.complete_notification(
            lease, encode_notification_receipt(receipt)
        )


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
    ) -> None:
        if poll_seconds <= 0 or batch_size < 1:
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
        self.last_errors: list[str] = []

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

    def run(self, *, should_stop: Callable[[], bool]) -> None:
        while not should_stop():
            if self.tick() == 0:
                self.sleep(self.poll_seconds)


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
