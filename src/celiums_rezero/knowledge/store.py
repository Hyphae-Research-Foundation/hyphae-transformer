"""Tenant-isolated in-memory authority used by Phase 0 tests and simulators."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from celiums_rezero.knowledge.schemas import AcquisitionJob, JobStatus, TenantId


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
        if job.status.terminal:
            raise ValueError("terminal jobs cannot transition")
        allowed = _ALLOWED_TRANSITIONS[job.status]
        if status not in allowed:
            raise ValueError(f"invalid job transition: {job.status} -> {status}")
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
        return sum(not job.status.terminal for job in self._jobs.get(tenant.value, {}).values())


_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {JobStatus.ACQUIRING, JobStatus.POLICY_DENIED, JobStatus.CANCELLED}
    ),
    JobStatus.ACQUIRING: frozenset(
        {JobStatus.QUARANTINED, JobStatus.FAILED, JobStatus.SECURITY_REJECTED}
    ),
    JobStatus.QUARANTINED: frozenset(
        {JobStatus.VALIDATING, JobStatus.SECURITY_REJECTED}
    ),
    JobStatus.VALIDATING: frozenset(
        {JobStatus.CHUNKING, JobStatus.LICENSE_UNKNOWN, JobStatus.SECURITY_REJECTED}
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
