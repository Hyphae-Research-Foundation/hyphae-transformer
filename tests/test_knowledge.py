from __future__ import annotations

import hashlib

import pytest

from celiums_rezero.knowledge import (
    AcquisitionPolicy,
    EvidenceBundle,
    EvidenceHit,
    InMemoryTenantStore,
    JobStatus,
    KnowledgeCoordinator,
    SufficiencyDecision,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.schemas import NO_KNOWLEDGE_MESSAGE, SourcePolicy


def source_policy() -> SourcePolicy:
    return SourcePolicy(
        source_id="official_docs",
        allowed_hosts=("docs.example.com",),
        allowed_path_prefixes=("/manual/",),
        allowed_license_ids=("Apache-2.0", "CC-BY-4.0"),
    )


def coordinator(*, max_active_jobs: int = 4) -> KnowledgeCoordinator:
    return KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(minimum_score=0.7, minimum_margin=0.05),
        acquisition=AcquisitionPolicy(
            version="policy-v1",
            sources=(source_policy(),),
            max_active_jobs=max_active_jobs,
        ),
        store=InMemoryTenantStore(),
        embedding_profile="bge-m3@sha256:fixture",
    )


def bundle(
    tenant: TenantId,
    query: str,
    scores: tuple[float, ...],
    *,
    conflicting: bool = False,
    blocked: bool = False,
) -> EvidenceBundle:
    normalized = normalize_query(query)
    hits = tuple(
        EvidenceHit(
            handle=f"passage_{index:016x}",
            source_id="official_docs",
            source_version="v1",
            text=f"Evidence {index}",
            score=score,
            content_digest=hashlib.sha256(f"Evidence {index}".encode()).hexdigest(),
        )
        for index, score in enumerate(scores)
    )
    return EvidenceBundle(
        tenant=tenant,
        query_digest=hashlib.sha256(normalized.encode()).hexdigest(),
        corpus_generation="generation-v1",
        hits=hits,
        conflicting=conflicting,
        blocked=blocked,
    )


def test_supported_evidence_does_not_create_a_job() -> None:
    item = coordinator()
    tenant = TenantId("tenant_a")
    response = item.answer_or_enqueue(
        tenant=tenant,
        query="How does the API work?",
        evidence=bundle(tenant, "How does the API work?", (0.91, 0.72)),
        source_id="official_docs",
    )
    assert response.status == "evidence_ready"
    assert response.decision is SufficiencyDecision.SUPPORTED
    assert response.job_id is None
    assert response.evidence_handles


def test_absent_evidence_enqueues_one_deduplicated_async_job() -> None:
    item = coordinator()
    tenant = TenantId("tenant_a")
    evidence = bundle(tenant, "Unknown feature", ())
    first = item.answer_or_enqueue(
        tenant=tenant,
        query="Unknown feature",
        evidence=evidence,
        source_id="official_docs",
    )
    second = item.answer_or_enqueue(
        tenant=tenant,
        query="  UNKNOWN   feature ",
        evidence=evidence,
        source_id="official_docs",
    )
    assert first.answer == NO_KNOWLEDGE_MESSAGE
    assert first.status == "knowledge_pending"
    assert first.job_id == second.job_id
    assert not first.deduplicated and second.deduplicated


def test_evidence_and_jobs_are_tenant_bound() -> None:
    item = coordinator()
    tenant_a = TenantId("tenant_a")
    tenant_b = TenantId("tenant_b")
    response = item.answer_or_enqueue(
        tenant=tenant_a,
        query="Unknown feature",
        evidence=bundle(tenant_a, "Unknown feature", ()),
        source_id="official_docs",
    )
    assert response.job_id is not None
    assert item.job_status(tenant_b, response.job_id) is None
    with pytest.raises(ValueError, match="tenant and query"):
        item.answer_or_enqueue(
            tenant=tenant_b,
            query="Unknown feature",
            evidence=bundle(tenant_a, "Unknown feature", ()),
            source_id="official_docs",
        )


def test_conflicting_or_blocked_evidence_never_auto_ingests() -> None:
    item = coordinator()
    tenant = TenantId("tenant_a")
    for values in (
        bundle(tenant, "Question", (0.9,), conflicting=True),
        bundle(tenant, "Question", (), blocked=True),
    ):
        response = item.answer_or_enqueue(
            tenant=tenant,
            query="Question",
            evidence=values,
            source_id="official_docs",
        )
        assert response.job_id is None
        assert response.status in {"conflict", "blocked"}


def test_unknown_source_and_job_quota_fail_closed() -> None:
    tenant = TenantId("tenant_a")
    item = coordinator(max_active_jobs=1)
    missing = bundle(tenant, "Question one", ())
    denied = item.answer_or_enqueue(
        tenant=tenant,
        query="Question one",
        evidence=missing,
        source_id="unknown",
    )
    assert denied.status == "source_denied"
    first = item.answer_or_enqueue(
        tenant=tenant,
        query="Question one",
        evidence=missing,
        source_id="official_docs",
    )
    assert first.job_id is not None
    exhausted = item.answer_or_enqueue(
        tenant=tenant,
        query="Question two",
        evidence=bundle(tenant, "Question two", ()),
        source_id="official_docs",
    )
    assert exhausted.status == "quota_exceeded"


def test_job_state_machine_rejects_skips_and_terminal_mutation() -> None:
    item = coordinator()
    tenant = TenantId("tenant_a")
    pending = item.answer_or_enqueue(
        tenant=tenant,
        query="Question",
        evidence=bundle(tenant, "Question", ()),
        source_id="official_docs",
    )
    assert pending.job_id is not None
    with pytest.raises(ValueError, match="invalid job transition"):
        item.store.transition(tenant, pending.job_id, JobStatus.READY)
    cancelled = item.cancel(tenant, pending.job_id)
    assert cancelled.status is JobStatus.CANCELLED
    with pytest.raises(ValueError, match="terminal"):
        item.store.transition(tenant, pending.job_id, JobStatus.ACQUIRING)


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.example.com/manual/page",
        "https://docs.example.com/other/page",
        "https://user@docs.example.com/manual/page",
        "https://127.0.0.1/manual/page",
        "file:///etc/passwd",
    ],
)
def test_source_policy_rejects_unsafe_urls(url: str) -> None:
    assert not source_policy().permits_url(url)


def test_source_policy_accepts_only_declared_https_path() -> None:
    assert source_policy().permits_url("https://docs.example.com/manual/page")
    with pytest.raises(ValueError, match="public normalized DNS"):
        SourcePolicy(source_id="bad", allowed_hosts=("127.0.0.1",))
