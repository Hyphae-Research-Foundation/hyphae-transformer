"""Deterministic answerability and asynchronous acquisition coordinator."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from celiums_rezero.knowledge.schemas import (
    NO_KNOWLEDGE_MESSAGE,
    AcquisitionJob,
    AcquisitionPolicy,
    EvidenceBundle,
    JobStatus,
    KnowledgeResponse,
    SufficiencyDecision,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.store import InMemoryTenantStore


class KnowledgeRetriever(Protocol):
    def retrieve(self, tenant: TenantId, query: str) -> EvidenceBundle: ...


class KnowledgeCoordinator:
    def __init__(
        self,
        *,
        sufficiency: SufficiencyPolicy,
        acquisition: AcquisitionPolicy,
        store: InMemoryTenantStore,
        embedding_profile: str,
    ) -> None:
        if not embedding_profile:
            raise ValueError("embedding profile is required")
        self.sufficiency = sufficiency
        self.acquisition = acquisition
        self.store = store
        self.embedding_profile = embedding_profile

    def answer_or_enqueue(
        self,
        *,
        tenant: TenantId,
        query: str,
        evidence: EvidenceBundle,
        source_id: str,
    ) -> KnowledgeResponse:
        normalized = normalize_query(query)
        query_digest = hashlib.sha256(normalized.encode()).hexdigest()
        if evidence.tenant != tenant or evidence.query_digest != query_digest:
            raise ValueError("evidence is not bound to this tenant and query")
        decision = self.sufficiency.decide(evidence)
        if decision is SufficiencyDecision.SUPPORTED:
            handles = tuple(hit.handle for hit in evidence.hits if hit.active and hit.trusted)
            return KnowledgeResponse(
                status="evidence_ready",
                answer="",
                decision=decision,
                evidence_handles=handles,
            )
        if decision in {SufficiencyDecision.BLOCKED, SufficiencyDecision.CONFLICT}:
            return KnowledgeResponse(
                status=decision.value,
                answer="No puedo adquirir ni responder con seguridad bajo la politica actual.",
                decision=decision,
            )
        if not self.acquisition.automatic:
            return KnowledgeResponse(
                status="approval_required",
                answer="No poseo suficiente conocimiento; se requiere aprobacion para adquirirlo.",
                decision=decision,
            )
        if self.acquisition.source(source_id) is None:
            return KnowledgeResponse(
                status="source_denied",
                answer=(
                    "No poseo este conocimiento y no existe una fuente permitida "
                    "para adquirirlo."
                ),
                decision=decision,
            )
        if self.store.active_count(tenant) >= self.acquisition.max_active_jobs:
            return KnowledgeResponse(
                status="quota_exceeded",
                answer="No poseo este conocimiento y la cuota de adquisicion esta agotada.",
                decision=decision,
            )
        job = AcquisitionJob(
            tenant=tenant,
            query=normalized,
            query_digest=query_digest,
            corpus_generation=evidence.corpus_generation,
            policy_version=self.acquisition.version,
            embedding_profile=self.embedding_profile,
            source_id=source_id,
        )
        queued, deduplicated = self.store.enqueue(job)
        return KnowledgeResponse(
            status="knowledge_pending",
            answer=NO_KNOWLEDGE_MESSAGE,
            decision=SufficiencyDecision.PENDING,
            job_id=queued.job_id,
            deduplicated=deduplicated,
        )

    def retrieve_or_enqueue(
        self,
        *,
        tenant: TenantId,
        query: str,
        retriever: KnowledgeRetriever,
        source_id: str,
    ) -> KnowledgeResponse:
        evidence = retriever.retrieve(tenant, query)
        return self.answer_or_enqueue(
            tenant=tenant,
            query=query,
            evidence=evidence,
            source_id=source_id,
        )

    def job_status(self, tenant: TenantId, job_id: str) -> AcquisitionJob | None:
        return self.store.get(tenant, job_id)

    def cancel(self, tenant: TenantId, job_id: str) -> AcquisitionJob:
        return self.store.transition(tenant, job_id, JobStatus.CANCELLED)


def normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip()).casefold()
    if not normalized or len(normalized.encode()) > 4096:
        raise ValueError("query is empty or exceeds the bounded UTF-8 size")
    return normalized
