"""Host-owned governed orchestration around a frozen model runtime."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.finalization import FinalAnswer, FinalizationTimeout
from celiums_rezero.knowledge.retrieval import GenerationRoutedRetriever
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EvidenceBundle,
    EvidenceHit,
    JobStatus,
    SufficiencyPolicy,
)


@dataclass(frozen=True, slots=True)
class FrozenModelIdentity:
    model_id: str
    revision: str
    manifest_digest: str
    runtime_version: str


@dataclass(frozen=True, slots=True)
class GovernedModelRequest:
    query: str
    generation_id: str
    passages: tuple[EvidenceHit, ...]
    maximum_output_bytes: int = 16_384


@dataclass(frozen=True, slots=True)
class QuotedClaim:
    handle: str
    quote: str


@dataclass(frozen=True, slots=True)
class GovernedModelResult:
    identity: FrozenModelIdentity
    decision: str
    claims: tuple[QuotedClaim, ...]


class FrozenGemmaRuntime(Protocol):
    @property
    def identity(self) -> FrozenModelIdentity: ...

    def infer(
        self, request: GovernedModelRequest, *, timeout_seconds: float
    ) -> GovernedModelResult: ...


class EvidenceProvider(Protocol):
    def retrieve_for_job(
        self, job: AcquisitionJob, *, timeout_seconds: float
    ) -> EvidenceBundle: ...


class GenerationRoutedEvidenceProvider:
    """Binds finalization retrieval to one durable answering job."""

    def __init__(self, *, retriever: GenerationRoutedRetriever) -> None:
        self.retriever = retriever

    def retrieve_for_job(
        self, job: AcquisitionJob, *, timeout_seconds: float
    ) -> EvidenceBundle:
        if job.status is not JobStatus.ANSWERING:
            raise ValueError("routed evidence requires an answering job")
        if job.tenant != self.retriever.tenant:
            raise PermissionError("routed evidence provider belongs to another tenant")
        normalized = normalize_query(job.query)
        if normalized != job.query or hashlib.sha256(normalized.encode()).hexdigest() != (
            job.query_digest
        ):
            raise ValueError("durable job query binding is invalid")
        try:
            bundle = self.retriever.retrieve(
                job.tenant,
                normalized,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError as error:
            raise FinalizationTimeout("routed evidence retrieval timed out") from error
        except Exception as error:
            if getattr(error, "code", None) == "deadline_exceeded":
                raise FinalizationTimeout("routed evidence retrieval timed out") from error
            raise
        if (
            bundle.tenant != job.tenant
            or bundle.query_digest != job.query_digest
            or bundle.corpus_generation != job.corpus_generation
        ):
            raise PermissionError("routed evidence does not match the finalization job")
        return bundle


class HostGemmaAnswerer:
    def __init__(
        self,
        *,
        runtime: FrozenGemmaRuntime,
        evidence: EvidenceProvider,
        expected_identity: FrozenModelIdentity | None = None,
        sufficiency: SufficiencyPolicy | None = None,
        maximum_evidence_bytes: int = 65_536,
    ) -> None:
        self.runtime = runtime
        self.evidence = evidence
        self.expected_identity = (
            runtime.identity if expected_identity is None else expected_identity
        )
        self.sufficiency = SufficiencyPolicy() if sufficiency is None else sufficiency
        self.maximum_evidence_bytes = maximum_evidence_bytes

    def answer(
        self, job: AcquisitionJob, *, timeout_seconds: float
    ) -> FinalAnswer | None:
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("answer timeout must be finite and positive")
        deadline = time.monotonic() + timeout_seconds
        bundle = self.evidence.retrieve_for_job(
            job,
            timeout_seconds=_remaining_seconds(deadline),
        )
        if (
            bundle.tenant != job.tenant
            or bundle.query_digest != job.query_digest
            or bundle.corpus_generation != job.corpus_generation
        ):
            raise PermissionError("orchestrator evidence crossed tenant or generation")
        if self.sufficiency.decide(bundle).value != "supported":
            return None
        passages: list[EvidenceHit] = []
        used = 0
        for hit in sorted(bundle.hits, key=lambda value: (-value.score, value.handle)):
            if not hit.active or not hit.trusted:
                continue
            encoded = len(hit.text.encode())
            if used + encoded > self.maximum_evidence_bytes:
                break
            passages.append(hit)
            used += encoded
        if not passages:
            return None
        identity = self.runtime.identity
        if identity != self.expected_identity:
            raise ValueError("frozen model identity drifted before inference")
        result = self.runtime.infer(
            GovernedModelRequest(job.query, job.corpus_generation, tuple(passages)),
            timeout_seconds=_remaining_seconds(deadline),
        )
        if result.identity != identity or result.decision not in {"answer", "insufficient"}:
            raise ValueError("frozen model identity or decision is invalid")
        if result.decision == "insufficient":
            return None
        by_handle = {passage.handle: passage for passage in passages}
        if not result.claims or len({claim.handle for claim in result.claims}) != len(
            result.claims
        ):
            raise ValueError("model claims are empty or duplicate evidence handles")
        rendered: list[str] = []
        handles: list[str] = []
        for claim in result.claims:
            passage = by_handle.get(claim.handle)
            if passage is None or not claim.quote or claim.quote not in passage.text:
                raise ValueError("model claim is not a verbatim cited quotation")
            rendered.append(claim.quote)
            handles.append(claim.handle)
        answer = "\n\n".join(rendered)
        if len(answer.encode()) > 16_384:
            raise ValueError("governed model answer exceeds its byte bound")
        return FinalAnswer(answer, tuple(handles))


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FinalizationTimeout("answer deadline exceeded")
    return remaining
