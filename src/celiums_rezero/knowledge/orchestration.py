"""Host-owned governed orchestration around a frozen model runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from celiums_rezero.knowledge.finalization import FinalAnswer
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EvidenceBundle,
    EvidenceHit,
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
    def retrieve_for_job(self, job: AcquisitionJob) -> EvidenceBundle: ...


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
        bundle = self.evidence.retrieve_for_job(job)
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
            timeout_seconds=timeout_seconds,
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
