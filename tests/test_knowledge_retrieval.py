from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from celiums_rezero.knowledge import (
    AcquisitionPolicy,
    HyphaeRetrievalGateway,
    InMemoryTenantStore,
    KnowledgeCoordinator,
    RetrievalConfig,
    RetrievalContractError,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.schemas import SourcePolicy


@dataclass(frozen=True)
class FakeResponse:
    kind: str
    value: object


class FakeHyphaeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[int, dict[str, object]]] = []

    def search_collection(
        self,
        collection: int,
        request: dict[str, object],
        *,
        options: object | None = None,
    ) -> FakeResponse:
        assert options is None
        self.calls.append((collection, request))
        return self.response


class FakeEmbedder:
    profile = "fixture-embedder-v1"

    def embed(self, text: str) -> tuple[float, ...]:
        assert text
        return (0.25, 0.75)


def snapshot() -> dict[str, object]:
    return {
        "directory_lineage": bytes(range(24)),
        "visible_csn": 7,
        "catalog_version": 3,
        "root_digest": bytes(range(32)),
        "logical_time_micros": 11,
    }


def search_response(*, score: float = 0.92, body: str = "Verified documentation") -> FakeResponse:
    return FakeResponse(
        "integrated_search",
        {
            "snapshot": snapshot(),
            "hits": [
                {
                    "object_id": 17,
                    "score": score,
                    "doc_values": {
                        "body": body,
                        "source_id": "official_docs",
                        "source_version": "v1",
                        "content_digest": hashlib.sha256(body.encode()).hexdigest(),
                        "corpus_generation": "generation-v1",
                    },
                }
            ],
            "vector_branches": [
                {
                    "target": "semantic",
                    "strategy": "exact_filtered",
                    "approximate": False,
                    "exact_reranked": True,
                    "eligible_documents": 1,
                    "candidate_count": 1,
                    "visited_nodes": 0,
                }
            ],
            "approximate": False,
        },
    )


def gateway(client: FakeHyphaeClient, tenant: str = "tenant_a") -> HyphaeRetrievalGateway:
    return HyphaeRetrievalGateway(
        tenant=TenantId(tenant),
        client=client,
        config=RetrievalConfig(
            collection=41,
            corpus_generation="generation-v1",
            vector_target="semantic",
            score_scale=1.0,
            require_exact_vector_receipt=True,
        ),
        embedder=FakeEmbedder(),
    )


def test_gateway_builds_hybrid_request_and_hydrates_verified_body() -> None:
    client = FakeHyphaeClient(search_response())
    evidence = gateway(client).retrieve(TenantId("tenant_a"), "How does this work?")
    assert evidence.hits[0].text == "Verified documentation"
    assert evidence.snapshot_fingerprint is not None
    collection, request = client.calls[0]
    assert collection == 41
    assert request["lexical"] == {
        "query": "how does this work?",
        "candidate_limit": 32,
        "weight": 1,
    }
    assert request["vectors"] == [
        {
            "target": "semantic",
            "query": [0.25, 0.75],
            "candidate_limit": 32,
            "weight": 1,
            "execution": {"kind": "exact"},
        }
    ]


def test_gateway_rejects_cross_tenant_access() -> None:
    with pytest.raises(PermissionError, match="different tenant"):
        gateway(FakeHyphaeClient(search_response())).retrieve(
            TenantId("tenant_b"), "question"
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda response: response.update(snapshot=None),
        lambda response: response["hits"][0]["doc_values"].pop("body"),
        lambda response: response["hits"][0]["doc_values"].update(body="tampered"),
        lambda response: response["hits"][0]["doc_values"].update(
            corpus_generation="other"
        ),
    ],
)
def test_gateway_fails_closed_on_unhydrated_or_tampered_hits(mutator: object) -> None:
    values = search_response().value
    assert isinstance(values, dict)
    assert callable(mutator)
    mutator(values)
    with pytest.raises(RetrievalContractError):
        gateway(FakeHyphaeClient(FakeResponse("integrated_search", values))).retrieve(
            TenantId("tenant_a"), "question"
        )


def test_gateway_rejects_non_exact_vector_receipt() -> None:
    values = search_response().value
    assert isinstance(values, dict)
    values["vector_branches"][0]["strategy"] = "filter_aware_ann"
    with pytest.raises(RetrievalContractError, match="not exact"):
        gateway(FakeHyphaeClient(FakeResponse("integrated_search", values))).retrieve(
            TenantId("tenant_a"), "question"
        )


def test_coordinator_retrieves_then_returns_ready_evidence() -> None:
    tenant = TenantId("tenant_a")
    retriever = gateway(FakeHyphaeClient(search_response()))
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(minimum_score=0.7, minimum_margin=0.05),
        acquisition=AcquisitionPolicy(
            version="policy-v1",
            sources=(
                SourcePolicy(
                    source_id="official_docs",
                    allowed_hosts=("docs.example.com",),
                ),
            ),
        ),
        store=InMemoryTenantStore(),
        embedding_profile="fixture-embedder-v1",
    )
    result = coordinator.retrieve_or_enqueue(
        tenant=tenant,
        query="Question",
        retriever=retriever,
        source_id="official_docs",
    )
    assert result.status == "evidence_ready"
    assert result.evidence_handles


def test_coordinator_retrieves_then_enqueues_when_evidence_is_below_threshold() -> None:
    tenant = TenantId("tenant_a")
    retriever = gateway(FakeHyphaeClient(search_response(score=0.2)))
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(minimum_score=0.7),
        acquisition=AcquisitionPolicy(
            version="policy-v1",
            sources=(
                SourcePolicy(
                    source_id="official_docs",
                    allowed_hosts=("docs.example.com",),
                ),
            ),
        ),
        store=InMemoryTenantStore(),
        embedding_profile="fixture-embedder-v1",
    )
    result = coordinator.retrieve_or_enqueue(
        tenant=tenant,
        query="Question",
        retriever=retriever,
        source_id="official_docs",
    )
    assert result.status == "knowledge_pending"
    assert result.job_id is not None


def test_approximate_search_is_partial_under_exact_only_policy() -> None:
    values = search_response().value
    assert isinstance(values, dict)
    values["approximate"] = True
    retriever = HyphaeRetrievalGateway(
        tenant=TenantId("tenant_a"),
        client=FakeHyphaeClient(FakeResponse("integrated_search", values)),
        config=RetrievalConfig(
            collection=41,
            corpus_generation="generation-v1",
            vector_target="semantic",
            score_scale=1.0,
            require_exact_vector_receipt=False,
        ),
        embedder=FakeEmbedder(),
    )
    tenant = TenantId("tenant_a")
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(minimum_score=0.7, allow_approximate=False),
        acquisition=AcquisitionPolicy(
            version="policy-v1",
            sources=(
                SourcePolicy(
                    source_id="official_docs",
                    allowed_hosts=("docs.example.com",),
                ),
            ),
        ),
        store=InMemoryTenantStore(),
        embedding_profile="fixture-embedder-v1",
    )
    result = coordinator.retrieve_or_enqueue(
        tenant=tenant,
        query="Question",
        retriever=retriever,
        source_id="official_docs",
    )
    assert result.status == "knowledge_pending"


def test_vector_target_requires_an_embedder() -> None:
    with pytest.raises(ValueError, match="requires a pinned embedding"):
        HyphaeRetrievalGateway(
            tenant=TenantId("tenant_a"),
            client=FakeHyphaeClient(search_response()),
            config=RetrievalConfig(
                collection=41,
                corpus_generation="generation-v1",
                vector_target="semantic",
            ),
        )
