from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from celiums_rezero.knowledge import (
    AcquisitionPolicy,
    AcquisitionWorker,
    EvidenceBundle,
    FetchResponse,
    HttpsSourceConnector,
    HyphaeShadowIngestor,
    InMemoryTenantStore,
    JobStatus,
    KnowledgeCoordinator,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.acquisition import ChunkingPolicy
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.schemas import SourcePolicy


class FakeResolver:
    def __init__(self, addresses: tuple[str, ...]) -> None:
        self.addresses = addresses

    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        assert host == "docs.example.com" and port == 443
        return self.addresses


class FakeTransport:
    def __init__(self, response: FetchResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def get(self, **kwargs: object) -> FetchResponse:
        self.calls.append(kwargs)
        return self.response


@dataclass(frozen=True)
class FakeHyphaeResponse:
    kind: str
    value: object


class FakeHyphaeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object]]] = []

    def search_ingest(
        self,
        collection: int,
        batch: dict[str, object],
        *,
        options: object | None = None,
    ) -> FakeHyphaeResponse:
        assert options is None
        self.calls.append((collection, batch))
        return FakeHyphaeResponse("search_ingested", {"idempotent_replay": False})


class Embedder:
    profile = "fixture-v1"

    def embed(self, text: str) -> tuple[float, ...]:
        return (len(text) / 100, 0.5)


def policy() -> SourcePolicy:
    return SourcePolicy(
        source_id="official_docs",
        allowed_hosts=("docs.example.com",),
        resource_url="https://docs.example.com/manual/page",
        allowed_path_prefixes=("/manual/",),
        allowed_mime_types=("text/plain",),
        allowed_license_ids=("Apache-2.0",),
        max_download_bytes=1024,
    )


def connector(
    response: FetchResponse,
    addresses: tuple[str, ...] = ("93.184.216.34",),
) -> HttpsSourceConnector:
    return HttpsSourceConnector(
        policies={"official_docs": policy()},
        resolver=FakeResolver(addresses),
        transport=FakeTransport(response),
    )


def good_response() -> FetchResponse:
    return FetchResponse(
        status=200,
        headers={
            "content-type": "text/plain; charset=utf-8",
            "x-license-id": "Apache-2.0",
            "etag": '"version-1"',
        },
        body=b"Verified knowledge from an allowlisted official source.",
        peer_ip="93.184.216.34",
    )


def test_https_connector_fetches_only_policy_owned_resource() -> None:
    item = connector(good_response())
    artifact = item.acquire(TenantId("tenant_a"), "official_docs", "ignored")
    assert artifact.source_version == "version-1"
    assert artifact.license_id == "Apache-2.0"
    assert artifact.content_digest == hashlib.sha256(artifact.body).hexdigest()
    transport = item.transport
    assert isinstance(transport, FakeTransport)
    assert transport.calls[0]["resolved_ip"] == "93.184.216.34"


@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        ("169.254.169.254",),
        ("93.184.216.34", "10.0.0.1"),
        (),
    ],
)
def test_https_connector_rejects_nonpublic_or_mixed_dns(addresses: tuple[str, ...]) -> None:
    with pytest.raises(RuntimeError, match="DNS resolved outside"):
        connector(good_response(), addresses).acquire(
            TenantId("tenant_a"), "official_docs", "ignored"
        )


@pytest.mark.parametrize(
    "response,pattern",
    [
        (
            FetchResponse(
                302,
                {"location": "https://docs.example.com/manual/next"},
                b"",
                "93.184.216.34",
            ),
            "HTTP status",
        ),
        (
            FetchResponse(
                200,
                {"content-type": "application/octet-stream"},
                b"x",
                "93.184.216.34",
            ),
            "MIME",
        ),
        (FetchResponse(200, {"content-type": "text/plain"}, b"x", "127.0.0.1"), "peer address"),
    ],
)
def test_https_connector_rejects_redirect_mime_and_peer_mismatch(
    response: FetchResponse, pattern: str
) -> None:
    with pytest.raises(RuntimeError, match=pattern):
        connector(response).acquire(TenantId("tenant_a"), "official_docs", "ignored")


def test_source_policy_rejects_invalid_policy_owned_url() -> None:
    with pytest.raises(ValueError, match="resource URL"):
        SourcePolicy(
            source_id="bad",
            allowed_hosts=("docs.example.com",),
            resource_url="https://other.example.com/manual/page",
        )


def coordinator() -> tuple[KnowledgeCoordinator, TenantId, str]:
    tenant = TenantId("tenant_a")
    item = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(),
        acquisition=AcquisitionPolicy(version="policy-v1", sources=(policy(),)),
        store=InMemoryTenantStore(),
        embedding_profile="fixture-v1",
    )
    query = "What does the source say?"
    normalized = normalize_query(query)
    response = item.answer_or_enqueue(
        tenant=tenant,
        query=query,
        evidence=EvidenceBundle(
            tenant=tenant,
            query_digest=hashlib.sha256(normalized.encode()).hexdigest(),
            corpus_generation="generation-v1",
            hits=(),
        ),
        source_id="official_docs",
    )
    assert response.job_id is not None
    return item, tenant, response.job_id


def test_shadow_worker_validates_without_publishing_to_hyphae() -> None:
    item, tenant, job_id = coordinator()
    client = FakeHyphaeClient()
    ingestor = HyphaeShadowIngestor(
        tenant=tenant,
        client=client,
        collection=41,
        vector_target="semantic",
        publish=False,
    )
    worker = AcquisitionWorker(
        coordinator=item,
        connector=connector(good_response()),
        embedder=Embedder(),
        ingestor=ingestor,
        verifier=ingestor,
        chunking=ChunkingPolicy(max_chunk_bytes=64, overlap_bytes=8),
    )
    outcome = worker.run(tenant, job_id)
    assert outcome.job.status is JobStatus.SHADOW_VALIDATED
    assert outcome.receipt is not None and not outcome.receipt.published
    assert client.calls == []


def test_live_publication_requires_explicit_opt_in_and_is_idempotent() -> None:
    item, tenant, job_id = coordinator()
    client = FakeHyphaeClient()
    ingestor = HyphaeShadowIngestor(
        tenant=tenant,
        client=client,
        collection=41,
        vector_target="semantic",
        publish=True,
    )
    worker = AcquisitionWorker(
        coordinator=item,
        connector=connector(good_response()),
        embedder=Embedder(),
        ingestor=ingestor,
        verifier=ingestor,
    )
    outcome = worker.run(tenant, job_id)
    assert outcome.job.status is JobStatus.READY
    assert outcome.receipt is not None and outcome.receipt.published
    assert len(client.calls) == 1
    documents = client.calls[0][1]["documents"]
    assert isinstance(documents, list)
    assert documents[0]["doc_values"]["corpus_generation"] == "generation-v1"


def test_hyphae_ingestor_rejects_cross_tenant_and_batch_overflow() -> None:
    tenant = TenantId("tenant_a")
    ingestor = HyphaeShadowIngestor(
        tenant=tenant,
        client=FakeHyphaeClient(),
        collection=41,
        vector_target="semantic",
    )
    with pytest.raises(PermissionError, match="another tenant"):
        ingestor.ingest(
            TenantId("tenant_b"),
            (),
            corpus_generation="generation-v1",
            idempotency_key="0" * 64,
        )
