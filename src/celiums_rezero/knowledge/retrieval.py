"""Read-only tenant-bound Hyphae retrieval and evidence hydration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.generation import GenerationAuthority
from celiums_rezero.knowledge.schemas import EvidenceBundle, EvidenceHit, TenantId
from celiums_rezero.lab.serialization import canonical_json, content_hash


class RetrievalContractError(RuntimeError):
    """Hyphae response did not satisfy the evidence contract."""


class HyphaeSearchClient(Protocol):
    """Smallest read-only subset of the Hyphae v2 Python client."""

    def search_collection(
        self,
        collection: int,
        request: dict[str, object],
        *,
        options: object | None = None,
    ) -> object: ...


class EmbeddingProvider(Protocol):
    """Pinned caller-owned query embedder; Hyphae never runs the model."""

    @property
    def profile(self) -> str: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    collection: int
    corpus_generation: str
    backend_id: str | None = None
    body_field: str = "body"
    source_id_field: str = "source_id"
    source_version_field: str = "source_version"
    digest_field: str = "content_digest"
    active_generation_field: str = "corpus_generation"
    vector_target: str | None = None
    limit: int = 8
    candidate_limit: int = 32
    lexical_weight: int = 1
    vector_weight: int = 1
    score_scale: float = 1.0
    require_exact_vector_receipt: bool = False

    def __post_init__(self) -> None:
        if self.collection < 1 or not self.corpus_generation:
            raise ValueError("collection and corpus generation are required")
        if self.backend_id is not None and (
            len(self.backend_id) != 64
            or any(character not in "0123456789abcdef" for character in self.backend_id)
        ):
            raise ValueError("retrieval backend ID must be lowercase SHA-256")
        if not 1 <= self.limit <= 64 or not self.limit <= self.candidate_limit <= 10_000:
            raise ValueError("retrieval result and candidate limits are invalid")
        if min(self.lexical_weight, self.vector_weight) < 1:
            raise ValueError("retrieval branch weights must be positive")
        if not isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError("score scale must be finite and positive")
        if self.require_exact_vector_receipt and self.vector_target is None:
            raise ValueError("exact vector receipt requires a vector target")
        fields = (
            self.body_field,
            self.source_id_field,
            self.source_version_field,
            self.digest_field,
            self.active_generation_field,
        )
        if any(not field or len(field) > 128 for field in fields):
            raise ValueError("retrieval field names are invalid")


class HyphaeRetrievalGateway:
    """Read-only gateway bound to one tenant, collection, and corpus generation."""

    def __init__(
        self,
        *,
        tenant: TenantId,
        client: HyphaeSearchClient,
        config: RetrievalConfig,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        if config.vector_target is not None and embedder is None:
            raise ValueError("a vector target requires a pinned embedding provider")
        self.tenant = tenant
        self.client = client
        self.config = config
        self.embedder = embedder

    def retrieve(self, tenant: TenantId, query: str) -> EvidenceBundle:
        if tenant != self.tenant:
            raise PermissionError("retrieval gateway is bound to a different tenant")
        normalized = normalize_query(query)
        response = self.client.search_collection(
            self.config.collection,
            self._request(normalized),
        )
        kind = getattr(response, "kind", None)
        value = getattr(response, "value", None)
        if kind != "integrated_search" or not isinstance(value, dict):
            raise RetrievalContractError("Hyphae returned an unexpected response kind")
        return self._bundle(normalized, value)

    def _request(self, query: str) -> dict[str, object]:
        vectors: list[dict[str, object]] = []
        if self.config.vector_target is not None:
            assert self.embedder is not None
            values = self.embedder.embed(query)
            if not values or any(not isfinite(value) for value in values):
                raise RetrievalContractError("embedding provider returned an invalid vector")
            vectors.append(
                {
                    "target": self.config.vector_target,
                    "query": list(values),
                    "candidate_limit": self.config.candidate_limit,
                    "weight": self.config.vector_weight,
                    "execution": {"kind": "exact"},
                }
            )
        return {
            "lexical": {
                "query": query,
                "candidate_limit": self.config.candidate_limit,
                "weight": self.config.lexical_weight,
            },
            "vectors": vectors,
            "filter": {
                "kind": "compare",
                "field": self.config.active_generation_field,
                "operator": "equal",
                "value": self.config.corpus_generation,
            },
            "sort": [],
            "facets": [],
            "aggregations": [],
            "limit": self.config.limit,
        }

    def _bundle(self, query: str, value: dict[str, Any]) -> EvidenceBundle:
        hits = value.get("hits")
        if not isinstance(hits, list):
            raise RetrievalContractError("Hyphae search result has no bounded hit list")
        hydrated = tuple(self._hit(hit) for hit in hits)
        approximate = value.get("approximate", False)
        if not isinstance(approximate, bool):
            raise RetrievalContractError("Hyphae approximation evidence is invalid")
        if self.config.require_exact_vector_receipt:
            branches = value.get("vector_branches")
            if not isinstance(branches, list) or len(branches) != 1:
                raise RetrievalContractError("Hyphae exact vector receipt is absent")
            branch = branches[0]
            if (
                not isinstance(branch, dict)
                or branch.get("target") != self.config.vector_target
                or branch.get("strategy") != "exact_filtered"
                or branch.get("approximate") is not False
                or branch.get("exact_reranked") is not True
                or approximate
            ):
                raise RetrievalContractError("Hyphae vector execution was not exact")
        snapshot = value.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RetrievalContractError("Hyphae search result has no snapshot identity")
        if self.config.backend_id is not None:
            lineage = snapshot.get("directory_lineage")
            if not isinstance(lineage, bytes) or len(lineage) != 24:
                raise RetrievalContractError("Hyphae directory lineage is invalid")
            if hashlib.sha256(lineage).hexdigest() != self.config.backend_id:
                raise RetrievalContractError("Hyphae backend identity does not match routing")
        fingerprint = hashlib.sha256(
            canonical_json(_snapshot_primitive(snapshot)).encode()
        ).hexdigest()
        return EvidenceBundle(
            tenant=self.tenant,
            query_digest=hashlib.sha256(query.encode()).hexdigest(),
            corpus_generation=self.config.corpus_generation,
            hits=hydrated,
            snapshot_fingerprint=fingerprint,
            approximate=approximate,
        )

    def _hit(self, hit: object) -> EvidenceHit:
        if not isinstance(hit, dict):
            raise RetrievalContractError("Hyphae hit must be an object")
        object_id = hit.get("object_id")
        raw_score = hit.get("score")
        doc_values = hit.get("doc_values")
        if isinstance(object_id, bool) or not isinstance(object_id, (int, str)):
            raise RetrievalContractError("Hyphae hit object identity is invalid")
        if not isinstance(raw_score, (int, float, str)) or isinstance(raw_score, bool):
            raise RetrievalContractError("Hyphae hit score is invalid")
        try:
            object_value = int(object_id)
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise RetrievalContractError("Hyphae hit score or identity is invalid") from error
        if object_value < 1 or not isfinite(score) or score < 0 or not isinstance(doc_values, dict):
            raise RetrievalContractError("Hyphae hit fields are invalid")
        body = doc_values.get(self.config.body_field)
        source_id = doc_values.get(self.config.source_id_field)
        source_version = doc_values.get(self.config.source_version_field)
        digest = doc_values.get(self.config.digest_field)
        generation = doc_values.get(self.config.active_generation_field)
        if not all(isinstance(item, str) for item in (body, source_id, source_version, digest)):
            raise RetrievalContractError("Hyphae hit is missing hydrated evidence fields")
        if generation != self.config.corpus_generation:
            raise RetrievalContractError("Hyphae returned a hit from another corpus generation")
        assert isinstance(body, str)
        assert isinstance(source_id, str)
        assert isinstance(source_version, str)
        assert isinstance(digest, str)
        observed = hashlib.sha256(body.encode()).hexdigest()
        if digest != observed:
            raise RetrievalContractError("Hyphae evidence body digest does not match")
        calibrated = min(score / self.config.score_scale, 1.0)
        handle_identity = {"collection": self.config.collection, "id": object_value}
        return EvidenceHit(
            handle=f"passage_{content_hash(handle_identity)}",
            source_id=source_id,
            source_version=source_version,
            text=body,
            score=calibrated,
            content_digest=digest,
        )


class GenerationRoutedRetriever:
    """Builds a fresh immutable gateway from one active-generation snapshot per query."""

    def __init__(
        self,
        *,
        tenant: TenantId,
        authority: GenerationAuthority,
        client: HyphaeSearchClient,
        embedder: EmbeddingProvider | None = None,
        limit: int = 8,
        candidate_limit: int = 32,
    ) -> None:
        self.tenant = tenant
        self.authority = authority
        self.client = client
        self.embedder = embedder
        self.limit = limit
        self.candidate_limit = candidate_limit
        if authority.store.tenant != tenant:
            raise PermissionError("generation authority belongs to another tenant")

    def retrieve(self, tenant: TenantId, query: str) -> EvidenceBundle:
        if tenant != self.tenant:
            raise PermissionError("generation router is bound to another tenant")
        snapshot = self.authority.snapshot()
        if snapshot.claims_paused:
            raise RetrievalContractError("active generation claims are paused")
        gateway = HyphaeRetrievalGateway(
            tenant=tenant,
            client=self.client,
            config=RetrievalConfig(
                collection=snapshot.target.collection,
                corpus_generation=snapshot.generation_id,
                backend_id=snapshot.target.backend_id,
                vector_target=(
                    snapshot.target.vector_target if self.embedder is not None else None
                ),
                limit=self.limit,
                candidate_limit=self.candidate_limit,
            ),
            embedder=self.embedder,
        )
        return gateway.retrieve(tenant, query)


def _snapshot_primitive(snapshot: dict[str, Any]) -> dict[str, object]:
    required = (
        "directory_lineage",
        "visible_csn",
        "catalog_version",
        "root_digest",
        "logical_time_micros",
    )
    if any(field not in snapshot for field in required):
        raise RetrievalContractError("Hyphae snapshot identity is incomplete")
    lineage = snapshot["directory_lineage"]
    root = snapshot["root_digest"]
    if not isinstance(lineage, (bytes, str)) or not isinstance(root, (bytes, str)):
        raise RetrievalContractError("Hyphae snapshot digests have invalid types")
    return {
        "directory_lineage": lineage.hex() if isinstance(lineage, bytes) else lineage,
        "visible_csn": snapshot["visible_csn"],
        "catalog_version": snapshot["catalog_version"],
        "root_digest": root.hex() if isinstance(root, bytes) else root,
        "logical_time_micros": snapshot["logical_time_micros"],
    }
