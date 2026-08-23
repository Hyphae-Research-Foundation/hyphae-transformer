"""Phase 3 shadow-mode HTTPS acquisition and bounded Hyphae ingestion adapters."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
from dataclasses import dataclass, replace
from typing import Protocol
from urllib.parse import urlparse

from celiums_rezero.knowledge.acquisition import AcquisitionError
from celiums_rezero.knowledge.publication import (
    PublicationReceiptStore,
    embedding_digest,
    receipt_primitive,
)
from celiums_rezero.knowledge.schemas import (
    AcquisitionReceipt,
    EmbeddedChunk,
    IngestMode,
    IngestReceipt,
    PublicationAuthorization,
    PublicationTarget,
    SourceArtifact,
    SourcePolicy,
    TenantId,
)
from celiums_rezero.lab.serialization import canonical_json, content_hash


class Resolver(Protocol):
    def resolve(self, host: str, port: int) -> tuple[str, ...]: ...


class FetchTransport(Protocol):
    def get(
        self,
        *,
        host: str,
        port: int,
        path: str,
        resolved_ip: str,
        max_bytes: int,
        timeout_seconds: float,
    ) -> FetchResponse: ...


class HyphaeIngestClient(Protocol):
    def search_ingest(
        self,
        collection: int,
        batch: dict[str, object],
        *,
        options: object | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class FetchResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    peer_ip: str


@dataclass(frozen=True, slots=True)
class HttpsFetchPolicy:
    timeout_seconds: float = 30.0
    max_redirects: int = 0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or not 0 <= self.max_redirects <= 5:
            raise ValueError("HTTPS fetch limits are invalid")


class SystemResolver:
    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        values = {
            str(item[4][0])
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
        return tuple(sorted(values))


class PinnedHttpsTransport:
    """HTTPS transport that connects to a validated IP while retaining hostname TLS."""

    def get(
        self,
        *,
        host: str,
        port: int,
        path: str,
        resolved_ip: str,
        max_bytes: int,
        timeout_seconds: float,
    ) -> FetchResponse:
        connection = _PinnedHttpsConnection(
            host=host,
            resolved_ip=resolved_ip,
            port=port,
            timeout=timeout_seconds,
        )
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Host": host,
                    "User-Agent": "hyphae-transformer-knowledge/0.1",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            content_length = response.getheader("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise AcquisitionError("source response exceeds its byte budget")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise AcquisitionError("source response exceeds its byte budget")
            peer = connection.sock.getpeername()[0] if connection.sock is not None else resolved_ip
            return FetchResponse(
                status=response.status,
                headers={name.lower(): value for name, value in response.getheaders()},
                body=body,
                peer_ip=peer,
            )
        finally:
            connection.close()


class HttpsSourceConnector:
    """Fetches one immutable policy-owned URL in shadow mode; model URLs are absent."""

    def __init__(
        self,
        *,
        policies: dict[str, SourcePolicy],
        resolver: Resolver | None = None,
        transport: FetchTransport | None = None,
        fetch_policy: HttpsFetchPolicy | None = None,
    ) -> None:
        self.policies = dict(policies)
        self.resolver = SystemResolver() if resolver is None else resolver
        self.transport = PinnedHttpsTransport() if transport is None else transport
        self.fetch_policy = HttpsFetchPolicy() if fetch_policy is None else fetch_policy

    def acquire(self, tenant: TenantId, source_id: str, query: str) -> SourceArtifact:
        del query
        policy = self.policies.get(source_id)
        if policy is None or policy.resource_url is None:
            raise AcquisitionError("source has no immutable policy-owned URL")
        url = policy.resource_url
        if not policy.permits_url(url):
            raise AcquisitionError("source URL is not allowed by policy")
        parsed = urlparse(url)
        host = parsed.hostname
        assert host is not None
        port = parsed.port or 443
        addresses = self.resolver.resolve(host, port)
        if not addresses or any(not _public_ip(address) for address in addresses):
            raise AcquisitionError("source DNS resolved outside the public Internet")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        response = self.transport.get(
            host=host,
            port=port,
            path=path,
            resolved_ip=addresses[0],
            max_bytes=policy.max_download_bytes,
            timeout_seconds=self.fetch_policy.timeout_seconds,
        )
        if response.peer_ip not in addresses or not _public_ip(response.peer_ip):
            raise AcquisitionError("source peer address did not match validated DNS")
        if response.status != 200:
            raise AcquisitionError(f"source returned unexpected HTTP status {response.status}")
        if "location" in response.headers:
            raise AcquisitionError("source redirects are disabled in Phase 3 shadow mode")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in policy.allowed_mime_types:
            raise AcquisitionError("source response MIME type is not allowed")
        version = response.headers.get("etag") or hashlib.sha256(response.body).hexdigest()
        license_id = response.headers.get("x-license-id", "UNKNOWN")
        return SourceArtifact(
            tenant=tenant,
            source_id=source_id,
            source_version=version.strip('"'),
            body=response.body,
            content_type=content_type,
            license_id=license_id,
            content_digest=hashlib.sha256(response.body).hexdigest(),
        )


class HyphaeShadowIngestor:
    """Builds canonical bounded batches; publication requires explicit opt-in."""

    def __init__(
        self,
        *,
        tenant: TenantId,
        client: HyphaeIngestClient,
        collection: int,
        vector_target: str,
        backend_id: str | None = None,
        publish: bool = False,
        max_batch_documents: int = 256,
        receipt_store: PublicationReceiptStore | None = None,
    ) -> None:
        if collection < 1 or not vector_target or not 1 <= max_batch_documents <= 256:
            raise ValueError("Hyphae ingestion configuration is invalid")
        if not isinstance(publish, bool):
            raise ValueError("Hyphae publication opt-in must be a boolean")
        self.tenant = tenant
        self.client = client
        self.collection = collection
        self.vector_target = vector_target
        self.publish = publish
        self.max_batch_documents = max_batch_documents
        if publish and (receipt_store is None or backend_id is None):
            raise ValueError("live Hyphae publication requires a durable receipt store")
        if not publish and backend_id is not None:
            raise ValueError("shadow Hyphae ingestion cannot declare a live backend")
        self._target = (
            None
            if backend_id is None
            else PublicationTarget(backend_id, collection, vector_target)
        )
        self.receipt_store = receipt_store
        self.receipts: dict[str, IngestReceipt] = {}

    @property
    def mode(self) -> IngestMode:
        return IngestMode.LIVE if self.publish else IngestMode.SHADOW

    @property
    def target(self) -> PublicationTarget | None:
        return self._target

    def ingest(
        self,
        tenant: TenantId,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        corpus_generation: str,
        idempotency_key: str,
        authorization: PublicationAuthorization | None = None,
    ) -> IngestReceipt:
        if tenant != self.tenant:
            raise PermissionError("Hyphae ingestor is bound to another tenant")
        if not chunks or len(chunks) > self.max_batch_documents:
            raise AcquisitionError("Hyphae ingest batch exceeds its document bound")
        first = chunks[0].chunk
        if any(
            item.chunk.source_id != first.source_id
            or item.chunk.source_version != first.source_version
            for item in chunks
        ):
            raise AcquisitionError("Hyphae ingest batch mixes source identities")
        durable = (
            None
            if self.receipt_store is None
            else self.receipt_store.load_ingest(tenant, idempotency_key)
        )
        existing = durable or self.receipts.get(idempotency_key)
        if existing is not None:
            if self.mode is IngestMode.LIVE:
                if authorization is None:
                    raise PermissionError("live Hyphae replay is not authorized")
                if existing.replayed or not _receipt_digest_matches(existing):
                    raise AcquisitionError("durable Hyphae replay receipt is invalid")
                stored_authorization = self.receipt_store.load_authorization(
                    tenant, authorization.authorization_id or ""
                ) if self.receipt_store is not None else None
                if stored_authorization != authorization:
                    raise PermissionError("durable Hyphae replay authorization is invalid")
                self._validate_authorization(
                    authorization,
                    chunks,
                    corpus_generation=corpus_generation,
                    idempotency_key=idempotency_key,
                )
            self._validate_replay(
                existing,
                chunks,
                corpus_generation=corpus_generation,
                authorization=authorization,
            )
            return IngestReceipt(
                tenant=existing.tenant,
                source_id=existing.source_id,
                source_version=existing.source_version,
                corpus_generation=existing.corpus_generation,
                chunk_ids=existing.chunk_ids,
                idempotency_key=existing.idempotency_key,
                replayed=True,
                published=existing.published,
                mode=existing.mode,
                target=existing.target,
                authorization_id=existing.authorization_id,
                backend_receipt_digest=existing.backend_receipt_digest,
                backend_receipt_json=existing.backend_receipt_json,
            )
        documents = [self._document(item, corpus_generation) for item in chunks]
        backend_receipt_digest = None
        backend_receipt_json = None
        primitive: object = None
        if self.publish:
            if authorization is None or self.receipt_store is None or self.target is None:
                raise PermissionError("live Hyphae publication is not authorized")
            stored_authorization = self.receipt_store.load_authorization(
                tenant, authorization.authorization_id or ""
            )
            if stored_authorization != authorization:
                raise PermissionError("publication authorization is absent or does not match")
            self._validate_authorization(
                authorization,
                chunks,
                corpus_generation=corpus_generation,
                idempotency_key=idempotency_key,
            )
            response = self.client.search_ingest(
                self.collection,
                {
                    "idempotency_id": _u128_idempotency(idempotency_key),
                    "documents": documents,
                },
            )
            kind = getattr(response, "kind", None)
            if kind != "search_ingested":
                raise AcquisitionError("Hyphae returned an unexpected ingest receipt")
            value = getattr(response, "value", None)
            _validate_hyphae_receipt(value, len(documents), self.target.backend_id)
            assert isinstance(value, dict)
            primitive = receipt_primitive(value)
            backend_receipt_json = canonical_json(primitive)
        receipt = IngestReceipt(
            tenant=tenant,
            source_id=first.source_id,
            source_version=first.source_version,
            corpus_generation=corpus_generation,
            chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            idempotency_key=idempotency_key,
            replayed=False,
            published=self.publish,
            mode=self.mode,
            target=self.target,
            authorization_id=None if authorization is None else authorization.authorization_id,
            backend_receipt_digest=("0" * 64 if self.publish else backend_receipt_digest),
            backend_receipt_json=backend_receipt_json,
        )
        if self.publish:
            receipt = replace(
                receipt,
                backend_receipt_digest=content_hash(
                    _backend_receipt_envelope(receipt, primitive), length=64
                ),
            )
        if self.receipt_store is not None:
            self.receipt_store.save_ingest(receipt)
        self.receipts[idempotency_key] = receipt
        return receipt

    def acquisition_receipt(
        self,
        *,
        artifact: SourceArtifact,
        chunks: tuple[EmbeddedChunk, ...],
        ingest: IngestReceipt,
        policy_version: str,
        source_url: str,
    ) -> AcquisitionReceipt:
        return AcquisitionReceipt(
            tenant=artifact.tenant,
            source_id=artifact.source_id,
            source_version=artifact.source_version,
            source_url=source_url,
            policy_version=policy_version,
            content_type=artifact.content_type,
            license_id=artifact.license_id,
            raw_bytes=len(artifact.body),
            raw_digest=artifact.content_digest,
            chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            embedding_profile=chunks[0].embedding_profile,
            ingest_idempotency_key=ingest.idempotency_key,
            published=ingest.published,
        )

    def _document(self, item: EmbeddedChunk, corpus_generation: str) -> dict[str, object]:
        chunk = item.chunk
        return {
            "object_id": _object_id(chunk.chunk_id),
            "text": chunk.text,
            "doc_values": {
                "body": chunk.text,
                "source_id": chunk.source_id,
                "source_version": chunk.source_version,
                "content_digest": chunk.content_digest,
                "corpus_generation": corpus_generation,
                "byte_start": chunk.byte_start,
                "byte_end": chunk.byte_end,
                "chunk_ordinal": chunk.ordinal,
            },
            "vectors": {self.vector_target: list(item.values)},
        }

    def verify(self, receipt: IngestReceipt, expected: tuple[EmbeddedChunk, ...]) -> bool:
        stored = (
            self.receipt_store.load_ingest(receipt.tenant, receipt.idempotency_key)
            if self.receipt_store is not None
            else self.receipts.get(receipt.idempotency_key)
        )
        expected_ids = tuple(item.chunk.chunk_id for item in expected)
        return (
            stored is not None
            and replace(stored, replayed=False) == replace(receipt, replayed=False)
            and receipt.chunk_ids == expected_ids
            and _receipt_digest_matches(receipt)
        )

    def _validate_replay(
        self,
        receipt: IngestReceipt,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        corpus_generation: str,
        authorization: PublicationAuthorization | None,
    ) -> None:
        chunk_ids = tuple(item.chunk.chunk_id for item in chunks)
        authorization_id = None if authorization is None else authorization.authorization_id
        if (
            receipt.chunk_ids != chunk_ids
            or receipt.source_id != chunks[0].chunk.source_id
            or receipt.source_version != chunks[0].chunk.source_version
            or receipt.corpus_generation != corpus_generation
            or receipt.authorization_id != authorization_id
            or receipt.target != self.target
        ):
            raise AcquisitionError("ingest idempotency key conflicts with durable receipt")

    def _validate_authorization(
        self,
        authorization: PublicationAuthorization,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        corpus_generation: str,
        idempotency_key: str,
    ) -> None:
        if authorization.tenant != self.tenant:
            raise PermissionError("publication authorization belongs to another tenant")
        first = chunks[0].chunk
        if (
            authorization.source_id != first.source_id
            or authorization.source_version != first.source_version
        ):
            raise PermissionError("publication authorization source does not match")
        if authorization.corpus_generation != corpus_generation:
            raise PermissionError("publication authorization generation does not match")
        if authorization.idempotency_key != idempotency_key:
            raise PermissionError("publication authorization idempotency does not match")
        if authorization.chunk_ids != tuple(item.chunk.chunk_id for item in chunks):
            raise PermissionError("publication authorization chunks do not match")
        if authorization.chunk_digests != tuple(
            item.chunk.content_digest for item in chunks
        ):
            raise PermissionError("publication authorization chunk digests do not match")
        if authorization.chunk_coordinates != tuple(
            (item.chunk.ordinal, item.chunk.byte_start, item.chunk.byte_end) for item in chunks
        ):
            raise PermissionError("publication authorization chunk coordinates do not match")
        if any(item.embedding_profile != authorization.embedding_profile for item in chunks):
            raise PermissionError("publication authorization embedding profile does not match")
        if authorization.target != self.target:
            raise PermissionError("publication authorization target does not match")
        if authorization.embedding_digests != tuple(embedding_digest(item) for item in chunks):
            raise PermissionError("publication authorization embeddings do not match")


class _PinnedHttpsConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        *,
        host: str,
        resolved_ip: str,
        port: int,
        timeout: float,
    ) -> None:
        context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=context)
        self._resolved_ip = resolved_ip
        self._tls_context = context

    def connect(self) -> None:
        raw = socket.create_connection(
            (self._resolved_ip, self.port),
            self.timeout,
        )
        self.sock = self._tls_context.wrap_socket(raw, server_hostname=self.host)


def _public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_global


def _object_id(chunk_id: str) -> int:
    value = int(hashlib.sha256(chunk_id.encode()).hexdigest()[:32], 16)
    return value or 1


def _u128_idempotency(digest: str) -> int:
    value = int(digest[:32], 16)
    return value or 1


def _validate_hyphae_receipt(
    value: object, expected_documents: int, expected_backend_id: str
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "snapshot",
        "documents",
        "idempotent_replay",
        "commit",
    }:
        raise AcquisitionError("Hyphae ingest receipt fields are invalid")
    documents = value["documents"]
    if (
        isinstance(documents, bool)
        or not isinstance(documents, int)
        or documents != expected_documents
        or not isinstance(value["idempotent_replay"], bool)
    ):
        raise AcquisitionError("Hyphae ingest receipt document evidence is invalid")
    snapshot = value["snapshot"]
    commit = value["commit"]
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "directory_lineage",
        "visible_csn",
        "catalog_version",
        "root_digest",
        "logical_time_micros",
    }:
        raise AcquisitionError("Hyphae ingest snapshot evidence is invalid")
    lineage = snapshot["directory_lineage"]
    root = snapshot["root_digest"]
    if not isinstance(lineage, bytes) or len(lineage) != 24 or lineage == bytes(24):
        raise AcquisitionError("Hyphae ingest directory lineage is invalid")
    if hashlib.sha256(lineage).hexdigest() != expected_backend_id:
        raise AcquisitionError("Hyphae ingest directory lineage does not match its backend")
    if not isinstance(root, bytes) or len(root) != 32 or root == bytes(32):
        raise AcquisitionError("Hyphae ingest snapshot root is invalid")
    visible_csn = snapshot["visible_csn"]
    catalog_version = snapshot["catalog_version"]
    logical_time = snapshot["logical_time_micros"]
    if (
        (
            visible_csn is not None
            and (
                isinstance(visible_csn, bool)
                or not isinstance(visible_csn, int)
                or visible_csn < 1
            )
        )
        or isinstance(catalog_version, bool)
        or not isinstance(catalog_version, int)
        or catalog_version < 1
        or isinstance(logical_time, bool)
        or not isinstance(logical_time, int)
        or logical_time < 0
    ):
        raise AcquisitionError("Hyphae ingest snapshot counters are invalid")
    if not isinstance(commit, dict) or set(commit) != {
        "transaction_id",
        "commit_csn",
        "catalog_version",
        "commit_lsn",
        "wal_block_digest",
        "durability",
        "durability_cohort_size",
        "durability_cohort_position",
    }:
        raise AcquisitionError("Hyphae ingest commit evidence is invalid")
    positive = ("transaction_id", "commit_csn", "catalog_version", "commit_lsn")
    if any(
        isinstance(commit[field], bool)
        or not isinstance(commit[field], int)
        or commit[field] < 1
        for field in positive
    ):
        raise AcquisitionError("Hyphae ingest commit counters are invalid")
    if commit["transaction_id"] >= 2**128 or any(commit[field] >= 2**64 for field in positive[1:]):
        raise AcquisitionError("Hyphae ingest commit counters exceed native bounds")
    wal_digest = commit["wal_block_digest"]
    cohort_size = commit["durability_cohort_size"]
    cohort_position = commit["durability_cohort_position"]
    if (
        commit["durability"] != "strict"
        or not isinstance(wal_digest, bytes)
        or len(wal_digest) != 32
        or wal_digest == bytes(32)
        or isinstance(cohort_size, bool)
        or not isinstance(cohort_size, int)
        or isinstance(cohort_position, bool)
        or not isinstance(cohort_position, int)
        or cohort_size < 1
        or not 0 <= cohort_position < cohort_size
    ):
        raise AcquisitionError("Hyphae ingest strict durability evidence is invalid")
    if visible_csn is None or visible_csn < commit["commit_csn"]:
        raise AcquisitionError("Hyphae ingest snapshot does not contain its commit")
    if snapshot["catalog_version"] < commit["catalog_version"]:
        raise AcquisitionError("Hyphae ingest snapshot catalog predates its commit")


def _receipt_digest_matches(receipt: IngestReceipt) -> bool:
    if receipt.mode is not IngestMode.LIVE:
        return receipt.backend_receipt_digest is None and receipt.backend_receipt_json is None
    if receipt.backend_receipt_digest is None or receipt.backend_receipt_json is None:
        return False
    try:
        primitive = json.loads(
            receipt.backend_receipt_json,
            object_pairs_hook=_unique_json_object,
        )
        if canonical_json(primitive) != receipt.backend_receipt_json:
            return False
        value = _restore_receipt_bytes(primitive)
        if receipt.target is None:
            return False
        _validate_hyphae_receipt(
            value, len(receipt.chunk_ids), receipt.target.backend_id
        )
    except (AcquisitionError, TypeError, ValueError):
        return False
    observed = content_hash(
        _backend_receipt_envelope(receipt, primitive), length=64
    )
    return observed == receipt.backend_receipt_digest


def _backend_receipt_envelope(receipt: IngestReceipt, primitive: object) -> dict[str, object]:
    return {
        "schema": "hyphae-ingest-receipt-v2",
        "tenant": receipt.tenant.value,
        "source_id": receipt.source_id,
        "source_version": receipt.source_version,
        "corpus_generation": receipt.corpus_generation,
        "target": receipt.target,
        "authorization_id": receipt.authorization_id,
        "idempotency_key": receipt.idempotency_key,
        "chunk_ids": receipt.chunk_ids,
        "value": primitive,
    }


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("backend receipt contains duplicate JSON keys")
        result[key] = value
    return result


def _restore_receipt_bytes(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {"bytes_hex"}:
            encoded = value["bytes_hex"]
            if not isinstance(encoded, str):
                raise ValueError("backend receipt byte encoding is invalid")
            return bytes.fromhex(encoded)
        return {key: _restore_receipt_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_receipt_bytes(item) for item in value]
    return value
