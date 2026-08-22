"""Phase 3 shadow-mode HTTPS acquisition and bounded Hyphae ingestion adapters."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from celiums_rezero.knowledge.acquisition import AcquisitionError
from celiums_rezero.knowledge.schemas import (
    AcquisitionReceipt,
    EmbeddedChunk,
    IngestReceipt,
    SourceArtifact,
    SourcePolicy,
    TenantId,
)


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
                    "User-Agent": "celiums-rezero-knowledge/0.1",
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
        publish: bool = False,
        max_batch_documents: int = 256,
    ) -> None:
        if collection < 1 or not vector_target or not 1 <= max_batch_documents <= 256:
            raise ValueError("Hyphae ingestion configuration is invalid")
        self.tenant = tenant
        self.client = client
        self.collection = collection
        self.vector_target = vector_target
        self.publish = publish
        self.max_batch_documents = max_batch_documents
        self.receipts: dict[str, IngestReceipt] = {}

    def ingest(
        self,
        tenant: TenantId,
        chunks: tuple[EmbeddedChunk, ...],
        *,
        corpus_generation: str,
        idempotency_key: str,
    ) -> IngestReceipt:
        if tenant != self.tenant:
            raise PermissionError("Hyphae ingestor is bound to another tenant")
        existing = self.receipts.get(idempotency_key)
        if existing is not None:
            return IngestReceipt(
                tenant=existing.tenant,
                source_id=existing.source_id,
                source_version=existing.source_version,
                corpus_generation=existing.corpus_generation,
                chunk_ids=existing.chunk_ids,
                idempotency_key=existing.idempotency_key,
                replayed=True,
                published=existing.published,
            )
        if not chunks or len(chunks) > self.max_batch_documents:
            raise AcquisitionError("Hyphae ingest batch exceeds its document bound")
        first = chunks[0].chunk
        documents = [self._document(item, corpus_generation) for item in chunks]
        if self.publish:
            response = self.client.search_ingest(
                self.collection,
                {
                    "idempotency_id": _u128_idempotency(idempotency_key),
                    "documents": documents,
                },
            )
            kind = getattr(response, "kind", None)
            if kind not in {"search_ingested", "search_ingest"}:
                raise AcquisitionError("Hyphae returned an unexpected ingest receipt")
        receipt = IngestReceipt(
            tenant=tenant,
            source_id=first.source_id,
            source_version=first.source_version,
            corpus_generation=corpus_generation,
            chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            idempotency_key=idempotency_key,
            replayed=False,
            published=self.publish,
        )
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
        stored = self.receipts.get(receipt.idempotency_key)
        expected_ids = tuple(item.chunk.chunk_id for item in expected)
        return stored == receipt and receipt.chunk_ids == expected_ids


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
