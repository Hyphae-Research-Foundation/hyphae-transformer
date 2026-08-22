"""Immutable durable publication authorization and ingest receipt storage."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EmbeddedChunk,
    IngestMode,
    IngestReceipt,
    PublicationAuthorization,
    PublicationTarget,
    SecurityScanReceipt,
    TenantId,
    ValidatedArtifact,
)
from celiums_rezero.lab.serialization import canonical_json, to_primitive

_MAX_RECEIPT_BYTES = 4_000_000


class PublicationReceiptStore:
    """Tenant-local write-once receipt store with atomic durable publication."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("publication receipt root must be a real directory")
        os.chmod(self.root, 0o700)
        self.root = self.root.resolve(strict=True)

    def save_authorization(self, authorization: PublicationAuthorization) -> Path:
        assert authorization.authorization_id is not None
        path = self._tenant_directory(authorization.tenant) / "authorizations"
        return self._write_once(
            path / f"{authorization.authorization_id}.json",
            {"schema": "knowledge-publication-authorization-v1", "value": authorization},
        )

    def load_authorization(
        self, tenant: TenantId, authorization_id: str
    ) -> PublicationAuthorization | None:
        if not _identifier(authorization_id, "authorization_"):
            raise ValueError("publication authorization ID is invalid")
        path = self._tenant_directory(tenant) / "authorizations" / f"{authorization_id}.json"
        if not path.exists():
            return None
        record = self._read_record(path, "knowledge-publication-authorization-v1")
        authorization = _authorization(record)
        if authorization.tenant != tenant or authorization.authorization_id != authorization_id:
            raise ValueError("publication authorization path binding is invalid")
        return authorization

    def save_ingest(self, receipt: IngestReceipt) -> Path:
        path = self._tenant_directory(receipt.tenant) / "ingest"
        return self._write_once(
            path / f"{receipt.idempotency_key}.json",
            {"schema": "knowledge-ingest-receipt-v1", "value": receipt},
        )

    def load_ingest(self, tenant: TenantId, idempotency_key: str) -> IngestReceipt | None:
        if not _digest(idempotency_key):
            raise ValueError("ingest idempotency key is invalid")
        path = self._tenant_directory(tenant) / "ingest" / f"{idempotency_key}.json"
        if not path.exists():
            return None
        record = self._read_record(path, "knowledge-ingest-receipt-v1")
        receipt = _ingest_receipt(record)
        if receipt.tenant != tenant or receipt.idempotency_key != idempotency_key:
            raise ValueError("ingest receipt path binding is invalid")
        return receipt

    def _tenant_directory(self, tenant: TenantId) -> Path:
        path = self.root / tenant.value
        try:
            path.mkdir(mode=0o700)
            _fsync_directory(self.root)
        except FileExistsError:
            pass
        if path.is_symlink() or not path.is_dir() or path.parent.resolve(strict=True) != self.root:
            raise ValueError("tenant receipt directory is unsafe")
        os.chmod(path, 0o700)
        return path

    def _write_once(self, path: Path, value: object) -> Path:
        _mkdir_durable(path.parent.parent, path.parent)
        if (
            path.parent.is_symlink()
            or not path.parent.is_dir()
            or path.parent.parent.is_symlink()
        ):
            raise ValueError("receipt directory cannot be a symlink")
        os.chmod(path.parent, 0o700)
        payload = canonical_json(value) + "\n"
        if len(payload.encode()) > _MAX_RECEIPT_BYTES:
            raise ValueError("publication receipt exceeds its byte bound")
        if path.exists():
            if canonical_json(self._read_json(path)) != canonical_json(value):
                raise FileExistsError(f"immutable publication receipt differs: {path}")
            return path
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="ascii") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if canonical_json(self._read_json(path)) != canonical_json(value):
                raise FileExistsError(f"immutable publication receipt differs: {path}") from None
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _read_record(self, path: Path, schema: str) -> dict[str, object]:
        record = self._read_json(path)
        if set(record) != {"schema", "value"} or record["schema"] != schema:
            raise ValueError("publication receipt schema is invalid")
        value = record["value"]
        if not isinstance(value, dict):
            raise ValueError("publication receipt value must be an object")
        return cast(dict[str, object], value)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("publication receipt file is invalid")
        with path.open("r", encoding="ascii") as source:
            parsed = json.load(source, object_pairs_hook=_unique_object)
        if not isinstance(parsed, dict):
            raise ValueError("publication receipt must be an object")
        return cast(dict[str, object], parsed)


@dataclass(frozen=True, slots=True)
class DurablePublicationAuthorizer:
    tenant: TenantId
    store: PublicationReceiptStore
    authority: str
    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("publication enablement must be a boolean")
        if not self.authority:
            raise ValueError("publication authority is required")

    def authorize(
        self,
        *,
        job: AcquisitionJob,
        validated: ValidatedArtifact,
        chunks: tuple[EmbeddedChunk, ...],
        idempotency_key: str,
        target: PublicationTarget,
    ) -> PublicationAuthorization:
        if not self.enabled:
            raise PermissionError("live publication authorization is disabled")
        if job.tenant != self.tenant or validated.artifact.tenant != self.tenant:
            raise PermissionError("publication authorizer is bound to another tenant")
        if (
            job.source_id != validated.artifact.source_id
            or job.embedding_profile == ""
            or idempotency_key == ""
        ):
            raise ValueError("publication artifact does not match its acquisition job")
        if not chunks or any(item.embedding_profile != job.embedding_profile for item in chunks):
            raise ValueError("publication chunks do not match the job embedding profile")
        _validate_chunks(validated, chunks)
        from celiums_rezero.knowledge.acquisition import validate_embedded_chunks

        validate_embedded_chunks(
            job,
            chunks,
            job.embedding_profile,
            idempotency_key,
            target=target,
        )
        authorization = PublicationAuthorization(
            tenant=self.tenant,
            source_id=validated.artifact.source_id,
            source_version=validated.artifact.source_version,
            corpus_generation=job.corpus_generation,
            policy_version=job.policy_version,
            raw_digest=validated.artifact.content_digest,
            parsed_digest=validated.content_digest,
            parser=validated.parser,
            parser_version=validated.parser_version,
            scans=validated.scans,
            chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            chunk_digests=tuple(item.chunk.content_digest for item in chunks),
            chunk_coordinates=tuple(
                (item.chunk.ordinal, item.chunk.byte_start, item.chunk.byte_end)
                for item in chunks
            ),
            embedding_profile=job.embedding_profile,
            idempotency_key=idempotency_key,
            authority=self.authority,
            target=target,
            embedding_digests=tuple(embedding_digest(item) for item in chunks),
        )
        self.store.save_authorization(authorization)
        return authorization


def _validate_chunks(
    validated: ValidatedArtifact, chunks: tuple[EmbeddedChunk, ...]
) -> None:
    body = validated.body
    previous_start = -1
    previous_end = -1
    for ordinal, item in enumerate(chunks):
        chunk = item.chunk
        if (
            chunk.ordinal != ordinal
            or chunk.byte_start <= previous_start
            or chunk.byte_end <= previous_end
            or chunk.byte_end > len(body)
            or body[chunk.byte_start : chunk.byte_end] != chunk.text.encode()
        ):
            raise ValueError("publication chunks are not bound to the parsed artifact")
        expected_id = hashlib.sha256(
            b"knowledge-chunk-v1\0"
            + validated.content_digest.encode()
            + chunk.byte_start.to_bytes(8, "big")
            + chunk.byte_end.to_bytes(8, "big")
        ).hexdigest()[:16]
        if chunk.chunk_id != f"chunk_{expected_id}":
            raise ValueError("publication chunk identity does not match the parsed artifact")
        previous_start = chunk.byte_start
        previous_end = chunk.byte_end


def embedding_digest(item: EmbeddedChunk) -> str:
    payload = {
        "schema": "knowledge-embedding-v1",
        "profile": item.embedding_profile,
        "values": [float(value).hex() for value in item.values],
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(parent: Path, path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
        _fsync_directory(parent)
    except FileExistsError:
        pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("publication receipt contains duplicate JSON keys")
        value[key] = item
    return value


def _identifier(value: str, prefix: str) -> bool:
    suffix = value.removeprefix(prefix)
    return value.startswith(prefix) and len(suffix) == 64 and all(
        character in "0123456789abcdef" for character in suffix
    )


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact(value: dict[str, object], fields: set[str]) -> None:
    if set(value) != fields:
        raise ValueError("publication receipt fields are invalid")


def _scan(value: object) -> SecurityScanReceipt:
    if not isinstance(value, dict):
        raise ValueError("security scan receipt is invalid")
    typed = cast(dict[str, object], value)
    _exact(typed, {"scanner", "version", "target", "content_digest", "findings"})
    findings = typed["findings"]
    if isinstance(findings, bool) or not isinstance(findings, int):
        raise ValueError("security scan finding count is invalid")
    return SecurityScanReceipt(
        scanner=_string(typed["scanner"], "security scan scanner"),
        version=_string(typed["version"], "security scan version"),
        target=_string(typed["target"], "security scan target"),
        content_digest=_string(typed["content_digest"], "security scan digest"),
        findings=findings,
    )


def _target(value: object) -> PublicationTarget:
    if not isinstance(value, dict):
        raise ValueError("publication target is invalid")
    typed = cast(dict[str, object], value)
    _exact(typed, {"backend_id", "collection", "vector_target"})
    collection = typed["collection"]
    if isinstance(collection, bool) or not isinstance(collection, int):
        raise ValueError("publication target collection is invalid")
    return PublicationTarget(
        backend_id=_string(typed["backend_id"], "publication backend ID"),
        collection=collection,
        vector_target=_string(typed["vector_target"], "publication vector target"),
    )


def _authorization(value: dict[str, object]) -> PublicationAuthorization:
    fields = {
        "tenant",
        "source_id",
        "source_version",
        "corpus_generation",
        "policy_version",
        "raw_digest",
        "parsed_digest",
        "parser",
        "parser_version",
        "scans",
        "chunk_ids",
        "chunk_digests",
        "chunk_coordinates",
        "embedding_profile",
        "idempotency_key",
        "authority",
        "target",
        "embedding_digests",
        "authorization_id",
    }
    _exact(value, fields)
    tenant = value["tenant"]
    scans = value["scans"]
    chunk_ids = value["chunk_ids"]
    chunk_digests = value["chunk_digests"]
    chunk_coordinates = value["chunk_coordinates"]
    embedding_digests = value["embedding_digests"]
    if (
        not isinstance(tenant, dict)
        or set(tenant) != {"value"}
        or not isinstance(scans, list)
        or not isinstance(chunk_ids, list)
        or not isinstance(chunk_digests, list)
        or not isinstance(chunk_coordinates, list)
        or not isinstance(embedding_digests, list)
    ):
        raise ValueError("publication authorization tenant or scans are invalid")
    return PublicationAuthorization(
        tenant=TenantId(_string(tenant["value"], "tenant ID")),
        source_id=_string(value["source_id"], "source ID"),
        source_version=_string(value["source_version"], "source version"),
        corpus_generation=_string(value["corpus_generation"], "corpus generation"),
        policy_version=_string(value["policy_version"], "policy version"),
        raw_digest=_string(value["raw_digest"], "raw digest"),
        parsed_digest=_string(value["parsed_digest"], "parsed digest"),
        parser=_string(value["parser"], "parser"),
        parser_version=_string(value["parser_version"], "parser version"),
        scans=tuple(_scan(item) for item in scans),
        chunk_ids=tuple(_string(item, "chunk ID") for item in chunk_ids),
        chunk_digests=tuple(_string(item, "chunk digest") for item in chunk_digests),
        chunk_coordinates=tuple(_coordinate(item) for item in chunk_coordinates),
        embedding_profile=_string(value["embedding_profile"], "embedding profile"),
        idempotency_key=_string(value["idempotency_key"], "idempotency key"),
        authority=_string(value["authority"], "authority"),
        target=_target(value["target"]),
        embedding_digests=tuple(
            _string(item, "embedding digest") for item in embedding_digests
        ),
        authorization_id=_string(value["authorization_id"], "authorization ID"),
    )


def _ingest_receipt(value: dict[str, object]) -> IngestReceipt:
    fields = {
        "tenant",
        "source_id",
        "source_version",
        "corpus_generation",
        "chunk_ids",
        "idempotency_key",
        "replayed",
        "published",
        "mode",
        "target",
        "authorization_id",
        "backend_receipt_digest",
        "backend_receipt_json",
    }
    _exact(value, fields)
    tenant = value["tenant"]
    chunk_ids = value["chunk_ids"]
    if not isinstance(tenant, dict) or set(tenant) != {"value"} or not isinstance(chunk_ids, list):
        raise ValueError("ingest receipt tenant or chunks are invalid")
    authorization_id = value["authorization_id"]
    backend_digest = value["backend_receipt_digest"]
    backend_json = value["backend_receipt_json"]
    replayed = value["replayed"]
    published = value["published"]
    if not isinstance(replayed, bool) or not isinstance(published, bool):
        raise ValueError("ingest receipt flags must be booleans")
    try:
        mode = IngestMode(_string(value["mode"], "ingest mode"))
    except ValueError as error:
        raise ValueError("ingest receipt mode is invalid") from error
    return IngestReceipt(
        tenant=TenantId(_string(tenant["value"], "tenant ID")),
        source_id=_string(value["source_id"], "source ID"),
        source_version=_string(value["source_version"], "source version"),
        corpus_generation=_string(value["corpus_generation"], "corpus generation"),
        chunk_ids=tuple(_string(item, "chunk ID") for item in chunk_ids),
        idempotency_key=_string(value["idempotency_key"], "idempotency key"),
        replayed=replayed,
        published=published,
        mode=mode,
        target=None if value["target"] is None else _target(value["target"]),
        authorization_id=(
            None if authorization_id is None else _string(authorization_id, "authorization ID")
        ),
        backend_receipt_digest=(
            None if backend_digest is None else _string(backend_digest, "backend digest")
        ),
        backend_receipt_json=(
            None if backend_json is None else _string(backend_json, "backend receipt")
        ),
    )


def _coordinate(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("chunk coordinate is invalid")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("chunk coordinate values are invalid")
    return cast(tuple[int, int, int], tuple(value))


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def receipt_primitive(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): receipt_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [receipt_primitive(item) for item in value]
    return to_primitive(value)
