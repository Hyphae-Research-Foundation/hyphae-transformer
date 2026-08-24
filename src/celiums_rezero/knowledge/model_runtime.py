"""Supervised frozen-runtime protocol for host-validated quoted claims."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from celiums_rezero.knowledge.finalization import (
    FinalizationTimeout,
    PermanentFinalizationError,
    TransientFinalizationError,
)
from celiums_rezero.knowledge.orchestration import (
    FrozenModelIdentity,
    GovernedModelRequest,
    GovernedModelResult,
    QuotedClaim,
)
from celiums_rezero.knowledge.schemas import EvidenceHit
from celiums_rezero.knowledge.supervisor import run_supervised
from celiums_rezero.lab.serialization import canonical_json

REQUEST_SCHEMA = "hyphae-frozen-runtime-request/v1"
RESPONSE_SCHEMA = "hyphae-frozen-runtime-response/v1"


@dataclass(frozen=True, slots=True)
class SupervisedFrozenRuntimeConfig:
    executable: Path
    identity: FrozenModelIdentity
    executable_sha256: str
    arguments: tuple[str, ...] = ()
    maximum_passages: int = 8
    maximum_evidence_bytes: int = 65_536
    maximum_request_bytes: int = 100_000
    maximum_response_bytes: int = 32_768
    termination_grace_seconds: float = 0.1

    def __post_init__(self) -> None:
        _validate_executable(self.executable, self.executable_sha256)
        if (
            not 1 <= self.maximum_passages <= 64
            or not 1 <= self.maximum_evidence_bytes <= 1_000_000
            or not self.maximum_evidence_bytes <= self.maximum_request_bytes <= 1_000_000
            or not 1 <= self.maximum_response_bytes <= 1_000_000
            or not isfinite(self.termination_grace_seconds)
            or self.termination_grace_seconds <= 0
        ):
            raise ValueError("frozen runtime bounds are invalid")
        if any(not value for value in asdict(self.identity).values()):
            raise ValueError("frozen runtime identity is incomplete")
        if any(not value or "\x00" in value for value in self.arguments):
            raise ValueError("frozen runtime arguments are invalid")


@dataclass(frozen=True, slots=True)
class FrozenRuntimeExchange:
    request_id: str
    request_payload: bytes
    response_payload: bytes
    result: GovernedModelResult


class SupervisedFrozenGemmaRuntime:
    """Executes one pinned local runtime process under a hard host deadline."""

    def __init__(self, config: SupervisedFrozenRuntimeConfig) -> None:
        self.config = config

    @property
    def identity(self) -> FrozenModelIdentity:
        return self.config.identity

    def infer(
        self, request: GovernedModelRequest, *, timeout_seconds: float
    ) -> GovernedModelResult:
        return self.infer_exchange(request, timeout_seconds=timeout_seconds).result

    def infer_exchange(
        self, request: GovernedModelRequest, *, timeout_seconds: float
    ) -> FrozenRuntimeExchange:
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("frozen runtime timeout must be finite and positive")
        _validate_executable(self.config.executable, self.config.executable_sha256)
        payload, request_id = _encode_request(request, self.config)
        result = run_supervised(
            (str(self.config.executable), *self.config.arguments),
            timeout_seconds=timeout_seconds,
            grace_seconds=self.config.termination_grace_seconds,
            maximum_output_bytes=self.config.maximum_response_bytes,
            input_bytes=payload,
        )
        if result.timed_out:
            raise FinalizationTimeout("frozen runtime deadline exceeded")
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:4096].strip()
            raise TransientFinalizationError(
                f"frozen runtime exited with status {result.returncode}"
                + (f": {stderr}" if stderr else "")
            )
        decoded = _decode_response(
            result.stdout,
            request=request,
            request_id=request_id,
            expected_identity=self.identity,
        )
        return FrozenRuntimeExchange(request_id, payload, result.stdout, decoded)


def _encode_request(
    request: GovernedModelRequest,
    config: SupervisedFrozenRuntimeConfig,
) -> tuple[bytes, str]:
    if not request.query.strip() or not request.generation_id:
        raise ValueError("frozen runtime request identity is incomplete")
    if not 1 <= request.maximum_output_bytes <= 16_384:
        raise ValueError("frozen runtime output bound is invalid")
    if not request.passages or len(request.passages) > config.maximum_passages:
        raise ValueError("frozen runtime passage count is invalid")
    if sum(len(item.text.encode()) for item in request.passages) > (
        config.maximum_evidence_bytes
    ):
        raise ValueError("frozen runtime evidence exceeds its byte bound")
    _validate_passages(request.passages)
    value = {
        "schema": REQUEST_SCHEMA,
        "identity": config.identity,
        "query": request.query,
        "generation_id": request.generation_id,
        "maximum_output_bytes": request.maximum_output_bytes,
        "passages": request.passages,
    }
    request_id = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    payload = (canonical_json({**value, "request_id": request_id}) + "\n").encode()
    if len(payload) > config.maximum_request_bytes:
        raise ValueError("frozen runtime request exceeds its byte bound")
    return payload, request_id


def _decode_response(
    payload: bytes,
    *,
    request: GovernedModelRequest,
    request_id: str,
    expected_identity: FrozenModelIdentity,
) -> GovernedModelResult:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermanentFinalizationError("frozen runtime response is malformed") from error
    fields = {"schema", "request_id", "identity", "decision", "claims"}
    if not isinstance(value, dict) or set(value) != fields:
        raise PermanentFinalizationError("frozen runtime response fields are invalid")
    if value["schema"] != RESPONSE_SCHEMA or value["request_id"] != request_id:
        raise PermanentFinalizationError("frozen runtime response binding is invalid")
    identity = _identity(value["identity"])
    if identity != expected_identity:
        raise PermanentFinalizationError("frozen runtime response identity drifted")
    decision = value["decision"]
    claims_value = value["claims"]
    if decision not in {"answer", "insufficient"} or not isinstance(claims_value, list):
        raise PermanentFinalizationError("frozen runtime decision is invalid")
    claims = tuple(_claim(item) for item in claims_value)
    if decision == "insufficient" and claims:
        raise PermanentFinalizationError("insufficient runtime response carried claims")
    if decision == "answer" and not claims:
        raise PermanentFinalizationError("answer runtime response has no claims")
    by_handle = {item.handle: item for item in request.passages}
    if len({item.handle for item in claims}) != len(claims) or any(
        item.handle not in by_handle
        or not item.quote
        or item.quote not in by_handle[item.handle].text
        for item in claims
    ):
        raise PermanentFinalizationError("frozen runtime claim is not a supplied quotation")
    answer = "\n\n".join(item.quote for item in claims)
    if len(answer.encode()) > request.maximum_output_bytes:
        raise PermanentFinalizationError("frozen runtime answer exceeds its byte bound")
    return GovernedModelResult(identity, cast(str, decision), claims)


def _validate_passages(passages: tuple[EvidenceHit, ...]) -> None:
    handles = [item.handle for item in passages]
    if len(handles) != len(set(handles)) or any(
        not item.active
        or not item.trusted
        or item.content_digest != hashlib.sha256(item.text.encode()).hexdigest()
        for item in passages
    ):
        raise ValueError("frozen runtime passage binding is invalid")


def _identity(value: object) -> FrozenModelIdentity:
    fields = {"model_id", "revision", "manifest_digest", "runtime_version"}
    if not isinstance(value, dict) or set(value) != fields or any(
        not isinstance(value[field], str) or not value[field] for field in fields
    ):
        raise PermanentFinalizationError("frozen runtime identity is invalid")
    return FrozenModelIdentity(
        cast(str, value["model_id"]),
        cast(str, value["revision"]),
        cast(str, value["manifest_digest"]),
        cast(str, value["runtime_version"]),
    )


def _claim(value: object) -> QuotedClaim:
    if (
        not isinstance(value, dict)
        or set(value) != {"handle", "quote"}
        or not isinstance(value["handle"], str)
        or not isinstance(value["quote"], str)
    ):
        raise PermanentFinalizationError("frozen runtime claim fields are invalid")
    return QuotedClaim(value["handle"], value["quote"])


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PermanentFinalizationError("frozen runtime response has duplicate keys")
        value[key] = item
    return value


def _validate_executable(path: Path, expected_sha256: str) -> None:
    if not path.is_absolute() or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("frozen runtime executable identity is invalid")
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(path, os.X_OK)
    ):
        raise PermissionError("frozen runtime executable is unsafe")
    if hasattr(os, "geteuid") and metadata.st_uid not in {0, os.geteuid()}:
        raise PermissionError("frozen runtime executable has another owner")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise PermissionError("frozen runtime executable digest changed")
