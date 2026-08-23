"""Fixed HTTPS notification sink with idempotency and strict acknowledgements."""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from celiums_rezero.knowledge.finalization import (
    PermanentFinalizationError,
    TransientFinalizationError,
)
from celiums_rezero.knowledge.schemas import NotificationReceipt, PreparedNotification
from celiums_rezero.lab.serialization import canonical_json


@dataclass(frozen=True, slots=True)
class HttpsNotificationConfig:
    sink_id: str
    endpoint: str
    bearer_token: str | None = None
    maximum_response_bytes: int = 65_536

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
            or parsed.port not in {None, 443}
        ):
            raise ValueError("notification endpoint must be fixed HTTPS")
        if not self.sink_id or not 1 <= self.maximum_response_bytes <= 1_000_000:
            raise ValueError("notification sink configuration is invalid")
        expected = f"https_{hashlib.sha256(self.endpoint.encode()).hexdigest()[:32]}"
        if self.sink_id != expected:
            raise ValueError("notification sink ID must bind the fixed endpoint")


class HttpsNotificationSink:
    def __init__(self, config: HttpsNotificationConfig) -> None:
        self.config = config

    @property
    def sink_id(self) -> str:
        return self.config.sink_id

    def deliver(
        self, command: PreparedNotification, *, timeout_seconds: float
    ) -> NotificationReceipt:
        parsed = urlparse(self.config.endpoint)
        body = canonical_json(
            {"schema": "knowledge-prepared-notification-v1", "value": command}
        ).encode()
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        connection = http.client.HTTPSConnection(
            parsed.hostname or "", parsed.port or 443, timeout=timeout_seconds, context=context
        )
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Idempotency-Key": command.notification_id or "",
            "X-Celiums-Command-Digest": command.command_digest or "",
            "Connection": "close",
        }
        if self.config.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        try:
            deadline = time.monotonic() + timeout_seconds
            connection.connect()
            _set_remaining_timeout(connection, deadline)
            connection.request("POST", path, body=body, headers=headers)
            _set_remaining_timeout(connection, deadline)
            response = connection.getresponse()
            payload = bytearray()
            while len(payload) <= self.config.maximum_response_bytes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("notification total deadline exceeded")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read(
                    min(4096, self.config.maximum_response_bytes + 1 - len(payload))
                )
                if not chunk:
                    break
                payload.extend(chunk)
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            raise TransientFinalizationError(
                f"notification transport failed: {type(error).__name__}"
            ) from error
        finally:
            connection.close()
        if len(payload) > self.config.maximum_response_bytes:
            raise PermanentFinalizationError("notification acknowledgement is oversized")
        if response.status in {408, 425, 429, 500, 502, 503, 504}:
            raise TransientFinalizationError(f"notification returned HTTP {response.status}")
        if response.status not in {200, 201}:
            raise PermanentFinalizationError(f"notification returned HTTP {response.status}")
        try:
            value = json.loads(bytes(payload), object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PermanentFinalizationError("notification acknowledgement is malformed") from error
        fields = {"schema", "notification_id", "command_digest", "provider_receipt"}
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or value["schema"] != "celiums-notification-ack-v1"
            or value["notification_id"] != command.notification_id
            or value["command_digest"] != command.command_digest
            or not isinstance(value["provider_receipt"], str)
        ):
            raise PermanentFinalizationError("notification acknowledgement binding is invalid")
        return NotificationReceipt(
            tenant=command.tenant,
            job_id=command.job_id,
            notification_id=command.notification_id or "",
            sink_id=command.sink_id,
            command_digest=command.command_digest or "",
            provider_receipt=value["provider_receipt"],
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PermanentFinalizationError(
                "notification acknowledgement contains duplicate keys"
            )
        value[key] = item
    return value


def _set_remaining_timeout(
    connection: http.client.HTTPSConnection, deadline: float
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("notification total deadline exceeded")
    if connection.sock is None:
        raise OSError("notification TLS connection has no socket")
    connection.sock.settimeout(remaining)
