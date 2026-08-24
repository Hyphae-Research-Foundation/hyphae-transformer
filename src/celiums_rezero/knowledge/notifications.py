"""Fixed HTTPS notification sink with idempotency and strict acknowledgements."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import sqlite3
import ssl
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from celiums_rezero.knowledge.finalization import (
    PermanentFinalizationError,
    TransientFinalizationError,
)
from celiums_rezero.knowledge.schemas import NotificationReceipt, PreparedNotification
from celiums_rezero.knowledge.store import encode_prepared_notification
from celiums_rezero.lab.serialization import canonical_json

_MAILBOX_SCHEMA = """
CREATE TABLE mailbox_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    tenant_id TEXT NOT NULL,
    sink_id TEXT NOT NULL
);
CREATE TABLE notifications (
    notification_id TEXT PRIMARY KEY,
    command_digest TEXT NOT NULL,
    command_json TEXT NOT NULL,
    provider_receipt TEXT NOT NULL,
    accepted_at_us INTEGER NOT NULL
);
"""


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


@dataclass(frozen=True, slots=True)
class SQLiteMailboxConfig:
    tenant_id: str
    path: Path
    mailbox_id: str
    timeout_seconds: float = 5.0

    @property
    def sink_id(self) -> str:
        identity = canonical_json(
            {
                "schema": "hyphae-sqlite-mailbox/v1",
                "tenant_id": self.tenant_id,
                "mailbox_id": self.mailbox_id,
            }
        )
        return f"sqlite_{hashlib.sha256(identity.encode()).hexdigest()[:32]}"

    def __post_init__(self) -> None:
        if (
            not self.tenant_id
            or not self.mailbox_id
            or not self.path.is_absolute()
            or self.timeout_seconds <= 0
        ):
            raise ValueError("SQLite mailbox configuration is invalid")


class SQLiteMailboxNotificationSink:
    """Durably accepts each logical notification once into a local mailbox."""

    def __init__(self, config: SQLiteMailboxConfig) -> None:
        self.config = config
        self.path = _prepare_mailbox_path(config.path)
        self._lock = Lock()
        created = self.path.stat().st_size == 0
        with self._connect(initializing=created) as connection:
            if created:
                connection.executescript(_MAILBOX_SCHEMA)
                connection.execute(
                    "INSERT INTO mailbox_meta VALUES (1, 1, ?, ?)",
                    (config.tenant_id, config.sink_id),
                )
                connection.commit()
                if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                    raise PermanentFinalizationError("SQLite mailbox could not enable WAL")
            self._verify(connection)

    @property
    def sink_id(self) -> str:
        return self.config.sink_id

    def deliver(
        self, command: PreparedNotification, *, timeout_seconds: float
    ) -> NotificationReceipt:
        if timeout_seconds <= 0:
            raise ValueError("SQLite mailbox timeout must be positive")
        if (
            command.tenant.value != self.config.tenant_id
            or command.sink_id != self.sink_id
            or command.notification_id is None
            or command.command_digest is None
        ):
            raise PermanentFinalizationError("SQLite mailbox command binding is invalid")
        payload = encode_prepared_notification(command)
        provider_receipt = "mailbox_" + hashlib.sha256(
            f"{self.sink_id}\0{command.notification_id}\0{command.command_digest}".encode()
        ).hexdigest()
        try:
            with self._lock, self._connect(timeout_seconds=timeout_seconds) as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT command_digest, command_json, provider_receipt "
                    "FROM notifications WHERE notification_id = ?",
                    (command.notification_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO notifications VALUES (?, ?, ?, ?, ?)",
                        (
                            command.notification_id,
                            command.command_digest,
                            payload,
                            provider_receipt,
                            time.time_ns() // 1000,
                        ),
                    )
                elif tuple(existing) != (
                    command.command_digest,
                    payload,
                    provider_receipt,
                ):
                    connection.execute("ROLLBACK")
                    raise PermanentFinalizationError("SQLite mailbox replay differs")
                connection.execute("COMMIT")
        except sqlite3.OperationalError as error:
            raise TransientFinalizationError("SQLite mailbox is temporarily unavailable") from error
        return NotificationReceipt(
            tenant=command.tenant,
            job_id=command.job_id,
            notification_id=command.notification_id,
            sink_id=command.sink_id,
            command_digest=command.command_digest,
            provider_receipt=provider_receipt,
        )

    def accepted_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM notifications").fetchone()[0])

    def _connect(
        self,
        *,
        initializing: bool = False,
        timeout_seconds: float | None = None,
    ) -> sqlite3.Connection:
        _validate_mailbox_file(self.path)
        connection = sqlite3.connect(
            self.path,
            timeout=self.config.timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA trusted_schema=OFF")
        if (
            not initializing
            and connection.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
        ):
            connection.close()
            raise PermanentFinalizationError("SQLite mailbox WAL durability changed")
        return connection

    def _verify(self, connection: sqlite3.Connection) -> None:
        meta = connection.execute(
            "SELECT schema_version, tenant_id, sink_id FROM mailbox_meta WHERE singleton = 1"
        ).fetchone()
        objects = {
            tuple(row)
            for row in connection.execute(
                "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            )
        }
        if meta is None or tuple(meta) != (1, self.config.tenant_id, self.sink_id):
            raise PermanentFinalizationError("SQLite mailbox identity changed")
        if objects != {("table", "mailbox_meta"), ("table", "notifications")}:
            raise PermanentFinalizationError("SQLite mailbox schema objects changed")


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


def _prepare_mailbox_path(path: Path) -> Path:
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError("SQLite mailbox path contains traversal")
    parent = path.parent.resolve(strict=True)
    if path.parent != parent or stat.S_IMODE(parent.stat().st_mode) & 0o077:
        raise PermissionError("SQLite mailbox directory is unsafe")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.close(descriptor)
    resolved = parent / path.name
    _validate_mailbox_file(resolved)
    return resolved


def _validate_mailbox_file(path: Path) -> None:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError("SQLite mailbox file is unsafe")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise PermissionError("SQLite mailbox file has another owner")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if os.path.lexists(sidecar):
            sidecar_metadata = sidecar.lstat()
            if (
                not stat.S_ISREG(sidecar_metadata.st_mode)
                or stat.S_IMODE(sidecar_metadata.st_mode) & 0o077
            ):
                raise PermissionError("SQLite mailbox sidecar is unsafe")
