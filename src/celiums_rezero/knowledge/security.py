"""Production scanner adapters for ClamAV and fixed-policy DLP services."""

from __future__ import annotations

import hashlib
import socket
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from celiums_rezero.knowledge.acquisition import SecurityRejection


@dataclass(frozen=True, slots=True)
class ClamDScanner:
    socket_path: Path
    version: str
    timeout_seconds: float = 30.0
    chunk_bytes: int = 64 * 1024
    name: str = "malware"
    target: str = "raw"

    def findings(self, content: bytes) -> tuple[str, ...]:
        if self.socket_path.is_symlink() or not self.socket_path.parent.is_dir():
            raise SecurityRejection("clamd socket path is unsafe")
        if self.timeout_seconds <= 0 or not 1 <= self.chunk_bytes <= 1_048_576:
            raise SecurityRejection("clamd scanner limits are invalid")
        deadline = time.monotonic() + self.timeout_seconds
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout_seconds)
                client.connect(str(self.socket_path))
                client.sendall(b"zINSTREAM\0")
                for start in range(0, len(content), self.chunk_bytes):
                    chunk = content[start : start + self.chunk_bytes]
                    client.settimeout(max(deadline - time.monotonic(), 0.001))
                    client.sendall(struct.pack(">I", len(chunk)) + chunk)
                client.sendall(b"\0\0\0\0")
                response = _read_nul(client, deadline, 65_536)
        except (OSError, TimeoutError) as error:
            raise SecurityRejection(f"clamd unavailable: {type(error).__name__}") from error
        prefix, separator, verdict = response.rpartition(b": ")
        if not separator or not prefix:
            raise SecurityRejection("clamd returned a malformed verdict")
        if verdict == b"OK":
            return ()
        if verdict.endswith(b" FOUND"):
            return (verdict.removesuffix(b" FOUND").decode(errors="replace"),)
        raise SecurityRejection("clamd returned no clean verdict")


@dataclass(frozen=True, slots=True)
class ExternalDlpScanner:
    """Fixed adapter callable; transport must return a strict response object."""

    name: str
    target: str
    version: str
    policy_revision: str
    scan_request: Callable[[dict[str, object]], object]

    def findings(self, content: bytes) -> tuple[str, ...]:
        digest = hashlib.sha256(content).hexdigest()
        try:
            response = self.scan_request(
                {
                    "schema": "celiums-dlp-request-v1",
                    "content_hex": content.hex(),
                    "content_digest": digest,
                    "policy_revision": self.policy_revision,
                    "controls": [self.name],
                }
            )
        except Exception as error:
            raise SecurityRejection(f"DLP unavailable: {type(error).__name__}") from error
        fields = {"schema", "content_digest", "policy_revision", "findings"}
        if not isinstance(response, dict) or set(response) != fields:
            raise SecurityRejection("DLP returned a malformed response")
        findings = response["findings"]
        if (
            response["schema"] != "celiums-dlp-response-v1"
            or response["content_digest"] != digest
            or response["policy_revision"] != self.policy_revision
            or not isinstance(findings, list)
            or any(not isinstance(item, str) or not item for item in findings)
        ):
            raise SecurityRejection("DLP response binding is invalid")
        return tuple(findings)


def _read_nul(client: socket.socket, deadline: float, maximum: int) -> bytes:
    response = bytearray()
    while len(response) <= maximum:
        client.settimeout(max(deadline - time.monotonic(), 0.001))
        chunk = client.recv(min(4096, maximum + 1 - len(response)))
        if not chunk:
            raise SecurityRejection("scanner response closed before terminator")
        response.extend(chunk)
        if b"\0" in chunk:
            value, _, trailing = bytes(response).partition(b"\0")
            if trailing:
                raise SecurityRejection("scanner response contains trailing bytes")
            return value
    raise SecurityRejection("scanner response exceeds its byte bound")
