"""Bounded parsing and deterministic fail-closed source security scans."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

from celiums_rezero.knowledge.acquisition import SecurityRejection
from celiums_rezero.knowledge.schemas import (
    SecurityScanReceipt,
    SourceArtifact,
    ValidatedArtifact,
)


class ArtifactParser(Protocol):
    name: str
    version: str

    def parse(self, artifact: SourceArtifact) -> bytes: ...


class ContentScanner(Protocol):
    name: str
    version: str
    target: str

    def findings(self, content: bytes) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class BoundedTextParser:
    """Strict UTF-8 text parser; rich and executable formats remain unsupported."""

    max_output_bytes: int = 20_000_000
    name: str = "bounded-text"
    version: str = "1"

    def __post_init__(self) -> None:
        if self.max_output_bytes < 1:
            raise ValueError("parser output bound must be positive")

    def parse(self, artifact: SourceArtifact) -> bytes:
        if artifact.content_type != "text/plain":
            raise SecurityRejection(
                f"no sandboxed parser is configured for {artifact.content_type}"
            )
        if len(artifact.body) > self.max_output_bytes:
            raise SecurityRejection("parsed source exceeds its expanded byte budget")
        try:
            text = artifact.body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise SecurityRejection("text source is not strict UTF-8") from error
        if "\x00" in text:
            raise SecurityRejection("text source contains a NUL byte")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode()
        if not normalized or len(normalized) > self.max_output_bytes:
            raise SecurityRejection("parsed source is empty or exceeds its byte budget")
        return normalized


@dataclass(frozen=True, slots=True)
class PatternScanner:
    name: str
    version: str
    target: str
    patterns: tuple[tuple[str, re.Pattern[bytes]], ...]

    def __post_init__(self) -> None:
        if self.target not in {"raw", "parsed"} or not self.patterns:
            raise ValueError("pattern scanner configuration is invalid")

    def findings(self, content: bytes) -> tuple[str, ...]:
        return tuple(label for label, pattern in self.patterns if pattern.search(content))


def _pattern(expression: bytes) -> re.Pattern[bytes]:
    return re.compile(expression, re.IGNORECASE | re.MULTILINE | re.DOTALL)


DEFAULT_SCANNERS = (
    PatternScanner(
        name="malware",
        version="rules-v1",
        target="raw",
        patterns=(("eicar", _pattern(rb"X5O!P%@AP\[4\\PZX54\(P\^\)7CC\)7}\$EICAR")),),
    ),
    PatternScanner(
        name="pii",
        version="rules-v1",
        target="parsed",
        patterns=(
            ("email", _pattern(rb"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
            ("us-ssn", _pattern(rb"\b\d{3}-\d{2}-\d{4}\b")),
            ("payment-card", _pattern(rb"\b(?:\d[ -]*?){13,19}\b")),
        ),
    ),
    PatternScanner(
        name="secrets",
        version="rules-v1",
        target="parsed",
        patterns=(
            ("private-key", _pattern(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
            ("aws-access-key", _pattern(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
            ("github-token", _pattern(rb"\bgh[opusr]_[A-Za-z0-9_]{20,}\b")),
        ),
    ),
    PatternScanner(
        name="prompt-injection",
        version="rules-v1",
        target="parsed",
        patterns=(
            (
                "instruction-override",
                _pattern(
                    rb"\b(?:ignore|disregard|forget)\b.{0,80}\b"
                    rb"(?:previous|prior|system|developer)\b.{0,40}\binstructions?\b"
                ),
            ),
            ("role-control-token", _pattern(rb"<\|(?:system|assistant|developer)\|>|\[INST\]")),
            (
                "system-prompt-exfiltration",
                _pattern(rb"\b(?:reveal|print|show)\b.{0,80}\bsystem prompt\b"),
            ),
        ),
    ),
)


class StrictArtifactValidator:
    required_scanners: ClassVar[dict[str, str]] = {
        "malware": "raw",
        "pii": "parsed",
        "secrets": "parsed",
        "prompt-injection": "parsed",
    }

    def __init__(
        self,
        *,
        parser: ArtifactParser | None = None,
        scanners: tuple[ContentScanner, ...] | None = None,
    ) -> None:
        self.parser = BoundedTextParser() if parser is None else parser
        configured: tuple[ContentScanner, ...] = (
            cast(tuple[ContentScanner, ...], DEFAULT_SCANNERS) if scanners is None else scanners
        )
        self.scanners = configured
        names = [scanner.name for scanner in configured]
        if len(names) != len(set(names)) or set(names) != set(self.required_scanners):
            raise ValueError("validator requires exactly the mandatory security scanners")
        if any(
            scanner.target != self.required_scanners[scanner.name] for scanner in configured
        ):
            raise ValueError("security scanner target does not match its mandatory phase")

    def validate(self, artifact: SourceArtifact) -> ValidatedArtifact:
        receipts: list[SecurityScanReceipt] = []
        receipts.extend(self._scan("raw", artifact.body))
        try:
            parsed = self.parser.parse(artifact)
        except SecurityRejection:
            raise
        except Exception as error:
            raise SecurityRejection(
                f"source parser failed closed: {type(error).__name__}"
            ) from error
        receipts.extend(self._scan("parsed", parsed))
        return ValidatedArtifact(
            artifact=artifact,
            body=parsed,
            content_digest=hashlib.sha256(parsed).hexdigest(),
            parser=self.parser.name,
            parser_version=self.parser.version,
            scans=tuple(receipts),
        )

    def _scan(self, target: str, content: bytes) -> list[SecurityScanReceipt]:
        receipts: list[SecurityScanReceipt] = []
        digest = hashlib.sha256(content).hexdigest()
        for scanner in self.scanners:
            if scanner.target != target:
                continue
            try:
                findings = scanner.findings(content)
            except Exception as error:
                raise SecurityRejection(
                    f"security scanner {scanner.name} failed closed: {type(error).__name__}"
                ) from error
            if not isinstance(findings, tuple) or any(
                not isinstance(finding, str) or not finding for finding in findings
            ):
                raise SecurityRejection(
                    f"security scanner {scanner.name} returned an invalid result"
                )
            if findings:
                raise SecurityRejection(
                    f"security scanner {scanner.name} rejected source: {', '.join(findings)}"
                )
            receipts.append(
                SecurityScanReceipt(
                    scanner=scanner.name,
                    version=scanner.version,
                    target=target,
                    content_digest=digest,
                )
            )
        return receipts
