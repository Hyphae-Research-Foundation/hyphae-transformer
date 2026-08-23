"""Prometheus text rendering and append-only hash-chain audit."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from celiums_rezero.knowledge.schemas import FinalizationQueueSnapshot
from celiums_rezero.lab.serialization import canonical_json


def render_prometheus(snapshot: FinalizationQueueSnapshot) -> str:
    states = (
        ("ready", snapshot.ready),
        ("answering_due", snapshot.answering_due),
        ("answering_deferred", snapshot.answering_deferred),
        ("notifying_due", snapshot.notifying_due),
        ("notifying_deferred", snapshot.notifying_deferred),
        ("leased", snapshot.leased),
        ("dead_lettered", snapshot.dead_lettered),
    )
    lines = [
        "# HELP celiums_rezero_finalization_jobs Finalization jobs by bounded state.",
        "# TYPE celiums_rezero_finalization_jobs gauge",
    ]
    lines.extend(
        f'celiums_rezero_finalization_jobs{{state="{state}"}} {value}'
        for state, value in states
    )
    lines.extend(
        [
            "# HELP celiums_rezero_notification_attempts_total Durable notification attempts.",
            "# TYPE celiums_rezero_notification_attempts_total counter",
            f"celiums_rezero_notification_attempts_total {snapshot.notification_attempts}",
            "# HELP celiums_rezero_oldest_claimable_seconds Oldest claimable finalization age.",
            "# TYPE celiums_rezero_oldest_claimable_seconds gauge",
            (
                "celiums_rezero_oldest_claimable_seconds "
                f"{snapshot.oldest_claimable_age_seconds:.6f}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_metrics(path: Path, snapshot: FinalizationQueueSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(render_prometheus(snapshot), encoding="ascii")
    temporary.chmod(0o600)
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    sequence: int
    occurred_at_us: int
    event_type: str
    subject_digest: str
    outcome: str
    detail: dict[str, object]
    previous_hash: str
    record_hash: str | None = None

    def __post_init__(self) -> None:
        unsigned = {
            "schema": "celiums-rezero-audit-v1",
            "sequence": self.sequence,
            "occurred_at_us": self.occurred_at_us,
            "event_type": self.event_type,
            "subject_digest": self.subject_digest,
            "outcome": self.outcome,
            "detail": self.detail,
            "previous_hash": self.previous_hash,
        }
        expected = hashlib.sha256(
            b"celiums-rezero-audit-v1\0" + canonical_json(unsigned).encode()
        ).hexdigest()
        if self.record_hash is None:
            object.__setattr__(self, "record_hash", expected)
        elif self.record_hash != expected:
            raise ValueError("audit record digest does not match its contents")


class AuditChain:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists() and path.is_symlink():
            raise ValueError("audit path cannot be a symlink")

    def append(
        self,
        *,
        occurred_at_us: int,
        event_type: str,
        subject_digest: str,
        outcome: str,
        detail: dict[str, object],
    ) -> AuditRecord:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self._verify_unlocked()
            previous = "0" * 64 if not records else records[-1].record_hash or ""
            record = AuditRecord(
                sequence=len(records) + 1,
                occurred_at_us=occurred_at_us,
                event_type=event_type,
                subject_digest=subject_digest,
                outcome=outcome,
                detail=detail,
                previous_hash=previous,
            )
            line = canonical_json(record) + "\n"
            if len(line.encode()) > 65_536:
                raise ValueError("audit record exceeds its byte bound")
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                view = memoryview(line.encode())
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(lock)
        return record

    def verify(self) -> tuple[AuditRecord, ...]:
        if not self.path.exists():
            return ()
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            fcntl.flock(lock, fcntl.LOCK_SH)
            return self._verify_unlocked()
        finally:
            os.close(lock)

    def _verify_unlocked(self) -> tuple[AuditRecord, ...]:
        if not self.path.exists():
            return ()
        records: list[AuditRecord] = []
        previous = "0" * 64
        for line in self.path.read_text(encoding="ascii").splitlines():
            value = json.loads(line)
            record = AuditRecord(
                sequence=value["sequence"],
                occurred_at_us=value["occurred_at_us"],
                event_type=value["event_type"],
                subject_digest=value["subject_digest"],
                outcome=value["outcome"],
                detail=value["detail"],
                previous_hash=value["previous_hash"],
                record_hash=value["record_hash"],
            )
            if record.sequence != len(records) + 1 or record.previous_hash != previous:
                raise ValueError("audit chain sequence or linkage is invalid")
            records.append(record)
            previous = record.record_hash or ""
        return tuple(records)
