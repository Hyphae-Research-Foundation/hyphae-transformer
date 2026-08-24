from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from celiums_rezero.knowledge import (
    SQLiteMailboxConfig,
    SQLiteMailboxNotificationSink,
    SupervisedFrozenGemmaRuntime,
    SupervisedFrozenRuntimeConfig,
    TenantId,
    check_notification_sink,
)
from celiums_rezero.knowledge.finalization import (
    FinalizationTimeout,
    PermanentFinalizationError,
)
from celiums_rezero.knowledge.model_runtime import RESPONSE_SCHEMA
from celiums_rezero.knowledge.orchestration import (
    FrozenModelIdentity,
    GovernedModelRequest,
)
from celiums_rezero.knowledge.schemas import EvidenceHit, PreparedNotification
from celiums_rezero.knowledge.supervisor import run_supervised


def executable(tmp_path: Path, body: str) -> tuple[Path, str]:
    path = tmp_path / "runtime"
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="ascii")
    path.chmod(0o700)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def model_request() -> GovernedModelRequest:
    text = "Verified quotation"
    return GovernedModelRequest(
        "question",
        "generation_one",
        (
            EvidenceHit(
                "passage_0123456789abcdef",
                "docs",
                "v1",
                text,
                1.0,
                hashlib.sha256(text.encode()).hexdigest(),
            ),
        ),
    )


def test_supervised_runtime_exchanges_bounded_quoted_claims(tmp_path: Path) -> None:
    identity = FrozenModelIdentity("model", "revision", "a" * 64, "runtime-v1")
    program = """
import hashlib,json,sys
request=json.load(sys.stdin)
passage=request['passages'][0]
print(json.dumps({
    'schema':'hyphae-frozen-runtime-response/v1',
    'request_id':request['request_id'],
    'identity':request['identity'],
    'decision':'answer',
    'claims':[{'handle':passage['handle'],'quote':passage['text']}],
},sort_keys=True,separators=(',',':')))
"""
    path, digest = executable(tmp_path, program)
    runtime = SupervisedFrozenGemmaRuntime(
        SupervisedFrozenRuntimeConfig(path, identity, digest)
    )

    result = runtime.infer(model_request(), timeout_seconds=2)
    exchange = runtime.infer_exchange(model_request(), timeout_seconds=2)

    assert result.identity == identity
    assert result.claims[0].quote == "Verified quotation"
    assert exchange.result == result
    assert hashlib.sha256(exchange.request_payload).hexdigest()
    assert hashlib.sha256(exchange.response_payload).hexdigest()


def test_supervised_runtime_enforces_timeout_and_executable_digest(tmp_path: Path) -> None:
    identity = FrozenModelIdentity("model", "revision", "a" * 64, "runtime-v1")
    path, digest = executable(tmp_path, "import time\ntime.sleep(10)\n")
    runtime = SupervisedFrozenGemmaRuntime(
        SupervisedFrozenRuntimeConfig(
            path,
            identity,
            digest,
            termination_grace_seconds=0.01,
        )
    )
    with pytest.raises(FinalizationTimeout):
        runtime.infer(model_request(), timeout_seconds=0.05)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o700)
    with pytest.raises(PermissionError, match="digest changed"):
        runtime.infer(model_request(), timeout_seconds=1)


def notification(sink_id: str) -> PreparedNotification:
    return PreparedNotification(
        tenant=TenantId("tenant_a"),
        job_id="job_0123456789abcdef",
        sink_id=sink_id,
        answer="Verified answer",
        evidence_handles=("passage_0123456789abcdef",),
        corpus_generation="generation_one",
        query_digest="a" * 64,
    )


def test_sqlite_mailbox_deduplicates_across_restart(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    config = SQLiteMailboxConfig(
        tenant_id="tenant_a",
        path=(tmp_path / "mailbox.sqlite3").absolute(),
        mailbox_id="owner-events-v1",
    )
    first = SQLiteMailboxNotificationSink(config)
    command = notification(first.sink_id)
    receipt = check_notification_sink(first, command, timeout_seconds=2)
    assert first.accepted_count() == 1

    recovered = SQLiteMailboxNotificationSink(config)

    assert recovered.deliver(command, timeout_seconds=2) == receipt
    assert recovered.accepted_count() == 1


def test_sqlite_mailbox_rejects_cross_tenant_command(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    config = SQLiteMailboxConfig(
        tenant_id="tenant_a",
        path=(tmp_path / "mailbox.sqlite3").absolute(),
        mailbox_id="owner-events-v1",
    )
    sink = SQLiteMailboxNotificationSink(config)
    command = notification(sink.sink_id)
    wrong = PreparedNotification(
        tenant=TenantId("tenant_b"),
        job_id=command.job_id,
        sink_id=command.sink_id,
        answer=command.answer,
        evidence_handles=command.evidence_handles,
        corpus_generation=command.corpus_generation,
        query_digest=command.query_digest,
    )
    with pytest.raises(PermanentFinalizationError, match="binding"):
        sink.deliver(wrong, timeout_seconds=2)


def test_runtime_response_schema_constant_is_stable() -> None:
    assert RESPONSE_SCHEMA == "hyphae-frozen-runtime-response/v1"


def test_supervisor_passes_bounded_stdin(tmp_path: Path) -> None:
    path, _ = executable(
        tmp_path,
        "import sys\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n",
    )
    result = run_supervised(
        (str(path),),
        timeout_seconds=2,
        input_bytes=b"bounded request",
    )
    assert result.returncode == 0
    assert result.stdout == b"bounded request"
