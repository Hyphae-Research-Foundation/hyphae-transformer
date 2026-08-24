from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path

from celiums_rezero.knowledge import (
    AcquisitionPolicy,
    DurableAcquisitionWorker,
    DurableFinalizationWorker,
    FinalAnswer,
    FinalizationPolicy,
    FinalizationTimeout,
    InMemoryKnowledgeIndex,
    InMemorySourceConnector,
    KnowledgeCoordinator,
    KnowledgeScheduler,
    PermanentFinalizationError,
    SQLiteTenantStore,
    SufficiencyPolicy,
    TenantId,
    TransientFinalizationError,
    check_notification_sink,
)
from celiums_rezero.knowledge.finalization import retry_delay_us
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    DeadLetterReason,
    FinalizationPhase,
    JobStatus,
    NotificationReceipt,
    PreparedNotification,
    SourcePolicy,
)


class Clock:
    def __init__(self, now: int = 1_800_000_000_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class Answerer:
    def __init__(self, answer: FinalAnswer | None) -> None:
        self.result = answer
        self.calls = 0

    def answer(
        self, job: AcquisitionJob, *, timeout_seconds: float
    ) -> FinalAnswer | None:
        assert job.status is JobStatus.ANSWERING
        assert timeout_seconds == 5
        self.calls += 1
        return self.result


class Sink:
    sink_id = "tenant-events-v1"

    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.calls: list[str] = []
        self.receipts: dict[str, NotificationReceipt] = {}

    def deliver(
        self, command: PreparedNotification, *, timeout_seconds: float
    ) -> NotificationReceipt:
        assert command.notification_id is not None
        assert timeout_seconds == 5
        self.calls.append(command.notification_id)
        if self.fail_once:
            self.fail_once = False
            raise TransientFinalizationError("temporary sink outage")
        receipt = self.receipts.get(command.notification_id)
        if receipt is None:
            receipt = NotificationReceipt(
                tenant=command.tenant,
                job_id=command.job_id,
                notification_id=command.notification_id,
                sink_id=command.sink_id,
                command_digest=command.command_digest or "",
                provider_receipt=f"accepted:{command.notification_id}",
            )
            self.receipts[command.notification_id] = receipt
        return receipt


def ready_store(
    tmp_path: Path, clock: Clock
) -> tuple[SQLiteTenantStore, TenantId, str]:
    tmp_path.chmod(0o700)
    tenant = TenantId("tenant_a")
    store = SQLiteTenantStore(tmp_path / "jobs.sqlite3", tenant=tenant, clock_us=clock)
    query = "durable finalization"
    item = AcquisitionJob(
        tenant=tenant,
        query=query,
        query_digest=hashlib.sha256(query.encode()).hexdigest(),
        corpus_generation="generation-v1",
        policy_version="policy-v1",
        embedding_profile="fixture-v1",
        source_id="official_docs",
    )
    created, _ = store.enqueue(item)
    assert created.job_id is not None
    with store._transaction() as database:
        database.execute(
            "UPDATE jobs SET status = 'ready' WHERE job_id = ?", (created.job_id,)
        )
    return store, tenant, created.job_id


def worker(
    store: SQLiteTenantStore,
    answerer: Answerer,
    sink: Sink,
    *,
    worker_id: str = "finalizer-a",
) -> DurableFinalizationWorker:
    return DurableFinalizationWorker(
        store=store,
        worker_id=worker_id,
        lease_seconds=10,
        answerer=answerer,
        sink=sink,
        policy=FinalizationPolicy(
            answer_timeout_seconds=5,
            notification_timeout_seconds=5,
            lease_safety_seconds=1,
            retry_base_seconds=5,
            retry_max_seconds=60,
        ),
    )


def test_finalization_stages_delivers_and_completes(tmp_path: Path) -> None:
    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    answerer = Answerer(FinalAnswer("Verified answer", ("passage_0123456789abcdef",)))
    sink = Sink()
    result = worker(store, answerer, sink).run_next(job_id=job_id)
    assert result is not None and result.status is JobStatus.COMPLETED
    command = store.prepared_notification(tenant, job_id)
    assert command is not None and command.answer == "Verified answer"
    assert len(sink.calls) == 1 and answerer.calls == 1


def test_insufficient_finalization_is_terminal_without_notification(tmp_path: Path) -> None:
    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    sink = Sink()
    result = worker(store, Answerer(None), sink).run_next(job_id=job_id)
    assert result is not None and result.status is JobStatus.INSUFFICIENT_AFTER_INGEST
    assert store.prepared_notification(tenant, job_id) is None
    assert sink.calls == []


def test_notification_retry_backoff_and_recovery(tmp_path: Path) -> None:
    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    answerer = Answerer(FinalAnswer("Verified answer", ()))
    sink = Sink(fail_once=True)
    first = worker(store, answerer, sink).run_next(job_id=job_id)
    assert first is not None and first.status is JobStatus.NOTIFYING
    assert store.notification_attempts(tenant, job_id) == 1
    assert worker(store, answerer, sink).run_next(job_id=job_id) is None
    clock.now += 5_000_000
    reopened = SQLiteTenantStore(store.path, tenant=tenant, clock_us=clock)
    completed = worker(reopened, answerer, sink, worker_id="finalizer-b").run_next(
        job_id=job_id
    )
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert len(set(sink.calls)) == 1


def test_expired_answering_lease_is_reclaimed(tmp_path: Path) -> None:
    clock = Clock()
    store, _, job_id = ready_store(tmp_path, clock)
    claimed = store.claim_finalization(
        owner_id="crashed-finalizer", lease_seconds=10, job_id=job_id
    )
    assert claimed is not None and claimed[0].status is JobStatus.ANSWERING
    clock.now = claimed[1].expires_at_us
    answerer = Answerer(FinalAnswer("Recovered answer", ()))
    completed = worker(store, answerer, Sink()).run_next(job_id=job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED


def test_notification_outbox_is_tenant_bound(tmp_path: Path) -> None:
    clock = Clock()
    store, _, job_id = ready_store(tmp_path, clock)
    worker(store, Answerer(FinalAnswer("Answer", ())), Sink()).run_next(job_id=job_id)
    try:
        store.prepared_notification(TenantId("tenant_b"), job_id)
    except PermissionError:
        pass
    else:
        raise AssertionError("cross-tenant notification lookup succeeded")


def test_scheduler_isolates_answerer_failure_and_reports_it(tmp_path: Path) -> None:
    clock = Clock()
    store, _, job_id = ready_store(tmp_path, clock)

    class BrokenAnswerer:
        def answer(
            self, job: AcquisitionJob, *, timeout_seconds: float
        ) -> FinalAnswer | None:
            del job
            assert timeout_seconds == 5
            raise TransientFinalizationError("answerer failed")

    finalization = DurableFinalizationWorker(
        store=store,
        worker_id="broken-finalizer",
        lease_seconds=10,
        answerer=BrokenAnswerer(),
        sink=Sink(),
        policy=FinalizationPolicy(
            answer_timeout_seconds=5,
            notification_timeout_seconds=5,
            lease_safety_seconds=1,
        ),
    )
    source = SourcePolicy(
        source_id="official_docs",
        allowed_hosts=("docs.example.com",),
        allowed_license_ids=("Apache-2.0",),
    )
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(),
        acquisition=AcquisitionPolicy(version="policy-v1", sources=(source,)),
        store=store,
        embedding_profile="fixture-v1",
    )

    class Embedder:
        profile = "fixture-v1"
        dimensions = 2

        def embed(self, text: str) -> tuple[float, ...]:
            return (len(text) / 100, 0.5)

    index = InMemoryKnowledgeIndex()
    acquisition = DurableAcquisitionWorker(
        worker_id="idle-acquisition",
        lease_seconds=30,
        coordinator=coordinator,
        connector=InMemorySourceConnector({}),
        embedder=Embedder(),
        ingestor=index,
        verifier=index,
    )
    scheduler = KnowledgeScheduler(
        store=store,
        acquisition=acquisition,
        finalization=finalization,
    )
    scheduler.tick()
    latest = store.get(TenantId("tenant_a"), job_id)
    assert latest is not None and latest.status is JobStatus.ANSWERING


def test_prepared_notification_rejects_mirrored_column_corruption(tmp_path: Path) -> None:
    clock = Clock()
    store, _, job_id = ready_store(tmp_path, clock)
    claimed = store.claim_finalization(
        owner_id="finalizer-a", lease_seconds=30, job_id=job_id
    )
    assert claimed is not None
    job_value, lease = claimed
    command = PreparedNotification(
        tenant=job_value.tenant,
        job_id=job_id,
        sink_id=Sink.sink_id,
        answer="Answer",
        evidence_handles=(),
        corpus_generation=job_value.corpus_generation,
        query_digest=job_value.query_digest,
    )
    store.stage_notification(lease, command)
    with store._transaction() as connection:
        connection.execute(
            "UPDATE notification_outbox SET sink_id = 'tampered-sink' WHERE job_id = ?",
            (job_id,),
        )
    try:
        store.prepared_notification(TenantId("tenant_a"), job_id)
    except RuntimeError as error:
        assert "integrity" in str(error)
    else:
        raise AssertionError("tampered mirrored notification fields were accepted")


def test_scheduler_recovers_expired_pre_outbox_work(tmp_path: Path) -> None:
    clock = Clock()
    tmp_path.chmod(0o700)
    tenant = TenantId("tenant_a")
    store = SQLiteTenantStore(tmp_path / "jobs.sqlite3", tenant=tenant, clock_us=clock)
    created, _ = store.enqueue(
        AcquisitionJob(
            tenant=tenant,
            query="scheduler recovery",
            query_digest=hashlib.sha256(b"scheduler recovery").hexdigest(),
            corpus_generation="generation-v1",
            policy_version="policy-v1",
            embedding_profile="fixture-v1",
            source_id="official_docs",
        )
    )
    claimed = store.claim(owner_id="crashed-worker", lease_seconds=10)
    assert claimed is not None
    clock.now = claimed[1].expires_at_us

    source = SourcePolicy(
        source_id="official_docs",
        allowed_hosts=("docs.example.com",),
        allowed_license_ids=("Apache-2.0",),
    )
    coordinator = KnowledgeCoordinator(
        sufficiency=SufficiencyPolicy(),
        acquisition=AcquisitionPolicy(version="policy-v1", sources=(source,)),
        store=store,
        embedding_profile="fixture-v1",
    )

    class Embedder:
        profile = "fixture-v1"
        dimensions = 2

        def embed(self, text: str) -> tuple[float, ...]:
            return (len(text) / 100, 0.5)

    index = InMemoryKnowledgeIndex()
    acquisition = DurableAcquisitionWorker(
        worker_id="scheduler-acquisition",
        lease_seconds=30,
        coordinator=coordinator,
        connector=InMemorySourceConnector({}),
        embedder=Embedder(),
        ingestor=index,
        verifier=index,
    )
    finalization = worker(store, Answerer(None), Sink())

    scheduler = KnowledgeScheduler(
        store=store,
        acquisition=acquisition,
        finalization=finalization,
    )
    assert scheduler.tick() >= 1
    recovered = store.get(tenant, created.job_id or "")
    assert recovered is not None and recovered.attempts == 2


def test_schema_v1_migrates_notification_outbox(tmp_path: Path) -> None:
    import celiums_rezero.knowledge.store as store_module

    tmp_path.chmod(0o700)
    path = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(store_module._SCHEMA_V1)
    connection.execute(
        "INSERT INTO tenant_meta VALUES (1, 'tenant_a', 1, ?)",
        (1_800_000_000_000_000,),
    )
    connection.commit()
    connection.execute("PRAGMA journal_mode=WAL")
    connection.close()
    path.chmod(0o600)

    migrated = SQLiteTenantStore(path, tenant=TenantId("tenant_a"), clock_us=Clock())
    with migrated._connect() as reopened:
        version = reopened.execute(
            "SELECT schema_version FROM tenant_meta WHERE singleton = 1"
        ).fetchone()[0]
        table = reopened.execute(
            "SELECT name FROM sqlite_master WHERE name = 'notification_outbox'"
        ).fetchone()
    assert version == 4 and table is not None


def test_schema_v2_migrates_operational_tables(tmp_path: Path) -> None:
    import celiums_rezero.knowledge.store as store_module

    tmp_path.chmod(0o700)
    path = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(store_module._SCHEMA_V2)
    connection.execute(
        "INSERT INTO tenant_meta VALUES (1, 'tenant_a', 2, ?)",
        (1_800_000_000_000_000,),
    )
    connection.commit()
    connection.execute("PRAGMA journal_mode=WAL")
    connection.close()
    path.chmod(0o600)
    migrated = SQLiteTenantStore(path, tenant=TenantId("tenant_a"), clock_us=Clock())
    with migrated._connect() as reopened:
        version = reopened.execute(
            "SELECT schema_version FROM tenant_meta WHERE singleton = 1"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in reopened.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert version == 4
    assert {"answer_retries", "finalization_dead_letters"}.issubset(tables)


def test_deterministic_jitter_is_bounded_and_stable() -> None:
    first = retry_delay_us(
        job_id="job_0123456789abcdef",
        phase=FinalizationPhase.NOTIFYING,
        failure=3,
        base_us=1_000_000,
        maximum_us=60_000_000,
    )
    second = retry_delay_us(
        job_id="job_0123456789abcdef",
        phase=FinalizationPhase.NOTIFYING,
        failure=3,
        base_us=1_000_000,
        maximum_us=60_000_000,
    )
    assert first == second
    assert 2_000_000 <= first <= 4_000_000


def test_permanent_sink_failure_dead_letters(tmp_path: Path) -> None:
    class PermanentSink(Sink):
        def deliver(
            self, command: PreparedNotification, *, timeout_seconds: float
        ) -> NotificationReceipt:
            del command, timeout_seconds
            raise PermanentFinalizationError("invalid destination")

    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    result = worker(
        store, Answerer(FinalAnswer("Answer", ())), PermanentSink()
    ).run_next(job_id=job_id)
    assert result is not None and result.status is JobStatus.FAILED
    letters = store.finalization_dead_letters(tenant)
    assert len(letters) == 1
    assert letters[0].reason is DeadLetterReason.PERMANENT
    assert letters[0].phase is FinalizationPhase.NOTIFYING


def test_transient_answer_failures_exhaust_to_dead_letter(tmp_path: Path) -> None:
    class TimeoutAnswerer:
        def answer(
            self, job: AcquisitionJob, *, timeout_seconds: float
        ) -> FinalAnswer | None:
            del job, timeout_seconds
            raise FinalizationTimeout("answer timed out")

    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    finalizer = DurableFinalizationWorker(
        store=store,
        worker_id="timeout-answerer",
        lease_seconds=10,
        answerer=TimeoutAnswerer(),
        sink=Sink(),
        policy=FinalizationPolicy(
            answer_timeout_seconds=5,
            notification_timeout_seconds=5,
            lease_safety_seconds=1,
            retry_base_seconds=1,
            retry_max_seconds=1,
            max_answer_failures=2,
        ),
    )
    first = finalizer.run_next(job_id=job_id)
    assert first is not None and first.status is JobStatus.ANSWERING
    clock.now += 1_000_000
    second = finalizer.run_next(job_id=job_id)
    assert second is not None and second.status is JobStatus.FAILED
    assert store.finalization_dead_letters(tenant)[0].reason is (
        DeadLetterReason.RETRIES_EXHAUSTED
    )


def test_queue_snapshot_reports_due_deferred_and_dead_letters(tmp_path: Path) -> None:
    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    snapshot = store.finalization_queue_snapshot(tenant)
    assert snapshot.ready == 1 and snapshot.dead_lettered == 0
    sink = Sink(fail_once=True)
    worker(store, Answerer(FinalAnswer("Answer", ())), sink).run_next(job_id=job_id)
    deferred = store.finalization_queue_snapshot(tenant)
    assert deferred.notifying_deferred == 1
    clock.now += 5_000_000
    due = store.finalization_queue_snapshot(tenant)
    assert due.notifying_due == 1


def test_sink_receipt_mismatch_is_permanent(tmp_path: Path) -> None:
    class BadSink(Sink):
        def deliver(
            self, command: PreparedNotification, *, timeout_seconds: float
        ) -> NotificationReceipt:
            receipt = super().deliver(command, timeout_seconds=timeout_seconds)
            return NotificationReceipt(
                tenant=receipt.tenant,
                job_id=receipt.job_id,
                notification_id=receipt.notification_id,
                sink_id="wrong-sink",
                command_digest=receipt.command_digest,
                provider_receipt=receipt.provider_receipt,
            )

    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    result = worker(store, Answerer(FinalAnswer("Answer", ())), BadSink()).run_next(
        job_id=job_id
    )
    assert result is not None and result.status is JobStatus.FAILED
    assert "does not match" in store.finalization_dead_letters(tenant)[0].error


def test_notification_sink_conformance_replays_original_receipt() -> None:
    command = PreparedNotification(
        tenant=TenantId("tenant_a"),
        job_id="job_0123456789abcdef",
        sink_id=Sink.sink_id,
        answer="Conformance answer",
        evidence_handles=(),
        corpus_generation="generation-v1",
        query_digest="1" * 64,
    )
    sink = Sink()
    receipt = check_notification_sink(sink, command, timeout_seconds=5)
    assert receipt.notification_id == command.notification_id
    assert len(sink.receipts) == 1
    try:
        check_notification_sink(sink, command, timeout_seconds=math.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite conformance timeout was accepted")


def test_unicode_callback_error_is_safely_dead_lettered(tmp_path: Path) -> None:
    class UnicodeAnswerer:
        def answer(
            self, job: AcquisitionJob, *, timeout_seconds: float
        ) -> FinalAnswer | None:
            del job, timeout_seconds
            raise PermanentFinalizationError("bad surrogate \ud800")

    clock = Clock()
    store, tenant, job_id = ready_store(tmp_path, clock)
    result = DurableFinalizationWorker(
        store=store,
        worker_id="unicode-answerer",
        lease_seconds=10,
        answerer=UnicodeAnswerer(),
        sink=Sink(),
        policy=FinalizationPolicy(
            answer_timeout_seconds=5,
            notification_timeout_seconds=5,
            lease_safety_seconds=1,
        ),
    ).run_next(job_id=job_id)
    assert result is not None and result.status is JobStatus.FAILED
    assert "?" in store.finalization_dead_letters(tenant)[0].error
