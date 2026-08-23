from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from celiums_rezero.knowledge import (
    AcquisitionPolicy,
    DurableAcquisitionWorker,
    DurableFinalizationWorker,
    FinalAnswer,
    InMemoryKnowledgeIndex,
    InMemorySourceConnector,
    KnowledgeCoordinator,
    KnowledgeScheduler,
    SQLiteTenantStore,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
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

    def answer(self, job: AcquisitionJob) -> FinalAnswer | None:
        assert job.status is JobStatus.ANSWERING
        self.calls += 1
        return self.result


class Sink:
    sink_id = "tenant-events-v1"

    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.calls: list[str] = []
        self.receipts: dict[str, NotificationReceipt] = {}

    def deliver(self, command: PreparedNotification) -> NotificationReceipt:
        assert command.notification_id is not None
        self.calls.append(command.notification_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary sink outage")
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
        lease_seconds=30,
        answerer=answerer,
        sink=sink,
        retry_base_seconds=5,
        retry_max_seconds=60,
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
        def answer(self, job: AcquisitionJob) -> FinalAnswer | None:
            del job
            raise RuntimeError("answerer failed")

    finalization = DurableFinalizationWorker(
        store=store,
        worker_id="broken-finalizer",
        lease_seconds=30,
        answerer=BrokenAnswerer(),
        sink=Sink(),
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
    assert scheduler.last_errors and "answerer failed" in scheduler.last_errors[0]
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
    assert version == 2 and table is not None
