#!/usr/bin/env python3
"""Run one isolated live Hyphae multi-step navigation canary with the ReZero pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from celiums_rezero.governed.data import host_control_values
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.navigation import HOST_CONTROL_CONTRACT
from celiums_rezero.governed.navigation_experiment import (
    decide_navigation_step,
    load_navigation_pilot,
)
from celiums_rezero.knowledge import (
    HYPHAE_210_RETRIEVAL_PROFILE,
    AcquisitionPolicy,
    DurableAcquisitionWorker,
    DurablePublicationAuthorizer,
    EvidenceBundle,
    GenerationAuthority,
    GenerationRoutedRetriever,
    HyphaeShadowIngestor,
    InMemorySourceConnector,
    JobStatus,
    KnowledgeCoordinator,
    PublicationReceiptStore,
    SQLiteTenantStore,
    StrictArtifactValidator,
    SufficiencyPolicy,
    TenantId,
)
from celiums_rezero.knowledge.acquisition import (
    ChunkingPolicy,
    chunk_validated_artifact,
    ingest_idempotency_key,
)
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.embedding import (
    MiniLML6V2EmbeddingProvider,
    checked_embedding,
    minilm_l6_v2_preflight,
)
from celiums_rezero.knowledge.live import _u128_idempotency
from celiums_rezero.knowledge.schemas import (
    EmbeddedChunk,
    GenerationManifest,
    SourceArtifact,
    SourcePolicy,
)

HYPHAE_ARCHIVE_SHA256 = "a1e8cf56d9b9a96ee5f230aa4dec92b2541792f7ca4bb40c0dbf761d9ed3e0aa"
HYPHAE_BINARY_SHA256 = "a00ea0cfc502ad63d65c42357664f7664f8a8c482fbdeb24a4f5511feceb45d0"
HYPHAE_WHEEL_SHA256 = "fd6503abbcac18db9a6705682b80a83904389f146e6dd0c4d17fdef49535a5fb"
COLLECTION_SHA256 = "181552f7f9666546db8f09b3e89be98e99f4c4e09be227f6d257da93029ea527"
NAVIGATION_BUNDLE_SHA256 = "5cb0381c03f944706819e6c5ce2d9dc71be63c27b88292cfd28fd2b489d7b7c8"
NAVIGATION_CHECKPOINT_SHA256 = "47940ec5f690fab92f13601ca6c1593b8897d062a04c3b853e4fc99fd762aca2"
QUERY = "what is the approved maintenance window?"
BODY = b"Service policy: approved maintenance window is 02:00-04:00 UTC."
DISTRACTOR = b"Office note: the lobby fern is watered on Tuesdays."
GENERATION = "generation_navigation_canary_v1"
DISTRACTOR_GENERATION = "generation_navigation_canary_distractor_v1"
TENANT = TenantId("tenant_navigation_canary")
_ACTIVE_DAEMON: subprocess.Popen[str] | None = None


class BoundedRecordingClient:
    def __init__(self, client: object, request_options: type[object]) -> None:
        self.client = client
        self.request_options = request_options
        self.last_search: dict[str, Any] | None = None

    def search_ingest(
        self,
        collection: int,
        batch: dict[str, object],
        *,
        options: object | None = None,
    ) -> object:
        bounded = options or self.request_options(
            deadline_micros=time.time_ns() // 1000 + 30_000_000,
            durability="strict",
        )
        return self.client.search_ingest(collection, batch, options=bounded)

    def search_collection(
        self,
        collection: int,
        request: dict[str, object],
        *,
        options: object | None = None,
    ) -> object:
        response = self.client.search_collection(collection, request, options=options)
        value = getattr(response, "value", None)
        if isinstance(value, dict):
            self.last_search = value
        return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyphae-archive", type=Path, required=True)
    parser.add_argument("--hyphae-binary", type=Path, required=True)
    parser.add_argument("--hyphae-wheel", type=Path, required=True)
    parser.add_argument("--minilm-model", type=Path, required=True)
    parser.add_argument("--gemma-model", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-patch-sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(arguments.out.iterdir()):
        allowed = {
            "gemma4-e4b-preflight.json",
            "minilm-preflight.json",
            "python-freeze.txt",
            "source-revision.txt",
        }
        if {item.name for item in arguments.out.iterdir()} - allowed:
            raise ValueError("navigation evidence directory contains unexpected files")
    _write_json(
        arguments.out / "navigation-campaign-report.json",
        {
            "schema": "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v1",
            "completed": False,
        },
    )
    daemon: subprocess.Popen[str] | None = None
    report: dict[str, object]
    try:
        report, daemon = run(arguments)
    except Exception:
        report = {
            "schema": "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v1",
            "completed": False,
            "passed": False,
            "failure": "navigation canary failed"[:4096],
            "source_revision": arguments.source_revision,
        }
        raise
    finally:
        active = daemon if daemon is not None else _ACTIVE_DAEMON
        if active is not None:
            _stop_daemon(active, arguments.work_root / "hyphae.sock")
        shutil.rmtree(arguments.work_root, ignore_errors=True)
        report["work_root_removed"] = not arguments.work_root.exists()
        _write_json(arguments.out / "navigation-campaign-report.json", report)
    return 0


def run(arguments: argparse.Namespace) -> tuple[dict[str, object], subprocess.Popen[str] | None]:
    if arguments.work_root.exists() or not arguments.work_root.is_absolute():
        raise ValueError("navigation work root must be a new absolute path")
    arguments.work_root.mkdir(mode=0o700)
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.source_revision) or not re.fullmatch(
        r"[0-9a-f]{64}", arguments.source_patch_sha256
    ):
        raise ValueError("navigation source identity is invalid")
    identities = {
        "hyphae_archive_sha256": _require_digest(arguments.hyphae_archive, HYPHAE_ARCHIVE_SHA256),
        "hyphae_binary_sha256": _require_digest(arguments.hyphae_binary, HYPHAE_BINARY_SHA256),
        "hyphae_wheel_sha256": _require_digest(arguments.hyphae_wheel, HYPHAE_WHEEL_SHA256),
        "navigation_bundle_sha256": _require_digest(arguments.pilot, NAVIGATION_BUNDLE_SHA256),
    }
    if arguments.hyphae_wheel.name != "hyphae_sdk-2.1.0-py3-none-any.whl":
        raise ValueError("navigation Hyphae wheel coordinate is invalid")
    checkpoint_bytes = subprocess.run(
        ["tar", "-xOzf", str(arguments.pilot), "navigation-control.pt"],
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    if hashlib.sha256(checkpoint_bytes).hexdigest() != NAVIGATION_CHECKPOINT_SHA256:
        raise ValueError("navigation checkpoint digest differs")
    checkpoint = torch.load(
        __import__("io").BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=False,
    )
    maximum_evidence_items = int(checkpoint["maximum_evidence_items"])
    if maximum_evidence_items != 8:
        raise ValueError("navigation checkpoint evidence bound differs")
    import hyphae_sdk
    from hyphae_sdk.v2 import HyphaeClient, RequestOptions
    from hyphae_sdk.v2.protocol import PROTOCOL_MAJOR, PROTOCOL_MINOR

    if hyphae_sdk.__version__ != "2.1.0" or (PROTOCOL_MAJOR, PROTOCOL_MINOR) != (1, 5):
        raise RuntimeError("navigation Hyphae SDK or protocol identity differs")
    binary_version = _run_json([str(arguments.hyphae_binary), "version", "--json"])
    if binary_version.get("engine_version") != "2.1.0":
        raise RuntimeError("navigation Hyphae binary version differs")
    embedder = MiniLML6V2EmbeddingProvider(arguments.minilm_model)
    minilm = minilm_l6_v2_preflight(arguments.minilm_model)
    _write_json(arguments.out / "minilm-preflight.json", minilm)
    policy = SufficiencyPolicy()
    backbone = Gemma4E4BFrozenBackbone(arguments.gemma_model, device="cuda:0")
    device = torch.device("cuda:0")
    checkpoint_path = arguments.work_root / "navigation-control.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    pilot = load_navigation_pilot(
        checkpoint_path,
        hidden_size=backbone.identity.hidden_size,
        device=device,
    )
    artifact = SourceArtifact(
        tenant=TENANT,
        source_id="official_docs",
        source_version="navigation-v1",
        body=BODY,
        content_type="text/plain",
        license_id="Apache-2.0",
        content_digest=hashlib.sha256(BODY).hexdigest(),
    )
    source_policy = SourcePolicy(
        source_id=artifact.source_id,
        allowed_hosts=("docs.example.com",),
        allowed_mime_types=("text/plain",),
        allowed_license_ids=("Apache-2.0",),
        max_download_bytes=4096,
    )
    validator = StrictArtifactValidator()
    validated = validator.validate(artifact)
    embedded = tuple(
        EmbeddedChunk(
            chunk,
            embedder.profile,
            checked_embedding(embedder, chunk.text),
        )
        for chunk in chunk_validated_artifact(validated, ChunkingPolicy())
    )
    if any(len(item.values) != 384 for item in embedded):
        raise RuntimeError("navigation MiniLM chunk dimensions differ")
    distractor = SourceArtifact(
        tenant=TENANT,
        source_id="office_notes",
        source_version="navigation-v1",
        body=DISTRACTOR,
        content_type="text/plain",
        license_id="Apache-2.0",
        content_digest=hashlib.sha256(DISTRACTOR).hexdigest(),
    )
    distractor_policy = SourcePolicy(
        source_id=distractor.source_id,
        allowed_hosts=("notes.example.com",),
        allowed_mime_types=("text/plain",),
        allowed_license_ids=("Apache-2.0",),
        max_download_bytes=4096,
    )
    distractor_embedded = tuple(
        EmbeddedChunk(
            chunk,
            embedder.profile,
            checked_embedding(embedder, chunk.text),
        )
        for chunk in chunk_validated_artifact(validator.validate(distractor), ChunkingPolicy())
    )
    if any(len(item.values) != 384 for item in distractor_embedded):
        raise RuntimeError("navigation distractor chunk dimensions differ")
    data = arguments.work_root / "native"
    socket_path = arguments.work_root / "hyphae.sock"
    owner_key = arguments.work_root / "owner.key"
    _run_json([str(arguments.hyphae_binary), "init", "--data-dir", str(data)])
    daemon = _start_daemon(arguments.hyphae_binary, data, socket_path, arguments.out)
    collection = json.loads(
        _run(
            [
                sys.executable,
                str(Path(__file__).with_name("hyphae_210_collection.py")),
                str(socket_path),
                "--vector-dimensions",
                "384",
            ]
        )
    )
    if collection.get("collection_definition_sha256") != COLLECTION_SHA256:
        raise RuntimeError("navigation 384D collection definition differs")
    _stop_daemon(daemon, socket_path)
    daemon = None
    _run_json(
        [
            str(arguments.hyphae_binary),
            "search",
            "--data-dir",
            str(data),
            "provision",
            "--collection",
            "13",
        ]
    )
    _run_json(
        [
            str(arguments.hyphae_binary),
            "security",
            "--data-dir",
            str(data),
            "bootstrap",
            "--name",
            "navigation-canary-owner",
            "--label",
            "navigation-canary-v1",
            "--key-out",
            str(owner_key),
        ]
    )
    key = owner_key.read_text(encoding="ascii").strip()
    daemon = _start_daemon(
        arguments.hyphae_binary,
        data,
        socket_path,
        arguments.out,
        authenticated=True,
    )
    with HyphaeClient.local_authenticated(str(socket_path), key) as raw_client:
        owner_key.unlink(missing_ok=True)
        status = raw_client.admin("status")
        lineage = status.value["snapshot"]["directory_lineage"]
        backend_id = hashlib.sha256(lineage).hexdigest()
        client = BoundedRecordingClient(raw_client, RequestOptions)
        store = SQLiteTenantStore(arguments.work_root / "routing.sqlite3", tenant=TENANT)
        receipts = PublicationReceiptStore(arguments.work_root / "receipts")
        coordinator = KnowledgeCoordinator(
            sufficiency=policy,
            acquisition=AcquisitionPolicy(
                version="navigation-policy-v1",
                sources=(source_policy, distractor_policy),
            ),
            store=store,
            embedding_profile=embedder.profile,
        )
        pending = coordinator.answer_or_enqueue(
            tenant=TENANT,
            query=QUERY,
            evidence=EvidenceBundle(
                TENANT,
                hashlib.sha256(normalize_query(QUERY).encode()).hexdigest(),
                DISTRACTOR_GENERATION,
                (),
            ),
            source_id=distractor.source_id,
        )
        if pending.job_id is None:
            raise RuntimeError("navigation distractor job did not enqueue")
        ingestor = HyphaeShadowIngestor(
            tenant=TENANT,
            client=client,
            collection=13,
            vector_target="semantic",
            backend_id=backend_id,
            publish=True,
            receipt_store=receipts,
        )
        job = coordinator.job_status(TENANT, pending.job_id)
        if job is None or ingestor.target is None:
            raise RuntimeError("navigation job or target is absent")
        distractor_key = ingest_idempotency_key(
            job, distractor_embedded, embedder.profile, target=ingestor.target
        )
        distractor_manifest = GenerationManifest(
            tenant=TENANT,
            generation_id=DISTRACTOR_GENERATION,
            target=ingestor.target,
            parent_generation_id=None,
            chunk_ids=tuple(item.chunk.chunk_id for item in distractor_embedded),
            ingest_idempotency_keys=(distractor_key,),
            ingest_receipt_digests=(),
        )
        authority = GenerationAuthority(store, receipts=receipts)
        authority.register(distractor_manifest)
        worker = DurableAcquisitionWorker(
            worker_id="navigation-canary-acquisition",
            lease_seconds=120,
            coordinator=coordinator,
            connector=InMemorySourceConnector(
                {
                    (TENANT.value, artifact.source_id): artifact,
                    (TENANT.value, distractor.source_id): distractor,
                }
            ),
            embedder=embedder,
            ingestor=ingestor,
            verifier=ingestor,
            validator=validator,
            authorizer=DurablePublicationAuthorizer(
                tenant=TENANT,
                store=receipts,
                authority="navigation-canary-operator",
                enabled=True,
            ),
        )
        outcome = worker.run_next(job_id=pending.job_id)
        if outcome is None or outcome.receipt is None or outcome.job.status is not JobStatus.READY:
            raise RuntimeError("navigation distractor publication did not become ready")
        authority.verify_candidate(distractor_manifest, (outcome.receipt,))
        distractor_activation = authority.activate(
            DISTRACTOR_GENERATION,
            expected_revision=0,
            actor="navigation-canary-operator",
            reason="navigation canary distractor generation",
        )
        router = GenerationRoutedRetriever(
            tenant=TENANT,
            authority=authority,
            client=client,
            profile=HYPHAE_210_RETRIEVAL_PROFILE,
            embedder=embedder,
            request_options_factory=lambda timeout: RequestOptions(
                deadline_micros=time.time_ns() // 1000 + int(timeout * 1_000_000)
            ),
        )
        distractor_evidence = router.retrieve(TENANT, QUERY, timeout_seconds=30)
        if not distractor_evidence.hits:
            raise RuntimeError("navigation distractor retrieval returned no evidence")
        step0_decision = decide_navigation_step(
            backbone=backbone,
            pilot=pilot,
            query=QUERY,
            evidence=distractor_evidence,
            policy=policy,
            search_steps_used=0,
            device=device,
            present_evidence=False,
        )
        if step0_decision.action != "search":
            raise RuntimeError("navigation pilot did not request the required search")
        body_pending = coordinator.answer_or_enqueue(
            tenant=TENANT,
            query=QUERY,
            evidence=EvidenceBundle(
                TENANT,
                hashlib.sha256(normalize_query(QUERY).encode()).hexdigest(),
                GENERATION,
                (),
            ),
            source_id=artifact.source_id,
        )
        if body_pending.job_id is None:
            raise RuntimeError("navigation body job did not enqueue")
        body_job = coordinator.job_status(TENANT, body_pending.job_id)
        if body_job is None:
            raise RuntimeError("navigation body job is absent")
        key_id = ingest_idempotency_key(
            body_job, embedded, embedder.profile, target=ingestor.target
        )
        manifest = GenerationManifest(
            tenant=TENANT,
            generation_id=GENERATION,
            target=ingestor.target,
            parent_generation_id=DISTRACTOR_GENERATION,
            chunk_ids=tuple(item.chunk.chunk_id for item in embedded),
            ingest_idempotency_keys=(key_id,),
            ingest_receipt_digests=(),
        )
        authority.register(manifest)
        body_outcome = worker.run_next(job_id=body_pending.job_id)
        if (
            body_outcome is None
            or body_outcome.receipt is None
            or body_outcome.job.status is not JobStatus.READY
        ):
            raise RuntimeError("navigation live publication did not become ready")
        receipt = body_outcome.receipt
        prepared = store.prepared_ingest(TENANT, body_pending.job_id)
        if prepared is None:
            raise RuntimeError("navigation prepared ingest is absent")
    _stop_daemon(daemon, socket_path)
    daemon = _start_daemon(
        arguments.hyphae_binary,
        data,
        socket_path,
        arguments.out,
        authenticated=True,
    )
    with HyphaeClient.local_authenticated(str(socket_path), key) as raw_client:
        client = BoundedRecordingClient(raw_client, RequestOptions)
        replay_documents = [ingestor._document(item, GENERATION) for item in prepared.chunks]
        replay = client.search_ingest(
            13,
            {
                "idempotency_id": _u128_idempotency(prepared.idempotency_key),
                "documents": replay_documents,
            },
        )
        if replay.value.get("idempotent_replay") is not True:
            raise RuntimeError("navigation backend replay did not survive restart")
        recovered = PublicationReceiptStore(arguments.work_root / "receipts")
        durable = recovered.load_ingest(TENANT, receipt.idempotency_key)
        if durable != receipt:
            raise RuntimeError("navigation durable receipt did not survive restart")
        reopened = SQLiteTenantStore(arguments.work_root / "routing.sqlite3", tenant=TENANT)
        authority = GenerationAuthority(reopened, receipts=recovered)
        authority.verify_candidate(manifest, (receipt,))
        activation = authority.activate(
            GENERATION,
            expected_revision=1,
            actor="navigation-canary-operator",
            reason="real navigation canary bootstrap",
        )
        router = GenerationRoutedRetriever(
            tenant=TENANT,
            authority=authority,
            client=client,
            profile=HYPHAE_210_RETRIEVAL_PROFILE,
            embedder=embedder,
            request_options_factory=lambda timeout: RequestOptions(
                deadline_micros=time.time_ns() // 1000 + int(timeout * 1_000_000)
            ),
        )
        evidence = router.retrieve(TENANT, QUERY, timeout_seconds=30)
        if not evidence.hits or evidence.hits[0].text != BODY.decode():
            raise RuntimeError("navigation retrieval did not hydrate the expected passage")
        live_certificate = _certificate(evidence, policy)
        live_decision = decide_navigation_step(
            backbone=backbone,
            pilot=pilot,
            query=QUERY,
            evidence=evidence,
            policy=policy,
            search_steps_used=1,
            device=device,
        )
        branch = (client.last_search or {}).get("vector_branches", [{}])[0]
        raw_scores = [
            float(hit.get("score", 0)) for hit in (client.last_search or {}).get("hits", [])
        ]
        steps = [
            {
                "step": 0,
                "search_steps_used": 0,
                "evidence_handles": [hit.handle for hit in distractor_evidence.hits],
                "action": step0_decision.action,
                "selected_handles": list(step0_decision.selected_handles),
            },
            {
                "step": 1,
                "search_steps_used": 1,
                "evidence_handles": [hit.handle for hit in evidence.hits],
                "action": live_decision.action,
                "selected_handles": list(live_decision.selected_handles),
            },
        ]
        passed = (
            step0_decision.action == "search"
            and not step0_decision.selected_handles
            and live_decision.action == "answer"
            and tuple(live_decision.selected_handles) == tuple(hit.handle for hit in evidence.hits)
        )
        report = {
            "schema": "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v1",
            "completed": True,
            "passed": passed,
            "failure": None,
            "source_revision": arguments.source_revision,
            "source_patch_sha256": arguments.source_patch_sha256,
            "dependencies": {
                **identities,
                "navigation_checkpoint_sha256": NAVIGATION_CHECKPOINT_SHA256,
                "minilm_artifact_manifest_sha256": embedder.artifact_manifest_sha256(),
                "gemma_artifact_manifest_sha256": (
                    Gemma4E4BFrozenBackbone.artifact_manifest_digest()
                ),
            },
            "native": {
                "version": binary_version,
                "sdk_version": hyphae_sdk.__version__,
                "protocol": [PROTOCOL_MAJOR, PROTOCOL_MINOR],
                "backend_id": backend_id,
                "collection_definition_sha256": COLLECTION_SHA256,
                "durability": "strict",
                "strategy": branch.get("strategy"),
                "approximate": (client.last_search or {}).get("approximate"),
                "exact_reranked": branch.get("exact_reranked"),
                "restart_replay": replay.value.get("idempotent_replay"),
                "raw_scores": raw_scores,
            },
            "embedding": minilm,
            "publication": _json_value(asdict(receipt)),
            "generation": {
                "manifest_digest": manifest.manifest_digest,
                "distractor_manifest_digest": distractor_manifest.manifest_digest,
                "distractor_activation": _json_value(asdict(distractor_activation)),
                "activation": _json_value(asdict(activation)),
                "snapshot": _json_value(asdict(authority.snapshot())),
            },
            "pilot": {
                "maximum_evidence_items": maximum_evidence_items,
                "distractor_certificate": list(_certificate(distractor_evidence, policy)),
                "live_certificate": list(live_certificate),
                "steps": steps,
            },
            "backbone_unchanged": True,
        }
        return report, daemon


def _certificate(evidence: EvidenceBundle, policy: SufficiencyPolicy) -> tuple[float, ...]:
    return host_control_values(evidence, policy, HOST_CONTROL_CONTRACT)


def _start_daemon(
    binary: Path,
    data: Path,
    socket_path: Path,
    evidence: Path,
    *,
    authenticated: bool = False,
) -> subprocess.Popen[str]:
    global _ACTIVE_DAEMON
    socket_path.unlink(missing_ok=True)
    command = [str(binary), "serve", "--data-dir", str(data), "--endpoint", str(socket_path)]
    if authenticated:
        command.append("--native-api-key-auth")
    process = subprocess.Popen(
        command,
        stdout=(evidence / "hyphae-daemon.stdout.log").open("a"),
        stderr=(evidence / "hyphae-daemon.stderr.log").open("a"),
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if socket_path.is_socket():
            _ACTIVE_DAEMON = process
            return process
        if process.poll() is not None:
            raise RuntimeError("navigation Hyphae daemon exited before readiness")
        time.sleep(0.1)
    _stop_daemon(process, socket_path)
    raise TimeoutError("navigation Hyphae daemon did not become ready")


def _stop_daemon(process: subprocess.Popen[str], socket_path: Path) -> None:
    global _ACTIVE_DAEMON
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    socket_path.unlink(missing_ok=True)
    if _ACTIVE_DAEMON is process:
        _ACTIVE_DAEMON = None


def _require_digest(path: Path, expected: str) -> str:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"navigation artifact digest differs: {path.name}")
    return expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    return subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _run_json(command: list[str]) -> dict[str, object]:
    value = json.loads(_run(command))
    if not isinstance(value, dict):
        raise RuntimeError("navigation CLI response is not an object")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value.value if hasattr(value, "value") else value


if __name__ == "__main__":
    raise SystemExit(main())
