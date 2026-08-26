#!/usr/bin/env python3
"""Run the live Hyphae depth-3 navigation canary with the calibrated ReZero pilot v3."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hyphae_minilm_gemma_navigation_canary as base

from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.navigation_experiment import decide_navigation_step
from celiums_rezero.knowledge import (
    HYPHAE_210_RETRIEVAL_PROFILE,
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
)
from celiums_rezero.knowledge.acquisition import (
    ChunkingPolicy,
    chunk_validated_artifact,
    ingest_idempotency_key,
)
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.embedding import checked_embedding
from celiums_rezero.knowledge.live import _u128_idempotency
from celiums_rezero.knowledge.schemas import (
    AcquisitionJob,
    EmbeddedChunk,
    GenerationManifest,
    SourceArtifact,
    SourcePolicy,
)

DISTRACTOR = base.DISTRACTOR
NEUTRAL = (
    b"Granite sample 7B: the quarry ledger lists hardness six and a pale grain. "
    b"Basalt core 4C records density near three grams per cubic centimetre."
)
CONTINUATION_GENERATION = "generation_navigation_canary_v3_continuation"
REPORT_SCHEMA_V3 = "hyphae-transformer.hyphae-minilm-gemma-navigation-canary/v3"


def run_depth3(arguments: Any) -> tuple[dict[str, object], subprocess.Popen[str] | None]:
    failure_reasons: list[str] = []
    identities = {
        "hyphae_archive_sha256": base._require_digest(
            arguments.hyphae_archive, base.HYPHAE_ARCHIVE_SHA256
        ),
        "hyphae_binary_sha256": base._require_digest(
            arguments.hyphae_binary, base.HYPHAE_BINARY_SHA256
        ),
        "hyphae_wheel_sha256": base._require_digest(
            arguments.hyphae_wheel, base.HYPHAE_WHEEL_SHA256
        ),
        "navigation_bundle_sha256": base._require_digest(
            arguments.pilot, base.NAVIGATION_BUNDLE_SHA256
        ),
    }
    checkpoint_bytes = subprocess.run(
        ["tar", "-xOzf", str(arguments.pilot), "navigation-control.pt"],
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    if hashlib.sha256(checkpoint_bytes).hexdigest() != base.NAVIGATION_CHECKPOINT_SHA256:
        raise ValueError("navigation checkpoint digest differs")
    import torch

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
    binary_version = base._run_json([str(arguments.hyphae_binary), "version", "--json"])
    if binary_version.get("engine_version") != "2.1.0":
        raise RuntimeError("navigation Hyphae binary version differs")
    from celiums_rezero.knowledge.embedding import (
        MiniLML6V2EmbeddingProvider,
        minilm_l6_v2_preflight,
    )

    embedder = MiniLML6V2EmbeddingProvider(arguments.minilm_model)
    minilm = minilm_l6_v2_preflight(arguments.minilm_model)
    base._write_json(arguments.out / "minilm-preflight.json", minilm)
    policy = SufficiencyPolicy()
    backbone = Gemma4E4BFrozenBackbone(arguments.gemma_model, device="cuda:0")
    device = torch.device("cuda:0")
    checkpoint_path = arguments.work_root / "navigation-control.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    from celiums_rezero.governed.navigation_experiment import load_navigation_pilot

    pilot = load_navigation_pilot(
        checkpoint_path,
        hidden_size=backbone.identity.hidden_size,
        device=device,
    )
    validator = StrictArtifactValidator()
    digest_query = hashlib.sha256(normalize_query(base.QUERY).encode()).hexdigest()

    artifact = SourceArtifact(
        tenant=base.TENANT,
        source_id="official_docs",
        source_version="navigation-v3",
        body=base.BODY,
        content_type="text/plain",
        license_id="Apache-2.0",
        content_digest=hashlib.sha256(base.BODY).hexdigest(),
    )
    distractor = SourceArtifact(
        tenant=base.TENANT,
        source_id="office_notes",
        source_version="navigation-v3",
        body=DISTRACTOR,
        content_type="text/plain",
        license_id="Apache-2.0",
        content_digest=hashlib.sha256(DISTRACTOR).hexdigest(),
    )
    neutral = SourceArtifact(
        tenant=base.TENANT,
        source_id="quarry_notes",
        source_version="navigation-v3",
        body=NEUTRAL,
        content_type="text/plain",
        license_id="Apache-2.0",
        content_digest=hashlib.sha256(NEUTRAL).hexdigest(),
    )
    policies = (
        SourcePolicy(
            source_id="official_docs",
            allowed_hosts=("docs.example.com",),
            allowed_mime_types=("text/plain",),
            allowed_license_ids=("Apache-2.0",),
            max_download_bytes=4096,
        ),
        SourcePolicy(
            source_id="office_notes",
            allowed_hosts=("notes.example.com",),
            allowed_mime_types=("text/plain",),
            allowed_license_ids=("Apache-2.0",),
            max_download_bytes=4096,
        ),
        SourcePolicy(
            source_id="quarry_notes",
            allowed_hosts=("quarry.example.com",),
            allowed_mime_types=("text/plain",),
            allowed_license_ids=("Apache-2.0",),
            max_download_bytes=4096,
        ),
    )

    def embed(source: SourceArtifact) -> tuple[EmbeddedChunk, ...]:
        return tuple(
            EmbeddedChunk(
                chunk,
                embedder.profile,
                checked_embedding(embedder, chunk.text),
            )
            for chunk in chunk_validated_artifact(validator.validate(source), ChunkingPolicy())
        )

    body_embedded = embed(artifact)
    distractor_embedded = embed(distractor)
    neutral_embedded = embed(neutral)
    for items in (body_embedded, distractor_embedded, neutral_embedded):
        if any(len(item.values) != 384 for item in items):
            raise RuntimeError("navigation v3 chunk dimensions differ")

    store = SQLiteTenantStore(arguments.work_root / "routing.sqlite3", tenant=base.TENANT)
    receipts = PublicationReceiptStore(arguments.work_root / "receipts")
    coordinator = KnowledgeCoordinator(
        sufficiency=policy,
        acquisition=base.AcquisitionPolicy(version="navigation-policy-v3", sources=policies),
        store=store,
        embedding_profile=embedder.profile,
    )
    authority = GenerationAuthority(store, receipts=receipts)
    connector = InMemorySourceConnector(
        {
            (base.TENANT.value, "official_docs"): artifact,
            (base.TENANT.value, "office_notes"): distractor,
            (base.TENANT.value, "quarry_notes"): neutral,
        }
    )
    authorizer = DurablePublicationAuthorizer(
        tenant=base.TENANT,
        store=receipts,
        authority="navigation-canary-operator",
        enabled=True,
    )

    def backend(name: str, label: str) -> dict[str, object]:
        data = arguments.work_root / name
        socket_path = arguments.work_root / (name + ".sock")
        owner_key = arguments.work_root / (name + ".key")
        base._run_json([str(arguments.hyphae_binary), "init", "--data-dir", str(data)])
        daemon = base._start_daemon(arguments.hyphae_binary, data, socket_path, arguments.out)
        definition = json.loads(
            base._run(
                [
                    sys.executable,
                    str(Path(base.__file__).with_name("hyphae_210_collection.py")),
                    str(socket_path),
                    "--vector-dimensions",
                    "384",
                ]
            )
        )
        if definition.get("collection_definition_sha256") != base.COLLECTION_SHA256:
            raise RuntimeError("navigation 384D collection definition differs")
        base._stop_daemon(daemon, socket_path)
        base._run_json(
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
        base._run_json(
            [
                str(arguments.hyphae_binary),
                "security",
                "--data-dir",
                str(data),
                "bootstrap",
                "--name",
                name,
                "--label",
                label,
                "--key-out",
                str(owner_key),
            ]
        )
        key = owner_key.read_text(encoding="ascii").strip()
        daemon = base._start_daemon(
            arguments.hyphae_binary,
            data,
            socket_path,
            arguments.out,
            authenticated=True,
        )
        raw_client = HyphaeClient.local_authenticated(str(socket_path), key)
        owner_key.unlink(missing_ok=True)
        status = raw_client.admin("status")
        lineage = status.value["snapshot"]["directory_lineage"]
        return {
            "data": data,
            "socket_path": socket_path,
            "key": key,
            "daemon": daemon,
            "raw_client": raw_client,
            "backend_id": hashlib.sha256(lineage).hexdigest(),
        }

    daemon = None
    first = backend("native", "navigation-canary-v3")
    daemon = first["daemon"]
    with first["raw_client"] as raw_client:
        client = base.BoundedRecordingClient(raw_client, RequestOptions)
        backend_id = str(first["backend_id"])
        pending = coordinator.answer_or_enqueue(
            tenant=base.TENANT,
            query=base.QUERY,
            evidence=EvidenceBundle(base.TENANT, digest_query, base.DISTRACTOR_GENERATION, ()),
            source_id=distractor.source_id,
        )
        if pending.job_id is None:
            raise RuntimeError("navigation distractor job did not enqueue")
        ingestor = HyphaeShadowIngestor(
            tenant=base.TENANT,
            client=client,
            collection=13,
            vector_target="semantic",
            backend_id=backend_id,
            publish=True,
            receipt_store=receipts,
        )
        job = coordinator.job_status(base.TENANT, pending.job_id)
        if job is None or ingestor.target is None:
            raise RuntimeError("navigation job or target is absent")
        distractor_key = ingest_idempotency_key(
            job, distractor_embedded, embedder.profile, target=ingestor.target
        )
        distractor_manifest = GenerationManifest(
            tenant=base.TENANT,
            generation_id=base.DISTRACTOR_GENERATION,
            target=ingestor.target,
            parent_generation_id=None,
            chunk_ids=tuple(item.chunk.chunk_id for item in distractor_embedded),
            ingest_idempotency_keys=(distractor_key,),
            ingest_receipt_digests=(),
        )
        authority.register(distractor_manifest)
        worker = DurableAcquisitionWorker(
            worker_id="navigation-canary-v3-acquisition",
            lease_seconds=120,
            coordinator=coordinator,
            connector=connector,
            embedder=embedder,
            ingestor=ingestor,
            verifier=ingestor,
            validator=validator,
            authorizer=authorizer,
        )
        outcome = worker.run_next(job_id=pending.job_id)
        if outcome is None or outcome.receipt is None or outcome.job.status is not JobStatus.READY:
            raise RuntimeError("navigation distractor publication did not become ready")
        authority.verify_candidate(distractor_manifest, (outcome.receipt,))
        distractor_activation = authority.activate(
            base.DISTRACTOR_GENERATION,
            expected_revision=0,
            actor="navigation-canary-operator",
            reason="navigation canary v3 distractor generation",
        )
        router = GenerationRoutedRetriever(
            tenant=base.TENANT,
            authority=authority,
            client=client,
            profile=HYPHAE_210_RETRIEVAL_PROFILE,
            embedder=embedder,
            request_options_factory=lambda timeout: RequestOptions(
                deadline_micros=time.time_ns() // 1000 + int(timeout * 1_000_000)
            ),
        )
        distractor_evidence = router.retrieve(base.TENANT, base.QUERY, timeout_seconds=30)
        if len(distractor_evidence.hits) != 2:
            raise RuntimeError("navigation v3 distractor retrieval expected two hits")
        step0 = decide_navigation_step(
            backbone=backbone,
            pilot=pilot,
            query=base.QUERY,
            evidence=distractor_evidence,
            policy=policy,
            search_steps_used=0,
            device=device,
            present_evidence=False,
        )
        if step0.action != "search":
            failure_reasons.append("step0 action is not search")
        if step0.selected_handles:
            failure_reasons.append("step0 selected handles despite search contract")
        claimed = store.claim_finalization(
            owner_id="navigation-canary-v3-finalizer",
            lease_seconds=60,
            job_id=pending.job_id,
        )
        if claimed is None:
            raise RuntimeError("navigation distractor finalization is unclaimable")
        store.complete_insufficient(
            claimed[1], failure="distractor generation cannot support the query"
        )
    base._stop_daemon(first["daemon"], first["socket_path"])
    daemon = None

    second = backend("native2", "navigation-canary-v3-continuation")
    daemon = second["daemon"]
    with second["raw_client"] as raw_client2:
        client2 = base.BoundedRecordingClient(raw_client2, RequestOptions)
        backend_id2 = str(second["backend_id"])
        cont_pending = coordinator.answer_or_enqueue(
            tenant=base.TENANT,
            query=base.QUERY,
            evidence=EvidenceBundle(base.TENANT, digest_query, CONTINUATION_GENERATION, ()),
            source_id=neutral.source_id,
        )
        if cont_pending.job_id is None:
            raise RuntimeError("navigation continuation job did not enqueue")
        ingestor2 = HyphaeShadowIngestor(
            tenant=base.TENANT,
            client=client2,
            collection=13,
            vector_target="semantic",
            backend_id=backend_id2,
            publish=True,
            receipt_store=receipts,
        )
        cont_job = coordinator.job_status(base.TENANT, cont_pending.job_id)
        if cont_job is None or ingestor2.target is None:
            raise RuntimeError("navigation continuation job or target is absent")
        continuation_embedded = (distractor_embedded[0], *neutral_embedded)
        cont_key = ingest_idempotency_key(
            cont_job, continuation_embedded, embedder.profile, target=ingestor2.target
        )
        cont_manifest = GenerationManifest(
            tenant=base.TENANT,
            generation_id=CONTINUATION_GENERATION,
            target=ingestor2.target,
            parent_generation_id=None,
            chunk_ids=tuple(item.chunk.chunk_id for item in continuation_embedded),
            ingest_idempotency_keys=(cont_key,),
            ingest_receipt_digests=(),
        )
        authority.register(cont_manifest)
        worker2 = DurableAcquisitionWorker(
            worker_id="navigation-canary-v3-continuation",
            lease_seconds=120,
            coordinator=coordinator,
            connector=connector,
            embedder=embedder,
            ingestor=ingestor2,
            verifier=ingestor2,
            validator=validator,
            authorizer=authorizer,
        )
        cont_outcome = worker2.run_next(job_id=cont_pending.job_id)
        if (
            cont_outcome is None
            or cont_outcome.receipt is None
            or cont_outcome.job.status is not JobStatus.READY
        ):
            raise RuntimeError("navigation continuation publication did not become ready")
        authority.verify_candidate(cont_manifest, (cont_outcome.receipt,))
        cont_activation = authority.activate(
            CONTINUATION_GENERATION,
            expected_revision=1,
            actor="navigation-canary-operator",
            reason="navigation canary v3 continuation generation",
        )
        router2 = GenerationRoutedRetriever(
            tenant=base.TENANT,
            authority=authority,
            client=client2,
            profile=HYPHAE_210_RETRIEVAL_PROFILE,
            embedder=embedder,
            request_options_factory=lambda timeout: RequestOptions(
                deadline_micros=time.time_ns() // 1000 + int(timeout * 1_000_000)
            ),
        )
        cont_evidence = router2.retrieve(base.TENANT, base.QUERY, timeout_seconds=30)
        if len(cont_evidence.hits) != 2:
            raise RuntimeError("navigation v3 continuation retrieval expected two hits")
        step1 = decide_navigation_step(
            backbone=backbone,
            pilot=pilot,
            query=base.QUERY,
            evidence=cont_evidence,
            policy=policy,
            search_steps_used=1,
            device=device,
            present_evidence=False,
        )
        if step1.action != "search":
            failure_reasons.append("step1 action is not search")
        if step1.selected_handles:
            failure_reasons.append("step1 selected handles despite search contract")
        claimed2 = store.claim_finalization(
            owner_id="navigation-canary-v3-finalizer2",
            lease_seconds=60,
            job_id=cont_pending.job_id,
        )
        if claimed2 is None:
            raise RuntimeError("navigation continuation finalization is unclaimable")
        store.complete_insufficient(
            claimed2[1], failure="continuation generation cannot support the query"
        )
    base._stop_daemon(second["daemon"], second["socket_path"])
    daemon = None

    third = backend("native3", "navigation-canary-v3-body")
    daemon = third["daemon"]
    with third["raw_client"] as raw_client3:
        client3 = base.BoundedRecordingClient(raw_client3, RequestOptions)
        backend_id3 = str(third["backend_id"])
        body_pending = coordinator.answer_or_enqueue(
            tenant=base.TENANT,
            query=base.QUERY,
            evidence=EvidenceBundle(base.TENANT, digest_query, base.GENERATION, ()),
            source_id=artifact.source_id,
        )
        if body_pending.job_id is None:
            raise RuntimeError("navigation body job did not enqueue")
        ingestor3 = HyphaeShadowIngestor(
            tenant=base.TENANT,
            client=client3,
            collection=13,
            vector_target="semantic",
            backend_id=backend_id3,
            publish=True,
            receipt_store=receipts,
        )
        ephemeral_job = AcquisitionJob(
            tenant=base.TENANT,
            query=normalize_query(base.QUERY),
            query_digest=digest_query,
            corpus_generation=base.GENERATION,
            policy_version="navigation-policy-v3",
            embedding_profile=embedder.profile,
            source_id=artifact.source_id,
        )
        key_id = ingest_idempotency_key(
            ephemeral_job, body_embedded, embedder.profile, target=ingestor3.target
        )
        manifest = GenerationManifest(
            tenant=base.TENANT,
            generation_id=base.GENERATION,
            target=ingestor3.target,
            parent_generation_id=None,
            chunk_ids=tuple(item.chunk.chunk_id for item in body_embedded),
            ingest_idempotency_keys=(key_id,),
            ingest_receipt_digests=(),
        )
        authority.register(manifest)
        worker3 = DurableAcquisitionWorker(
            worker_id="navigation-canary-v3-body",
            lease_seconds=120,
            coordinator=coordinator,
            connector=connector,
            embedder=embedder,
            ingestor=ingestor3,
            verifier=ingestor3,
            validator=validator,
            authorizer=authorizer,
        )
        body_outcome = worker3.run_next(job_id=body_pending.job_id)
        if (
            body_outcome is None
            or body_outcome.receipt is None
            or body_outcome.job.status is not JobStatus.READY
        ):
            raise RuntimeError("navigation live publication did not become ready")
        receipt = body_outcome.receipt
        prepared = store.prepared_ingest(base.TENANT, body_pending.job_id)
        if prepared is None:
            raise RuntimeError("navigation prepared ingest is absent")
    base._stop_daemon(third["daemon"], third["socket_path"])
    daemon = base._start_daemon(
        arguments.hyphae_binary,
        third["data"],
        third["socket_path"],
        arguments.out,
        authenticated=True,
    )
    with HyphaeClient.local_authenticated(str(third["socket_path"]), third["key"]) as raw_client3:
        client3 = base.BoundedRecordingClient(raw_client3, RequestOptions)
        replay_documents = [ingestor3._document(item, base.GENERATION) for item in prepared.chunks]
        replay = client3.search_ingest(
            13,
            {
                "idempotency_id": _u128_idempotency(prepared.idempotency_key),
                "documents": replay_documents,
            },
        )
        if replay.value.get("idempotent_replay") is not True:
            raise RuntimeError("navigation backend replay did not survive restart")
        recovered = PublicationReceiptStore(arguments.work_root / "receipts")
        durable = recovered.load_ingest(base.TENANT, receipt.idempotency_key)
        if durable != receipt:
            raise RuntimeError("navigation durable receipt did not survive restart")
        reopened = SQLiteTenantStore(arguments.work_root / "routing.sqlite3", tenant=base.TENANT)
        authority = GenerationAuthority(reopened, receipts=recovered)
        authority.verify_candidate(manifest, (receipt,))
        pause_receipt = authority.pause(
            expected_revision=authority.snapshot().revision,
            actor="navigation-canary-operator",
        )
        activation = authority.activate(
            base.GENERATION,
            expected_revision=pause_receipt.resulting_revision,
            actor="navigation-canary-operator",
            reason="real navigation canary v3 bootstrap",
        )
        authority.resume(
            expected_revision=activation.resulting_revision,
            actor="navigation-canary-operator",
        )
        router3 = GenerationRoutedRetriever(
            tenant=base.TENANT,
            authority=authority,
            client=client3,
            profile=HYPHAE_210_RETRIEVAL_PROFILE,
            embedder=embedder,
            request_options_factory=lambda timeout: RequestOptions(
                deadline_micros=time.time_ns() // 1000 + int(timeout * 1_000_000)
            ),
        )
        evidence = router3.retrieve(base.TENANT, base.QUERY, timeout_seconds=30)
        if not evidence.hits or evidence.hits[0].text != base.BODY.decode():
            raise RuntimeError("navigation retrieval did not hydrate the expected passage")
        step2 = decide_navigation_step(
            backbone=backbone,
            pilot=pilot,
            query=base.QUERY,
            evidence=evidence,
            policy=policy,
            search_steps_used=2,
            device=device,
        )
        branch = (client3.last_search or {}).get("vector_branches", [{}])[0]
        raw_scores = [
            float(hit.get("score", 0)) for hit in (client3.last_search or {}).get("hits", [])
        ]
        steps = [
            {
                "step": 0,
                "search_steps_used": 0,
                "evidence_handles": [hit.handle for hit in distractor_evidence.hits],
                "action": step0.action,
                "selected_handles": list(step0.selected_handles),
                "action_logits": list(step0.action_logits),
            },
            {
                "step": 1,
                "search_steps_used": 1,
                "evidence_handles": [hit.handle for hit in cont_evidence.hits],
                "action": step1.action,
                "selected_handles": list(step1.selected_handles),
                "action_logits": list(step1.action_logits),
            },
            {
                "step": 2,
                "search_steps_used": 2,
                "evidence_handles": [hit.handle for hit in evidence.hits],
                "action": step2.action,
                "selected_handles": list(step2.selected_handles),
                "action_logits": list(step2.action_logits),
            },
        ]
        if step2.action != "answer":
            failure_reasons.append("live action is not answer")
        if tuple(step2.selected_handles) != tuple(hit.handle for hit in evidence.hits):
            failure_reasons.append("live selected handles differ from retrieval")
        passed = (
            step0.action == "search"
            and not step0.selected_handles
            and step1.action == "search"
            and not step1.selected_handles
            and step2.action == "answer"
            and tuple(step2.selected_handles) == tuple(hit.handle for hit in evidence.hits)
        )
        report = {
            "schema": REPORT_SCHEMA_V3,
            "completed": True,
            "passed": passed,
            "failure": None if passed else "; ".join(failure_reasons),
            "source_revision": arguments.source_revision,
            "source_patch_sha256": arguments.source_patch_sha256,
            "dependencies": {
                **identities,
                "navigation_checkpoint_sha256": base.NAVIGATION_CHECKPOINT_SHA256,
                "minilm_artifact_manifest_sha256": embedder.artifact_manifest_sha256(),
                "gemma_artifact_manifest_sha256": (
                    Gemma4E4BFrozenBackbone.artifact_manifest_digest()
                ),
            },
            "native": {
                "version": binary_version,
                "sdk_version": hyphae_sdk.__version__,
                "protocol": [PROTOCOL_MAJOR, PROTOCOL_MINOR],
                "backend_id": backend_id3,
                "distractor_backend_id": backend_id,
                "continuation_backend_id": backend_id2,
                "collection_definition_sha256": base.COLLECTION_SHA256,
                "durability": "strict",
                "strategy": branch.get("strategy"),
                "approximate": (client3.last_search or {}).get("approximate"),
                "exact_reranked": branch.get("exact_reranked"),
                "restart_replay": replay.value.get("idempotent_replay"),
                "raw_scores": raw_scores,
            },
            "embedding": minilm,
            "publication": base._json_value(asdict(receipt)),
            "generation": {
                "manifest_digest": manifest.manifest_digest,
                "distractor_manifest_digest": distractor_manifest.manifest_digest,
                "continuation_manifest_digest": cont_manifest.manifest_digest,
                "distractor_activation": base._json_value(asdict(distractor_activation)),
                "continuation_activation": base._json_value(asdict(cont_activation)),
                "activation": base._json_value(asdict(activation)),
                "snapshot": base._json_value(asdict(authority.snapshot())),
            },
            "pilot": {
                "maximum_evidence_items": maximum_evidence_items,
                "distractor_certificate": list(base._certificate(distractor_evidence, policy)),
                "continuation_certificate": list(base._certificate(cont_evidence, policy)),
                "live_certificate": list(base._certificate(evidence, policy)),
                "steps": steps,
            },
            "backbone_unchanged": True,
        }
        return report, daemon


def main() -> int:
    base.REPORT_SCHEMA = REPORT_SCHEMA_V3
    arguments = base._parse_args()
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
    base._write_json(
        arguments.out / "navigation-campaign-report.json",
        {"schema": REPORT_SCHEMA_V3, "completed": False},
    )
    daemon = None
    try:
        report, daemon = run_depth3(arguments)
    except Exception:
        report = {
            "schema": REPORT_SCHEMA_V3,
            "completed": False,
            "passed": False,
            "failure": "navigation v3 canary failed",
            "source_revision": arguments.source_revision,
        }
        raise
    finally:
        active = daemon if daemon is not None else base._ACTIVE_DAEMON
        if active is not None:
            base._stop_daemon(active, arguments.work_root / "native.sock")
            base._stop_daemon(active, arguments.work_root / "native2.sock")
            base._stop_daemon(active, arguments.work_root / "native3.sock")
        shutil.rmtree(arguments.work_root, ignore_errors=True)
        report["work_root_removed"] = not arguments.work_root.exists()
        base._write_json(arguments.out / "navigation-campaign-report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
