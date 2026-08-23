"""Tenant-local generation manifests, atomic routing cutover, and rollback."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from celiums_rezero.knowledge.live import _receipt_digest_matches
from celiums_rezero.knowledge.publication import PublicationReceiptStore
from celiums_rezero.knowledge.schemas import (
    GenerationChangeReceipt,
    GenerationManifest,
    GenerationSnapshot,
    IngestReceipt,
)
from celiums_rezero.knowledge.store import SQLiteTenantStore
from celiums_rezero.lab.serialization import canonical_json


@dataclass(frozen=True, slots=True)
class GenerationAuthority:
    store: SQLiteTenantStore
    receipts: PublicationReceiptStore | None = None

    def register(self, manifest: GenerationManifest) -> None:
        if manifest.tenant != self.store.tenant:
            raise PermissionError("generation manifest belongs to another tenant")
        self.store.register_generation(manifest)

    def verify_candidate(
        self,
        manifest: GenerationManifest,
        receipts: tuple[IngestReceipt, ...],
    ) -> None:
        if not receipts:
            raise ValueError("generation verification requires ingest receipts")
        if self.receipts is None:
            raise PermissionError("generation verification requires durable receipt authority")
        observed_chunks = tuple(
            chunk_id for receipt in receipts for chunk_id in receipt.chunk_ids
        )
        if len(observed_chunks) != len(set(observed_chunks)) or set(observed_chunks) != set(
            manifest.chunk_ids
        ):
            raise ValueError("generation receipts do not exactly cover the manifest")
        if any(
            receipt.tenant != manifest.tenant
            or receipt.corpus_generation != manifest.generation_id
            or receipt.target != manifest.target
            or not receipt.published
            or not _receipt_digest_matches(receipt)
            for receipt in receipts
        ):
            raise ValueError("generation receipt binding is invalid")
        if set(receipt.idempotency_key for receipt in receipts) != set(
            manifest.ingest_idempotency_keys
        ):
            raise ValueError("generation ingest identities do not match the manifest")
        for receipt in receipts:
            durable = self.receipts.load_ingest(
                manifest.tenant, receipt.idempotency_key
            )
            if durable != receipt:
                raise ValueError("generation receipt is not the durable publication receipt")
        receipt_digests = tuple(
            receipt.backend_receipt_digest for receipt in receipts
        )
        if any(value is None for value in receipt_digests):
            raise ValueError("generation receipt digest set is incomplete")
        completed = GenerationManifest(
            tenant=manifest.tenant,
            generation_id=manifest.generation_id,
            target=manifest.target,
            parent_generation_id=manifest.parent_generation_id,
            chunk_ids=manifest.chunk_ids,
            ingest_idempotency_keys=manifest.ingest_idempotency_keys,
            ingest_receipt_digests=tuple(
                value for value in receipt_digests if value is not None
            ),
            verification_token=hashlib.sha256(
                b"generation-verification-v1\0"
                + (manifest.manifest_digest or "").encode()
                + b"".join(
                    value.encode() for value in receipt_digests if value is not None
                )
            ).hexdigest(),
        )
        self.store._complete_generation_manifest(manifest, completed)

    def snapshot(self) -> GenerationSnapshot:
        return self.store.generation_snapshot(self.store.tenant)

    def activate(
        self,
        generation_id: str,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> GenerationChangeReceipt:
        return self.store.activate_generation(
            generation_id,
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
        )

    def pause(self, *, expected_revision: int, actor: str) -> GenerationChangeReceipt:
        return self.store.pause_generation(expected_revision=expected_revision, actor=actor)

    def resume(self, *, expected_revision: int, actor: str) -> GenerationChangeReceipt:
        return self.store.resume_generation(expected_revision=expected_revision, actor=actor)

    def rollback(
        self,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> GenerationChangeReceipt:
        return self.store.rollback_generation(
            expected_revision=expected_revision,
            actor=actor,
            reason=reason,
        )


def generation_metrics(snapshot: GenerationSnapshot) -> str:
    paused = 1 if snapshot.claims_paused else 0
    return (
        "# HELP celiums_rezero_generation_revision Active routing revision.\n"
        "# TYPE celiums_rezero_generation_revision gauge\n"
        f"celiums_rezero_generation_revision {snapshot.revision}\n"
        "# HELP celiums_rezero_generation_claims_paused Whether generation claims are paused.\n"
        "# TYPE celiums_rezero_generation_claims_paused gauge\n"
        f"celiums_rezero_generation_claims_paused {paused}\n"
    )


def manifest_json(manifest: GenerationManifest) -> str:
    return canonical_json({"schema": "knowledge-generation-manifest-v1", "value": manifest})
