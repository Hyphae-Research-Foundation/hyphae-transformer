"""Strict deployment bundles and host-authoritative shadow control."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import torch

from celiums_rezero.governed.backbone import FrozenTextBackbone, validate_frozen_features
from celiums_rezero.governed.model import GovernedControlHead, decode_control
from celiums_rezero.governed.schemas import ControlAction
from celiums_rezero.knowledge.operations import AuditChain
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyDecision,
)
from celiums_rezero.lab.serialization import canonical_json

BUNDLE_SCHEMA = "hyphae-transformer.governed-control-bundle/v1"
SHADOW_SCHEMA = "hyphae-transformer.governed-control-shadow/v1"
ACTION_ORDER = ("answer", "request_evidence", "abstain")


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DeploymentBundleManifest:
    schema: str
    bundle_id: str
    source_revision: str
    model_id: str
    model_revision: str
    feature_contract: str
    dataset_id: str
    seed: int
    maximum_evidence_items: int
    pointer_rank: int
    normalized_features: bool
    use_evidence_scores: bool
    use_host_control_features: bool
    pointer_policy_score: float | None
    pointer_policy_scale: float
    pointer_threshold: float
    minimum_confidence: float
    action_order: tuple[str, ...]
    artifacts: tuple[ArtifactDigest, ...]


@dataclass(frozen=True, slots=True)
class ShadowControlResult:
    schema: str
    host_decision: SufficiencyDecision
    predicted_action: ControlAction
    selected_handles: tuple[str, ...]
    action_confidence: float
    divergent: bool
    bundle_id: str


class ShadowObserver(Protocol):
    def observe(
        self,
        *,
        query: str,
        evidence: EvidenceBundle,
        host_decision: SufficiencyDecision,
    ) -> ShadowControlResult: ...


def build_deployment_bundle(
    *,
    output: Path,
    checkpoint: Path,
    training_report: Path,
    preregistration: Path,
    dataset_manifest: Path,
    source_revision: str,
    seed: int = 17,
) -> DeploymentBundleManifest:
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("deployment source revision must be a full lowercase Git SHA")
    report = json.loads(training_report.read_text())
    prereg = json.loads(preregistration.read_text())
    dataset = json.loads(dataset_manifest.read_text())
    seed_report = next((item for item in report["seeds"] if item["seed"] == seed), None)
    if (
        report.get("completed") is not True
        or report.get("passed") is not True
        or seed_report is None
        or seed_report.get("passed") is not True
    ):
        raise ValueError("deployment requires a completed passing canonical seed")
    if _sha256(checkpoint) != seed_report["training"]["checkpoint_sha256"]:
        raise ValueError("canonical checkpoint digest does not match its training report")
    training = prereg["training"]
    if dataset.get("dataset_id") != report.get("dataset_id") or dataset.get(
        "dataset_id"
    ) != prereg["dataset"]["governed_dataset_id"]:
        raise ValueError("deployment dataset identities do not match")
    artifact_sources = {
        "control-head.pt": checkpoint,
        "training-report.json": training_report,
        "preregistration.json": preregistration,
        "dataset-manifest.json": dataset_manifest,
    }
    artifacts = tuple(
        ArtifactDigest(name, path.stat().st_size, _sha256(path))
        for name, path in sorted(artifact_sources.items())
    )
    manifest = DeploymentBundleManifest(
        schema=BUNDLE_SCHEMA,
        bundle_id="",
        source_revision=source_revision,
        model_id=str(report["model_id"]),
        model_revision=str(report["model_revision"]),
        feature_contract=str(prereg["backbone"]["feature_contract"]),
        dataset_id=str(report["dataset_id"]),
        seed=seed,
        maximum_evidence_items=int(training["maximum_evidence_items"]),
        pointer_rank=int(training["pointer_rank"]),
        normalized_features=bool(training["normalized_features"]),
        use_evidence_scores=bool(training["use_evidence_scores"]),
        use_host_control_features=bool(training["use_host_control_features"]),
        pointer_policy_score=float(training["pointer_policy_score"]),
        pointer_policy_scale=float(training["pointer_policy_scale"]),
        pointer_threshold=float(training["pointer_threshold"]),
        minimum_confidence=float(training["minimum_confidence"]),
        action_order=ACTION_ORDER,
        artifacts=artifacts,
    )
    identity = _manifest_dict(manifest)
    identity.pop("bundle_id")
    manifest = replace(
        manifest,
        bundle_id=f"gcb_{hashlib.sha256(canonical_json(identity).encode()).hexdigest()}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary_name:
        root = Path(temporary_name)
        for name, source in artifact_sources.items():
            (root / name).write_bytes(source.read_bytes())
        (root / "manifest.json").write_text(
            json.dumps(_manifest_dict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with output.open("wb") as raw_output, gzip.GzipFile(
            fileobj=raw_output, mode="wb", mtime=0, filename=""
        ) as compressed, tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                info = archive.gettarinfo(str(path), arcname=path.name)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as file_source:
                    archive.addfile(info, file_source)
    return manifest


def inspect_deployment_bundle(bundle: Path) -> DeploymentBundleManifest:
    with tarfile.open(bundle, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        manifest_member = members.get("manifest.json")
        if manifest_member is None or not manifest_member.isfile():
            raise ValueError("deployment manifest is absent")
        manifest = _manifest(json.loads(_archive_bytes(archive, manifest_member)))
        for artifact in manifest.artifacts:
            member = members.get(artifact.path)
            if member is None or not member.isfile():
                raise ValueError("deployment artifact is absent")
            payload = _archive_bytes(archive, member)
            if (
                len(payload) != artifact.bytes
                or hashlib.sha256(payload).hexdigest() != artifact.sha256
            ):
                raise ValueError("deployment artifact digest does not match")
        if set(members) != {"manifest.json", *(item.path for item in manifest.artifacts)}:
            raise ValueError("deployment bundle member set is invalid")
        return manifest


def load_deployment_bundle(
    bundle: Path,
    *,
    expected_bundle_sha256: str,
    backbone: FrozenTextBackbone,
    device: torch.device,
) -> GovernedShadowController:
    if bundle.is_symlink() or not bundle.is_file() or _sha256(bundle) != expected_bundle_sha256:
        raise ValueError("deployment bundle digest does not match")
    verified = inspect_deployment_bundle(bundle)
    with tarfile.open(bundle, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        expected = {
            "manifest.json",
            "control-head.pt",
            "training-report.json",
            "preregistration.json",
            "dataset-manifest.json",
        }
        if set(members) != expected or any(
            not member.isfile() or member.name.startswith(("/", "../"))
            for member in members.values()
        ):
            raise ValueError("deployment bundle member set is invalid")
        payloads = {
            name: _archive_bytes(archive, members[name]) for name in expected
        }
    values = json.loads(payloads["manifest.json"])
    manifest = _manifest(values)
    if manifest != verified:
        raise ValueError("deployment manifest changed during load")
    artifacts = {item.path: item for item in manifest.artifacts}
    for name in expected - {"manifest.json"}:
        item = artifacts.get(name)
        if item is None or item.bytes != len(payloads[name]) or item.sha256 != hashlib.sha256(
            payloads[name]
        ).hexdigest():
            raise ValueError("deployment artifact digest does not match")
    if (
        backbone.identity.model_id != manifest.model_id
        or backbone.identity.revision != manifest.model_revision
        or backbone.identity.feature_contract != manifest.feature_contract
        or backbone.identity.hidden_size < 1
    ):
        raise ValueError("deployment backbone identity does not match")
    checkpoint = torch.load(
        __import__("io").BytesIO(payloads["control-head.pt"]),
        map_location=device,
        weights_only=False,
    )
    if (
        checkpoint.get("version") != 1
        or tuple(checkpoint.get("action_order", ())) != manifest.action_order
        or checkpoint.get("maximum_evidence_items") != manifest.maximum_evidence_items
        or json.loads(checkpoint.get("backbone", "{}")) != json.loads(
            canonical_json(backbone.identity)
        )
    ):
        raise ValueError("deployment checkpoint contract is invalid")
    head = GovernedControlHead(
        backbone.identity.hidden_size,
        pointer_rank=manifest.pointer_rank,
        normalized_features=manifest.normalized_features,
        use_evidence_scores=manifest.use_evidence_scores,
        pointer_policy_score=manifest.pointer_policy_score,
        pointer_policy_scale=manifest.pointer_policy_scale,
        use_host_control_features=manifest.use_host_control_features,
    ).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    return GovernedShadowController(manifest, backbone, head, device=device)


class GovernedShadowController:
    def __init__(
        self,
        manifest: DeploymentBundleManifest,
        backbone: FrozenTextBackbone,
        head: GovernedControlHead,
        *,
        device: torch.device,
    ) -> None:
        self.manifest = manifest
        self.backbone = backbone
        self.head = head
        self.device = device

    @torch.inference_mode()
    def observe(
        self,
        *,
        query: str,
        evidence: EvidenceBundle,
        host_decision: SufficiencyDecision,
    ) -> ShadowControlResult:
        ordered = tuple(
            sorted(
                (hit for hit in evidence.hits if hit.active and hit.trusted),
                key=lambda hit: hit.content_digest,
            )
        )
        if len(ordered) > self.manifest.maximum_evidence_items:
            raise ValueError("shadow evidence exceeds the deployment item bound")
        context = self.backbone.encode(
            (_context_text(query, ordered, evidence),), device=self.device
        )
        validate_frozen_features(self.backbone, context, items=1)
        hidden = self.backbone.identity.hidden_size
        evidence_features = torch.zeros(
            (1, self.manifest.maximum_evidence_items, hidden),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros(
            (1, self.manifest.maximum_evidence_items), dtype=torch.bool, device=self.device
        )
        scores = torch.zeros_like(mask, dtype=torch.float32)
        if ordered:
            encoded = self.backbone.encode(tuple(hit.text for hit in ordered), device=self.device)
            validate_frozen_features(self.backbone, encoded, items=len(ordered))
            evidence_features[0, : len(ordered)] = encoded
            mask[0, : len(ordered)] = True
            scores[0, : len(ordered)] = torch.tensor(
                [hit.score for hit in ordered], dtype=torch.float32, device=self.device
            )
        host = torch.tensor(
            [[
                float(evidence.blocked),
                float(evidence.conflicting),
                float(not ordered),
                max((hit.score for hit in ordered), default=0.0),
                float(len(ordered)),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.head(context, evidence_features, mask, scores, host)
        actions, pointers = decode_control(
            logits,
            mask,
            blocked=torch.tensor([evidence.blocked], device=self.device),
            conflicting=torch.tensor([evidence.conflicting], device=self.device),
            minimum_confidence=self.manifest.minimum_confidence,
            pointer_threshold=self.manifest.pointer_threshold,
        )
        predicted = tuple(ControlAction)[int(actions.item())]
        selected = tuple(
            hit.handle
            for hit, selected in zip(
                ordered, pointers[0, : len(ordered)].tolist(), strict=True
            )
            if selected
        )
        expected = _host_action(host_decision)
        confidence = float(torch.softmax(logits.action_logits, -1).max().item())
        return ShadowControlResult(
            SHADOW_SCHEMA,
            host_decision,
            predicted,
            selected,
            confidence,
            predicted is not expected or (
                predicted is ControlAction.ANSWER
                and not set(selected) <= {hit.handle for hit in ordered}
            ),
            self.manifest.bundle_id,
        )


class AuditedShadowObserver:
    def __init__(self, controller: GovernedShadowController, audit: AuditChain) -> None:
        self.controller = controller
        self.audit = audit

    def observe(
        self,
        *,
        query: str,
        evidence: EvidenceBundle,
        host_decision: SufficiencyDecision,
    ) -> ShadowControlResult:
        started = time.perf_counter_ns()
        result = self.controller.observe(
            query=query,
            evidence=evidence,
            host_decision=host_decision,
        )
        latency_us = (time.perf_counter_ns() - started) // 1000
        self.audit.append(
            occurred_at_us=time.time_ns() // 1000,
            event_type="governed_control_shadow",
            subject_digest=hashlib.sha256(
                f"{evidence.tenant.value}\0{evidence.query_digest}".encode()
            ).hexdigest(),
            outcome="divergent" if result.divergent else "matched",
            detail={
                "bundle_id": result.bundle_id,
                "host_decision": result.host_decision.value,
                "predicted_action": result.predicted_action.value,
                "selected_handles": list(result.selected_handles),
                "action_confidence": result.action_confidence,
                "latency_us": latency_us,
                "corpus_generation": evidence.corpus_generation,
                "snapshot_fingerprint": evidence.snapshot_fingerprint,
            },
        )
        return result


def _context_text(query: str, evidence: tuple[EvidenceHit, ...], bundle: EvidenceBundle) -> str:
    return canonical_json(
        {
            "schema": "governed-control-context-v1",
            "query": query,
            "evidence": [
                {
                    "text": hit.text,
                    "score": hit.score,
                    "trusted": hit.trusted,
                    "active": hit.active,
                }
                for hit in evidence
            ],
            "approximate": bundle.approximate,
            "conflicting": bundle.conflicting,
            "blocked": bundle.blocked,
        }
    )


def _host_action(decision: SufficiencyDecision) -> ControlAction:
    return {
        SufficiencyDecision.SUPPORTED: ControlAction.ANSWER,
        SufficiencyDecision.PARTIAL: ControlAction.REQUEST_EVIDENCE,
        SufficiencyDecision.ABSENT: ControlAction.REQUEST_EVIDENCE,
        SufficiencyDecision.CONFLICT: ControlAction.ABSTAIN,
        SufficiencyDecision.BLOCKED: ControlAction.ABSTAIN,
        SufficiencyDecision.PENDING: ControlAction.REQUEST_EVIDENCE,
    }[decision]


def _manifest_dict(value: DeploymentBundleManifest) -> dict[str, object]:
    result = json.loads(canonical_json(value))
    if not isinstance(result, dict):
        raise TypeError("deployment manifest serialization is invalid")
    return cast(dict[str, object], result)


def _manifest(value: object) -> DeploymentBundleManifest:
    if not isinstance(value, dict):
        raise ValueError("deployment manifest must be an object")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("deployment artifact manifest is invalid")
    manifest = DeploymentBundleManifest(
        **{key: item for key, item in value.items() if key not in {"artifacts", "action_order"}},
        action_order=tuple(cast(list[str], value["action_order"])),
        artifacts=tuple(ArtifactDigest(**item) for item in artifacts),
    )
    identity = _manifest_dict(manifest)
    identity.pop("bundle_id")
    expected = f"gcb_{hashlib.sha256(canonical_json(identity).encode()).hexdigest()}"
    if manifest.schema != BUNDLE_SCHEMA or manifest.bundle_id != expected:
        raise ValueError("deployment bundle identity is invalid")
    return manifest


def _archive_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("deployment archive member is unreadable")
    return source.read()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
