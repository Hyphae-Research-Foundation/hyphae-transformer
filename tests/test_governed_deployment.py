from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
import torch

from celiums_rezero.governed import (
    AuditedShadowObserver,
    ControlAction,
    GovernedControlHead,
    ReZeroSequenceControlHead,
    build_deployment_bundle,
    build_rezero_deployment_bundle,
    inspect_deployment_bundle,
    inspect_rezero_deployment_bundle,
    load_deployment_bundle,
    load_rezero_deployment_bundle,
)
from celiums_rezero.governed.backbone import FixtureBackboneV1
from celiums_rezero.knowledge.operations import AuditChain
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyDecision,
    TenantId,
)
from celiums_rezero.lab.serialization import canonical_json


def test_bundle_build_load_and_shadow_observation(tmp_path: Path) -> None:
    backbone = FixtureBackboneV1()
    head = GovernedControlHead(
        backbone.hidden_size,
        normalized_features=True,
        pointer_policy_score=0.72,
        pointer_policy_scale=20,
        use_host_control_features=True,
    )
    with torch.no_grad():
        head.context.weight.zero_()
        head.evidence.weight.zero_()
        head.action.weight.zero_()
        head.action.bias.zero_()
        head.action.weight[0, backbone.hidden_size + 3] = 10
    checkpoint = tmp_path / "control-head.pt"
    torch.save(
        {
            "version": 1,
            "head": head.state_dict(),
            "optimizer": {},
            "config": {},
            "backbone": json.dumps(
                {
                    "family": backbone.identity.family,
                    "model_id": backbone.identity.model_id,
                    "revision": backbone.identity.revision,
                    "artifact_manifest_sha256": backbone.identity.artifact_manifest_sha256,
                    "tokenizer_manifest_sha256": backbone.identity.tokenizer_manifest_sha256,
                    "runtime_version": backbone.identity.runtime_version,
                    "feature_contract": backbone.identity.feature_contract,
                    "hidden_size": backbone.identity.hidden_size,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "backbone_state": backbone.state_fingerprint(),
            "maximum_evidence_items": 8,
            "action_order": ["answer", "request_evidence", "abstain"],
            "record_digest": "0" * 64,
        },
        checkpoint,
    )
    report = tmp_path / "training-report.json"
    report.write_text(
        json.dumps(
            {
                "completed": True,
                "passed": True,
                "model_id": backbone.identity.model_id,
                "model_revision": backbone.identity.revision,
                "dataset_id": "gtd_fixture",
                "seeds": [
                    {
                        "seed": 17,
                        "passed": True,
                        "training": {
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest()
                        },
                    }
                ],
            }
        )
    )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "backbone": {"feature_contract": backbone.identity.feature_contract},
                "dataset": {"governed_dataset_id": "gtd_fixture"},
                "training": {
                    "maximum_evidence_items": 8,
                    "pointer_rank": 32,
                    "normalized_features": True,
                    "use_evidence_scores": False,
                    "use_host_control_features": True,
                    "pointer_policy_score": 0.72,
                    "pointer_policy_scale": 20,
                    "pointer_threshold": 0.5,
                    "minimum_confidence": 0.5,
                },
            }
        )
    )
    dataset = tmp_path / "dataset-manifest.json"
    dataset.write_text(json.dumps({"dataset_id": "gtd_fixture"}))
    bundle = tmp_path / "bundle.tar.gz"
    manifest = build_deployment_bundle(
        output=bundle,
        checkpoint=checkpoint,
        training_report=report,
        preregistration=preregistration,
        dataset_manifest=dataset,
        source_revision="1" * 40,
    )
    second = tmp_path / "second.tar.gz"
    build_deployment_bundle(
        output=second,
        checkpoint=checkpoint,
        training_report=report,
        preregistration=preregistration,
        dataset_manifest=dataset,
        source_revision="1" * 40,
    )
    assert second.read_bytes() == bundle.read_bytes()
    assert inspect_deployment_bundle(bundle) == manifest
    controller = load_deployment_bundle(
        bundle,
        expected_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        backbone=backbone,
        device=torch.device("cpu"),
    )
    text = "supported"
    hit = EvidenceHit(
        "passage_0123456789abcdef",
        "docs",
        "v1",
        text,
        0.95,
        hashlib.sha256(text.encode()).hexdigest(),
    )
    result = controller.observe(
        query="question",
        evidence=EvidenceBundle(TenantId("tenant_a"), "0" * 64, "generation", (hit,)),
        host_decision=SufficiencyDecision.SUPPORTED,
    )
    assert result.predicted_action is ControlAction.ANSWER
    assert result.selected_handles == (hit.handle,)
    assert not result.divergent


def test_bundle_loader_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "not-a-bundle.tar.gz"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        load_deployment_bundle(
            path,
            expected_bundle_sha256="0" * 64,
            backbone=FixtureBackboneV1(),
            device=torch.device("cpu"),
        )


def test_audited_shadow_observer_appends_result(tmp_path: Path) -> None:
    class Controller:
        def observe(self, **_values):
            from celiums_rezero.governed.deployment import ShadowControlResult

            return ShadowControlResult(
                "hyphae-transformer.governed-control-shadow/v1",
                SufficiencyDecision.ABSENT,
                ControlAction.REQUEST_EVIDENCE,
                (),
                0.9,
                False,
                "gcb_fixture",
            )

    chain = AuditChain(tmp_path / "audit.jsonl")
    observer = AuditedShadowObserver(Controller(), chain)  # type: ignore[arg-type]
    evidence = EvidenceBundle(TenantId("tenant_a"), "0" * 64, "generation", ())
    observer.observe(
        query="question",
        evidence=evidence,
        host_decision=SufficiencyDecision.ABSENT,
    )
    records = chain.verify()
    assert len(records) == 1
    assert records[0].outcome == "matched"
    assert records[0].detail["bundle_id"] == "gcb_fixture"


def test_rezero_bundle_is_deterministic_and_loadable(tmp_path: Path) -> None:
    backbone = FixtureBackboneV1()
    head = ReZeroSequenceControlHead(
        backbone.hidden_size,
        control_size=32,
        n_layers=1,
        n_heads=4,
    )
    checkpoint = tmp_path / "rezero-control.pt"
    torch.save(
        {
            "version": 1,
            "head": head.state_dict(),
            "optimizer": {},
            "config": {"learning_rate": 0.001},
            "backbone": json.dumps(
                {
                    "family": backbone.identity.family,
                    "model_id": backbone.identity.model_id,
                    "revision": backbone.identity.revision,
                    "artifact_manifest_sha256": backbone.identity.artifact_manifest_sha256,
                    "tokenizer_manifest_sha256": backbone.identity.tokenizer_manifest_sha256,
                    "runtime_version": backbone.identity.runtime_version,
                    "feature_contract": backbone.identity.feature_contract,
                    "hidden_size": backbone.identity.hidden_size,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "backbone_state": backbone.state_fingerprint(),
            "maximum_evidence_items": 8,
            "action_order": ["answer", "request_evidence", "abstain"],
            "record_digest": "0" * 64,
        },
        checkpoint,
    )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "candidate": {
                    "action_policy_prior_scale": 0.0,
                    "action_residual_bound": None,
                    "control_size": 32,
                    "layers": 1,
                    "n_heads": 4,
                    "maximum_evidence_items": 8,
                    "residual_strategy": "rezero_rms_shared",
                    "gate_init": 0.0,
                    "host_control_contract": "host-policy-summary-v1",
                    "pointer_residual_bound": None,
                },
                "dataset": {"governed_dataset_id": "gtd_fixture"},
                "training_search": {
                    "candidate_learning_rates": [0.001],
                    "pointer_policy_score": 0.72,
                    "pointer_policy_scale": 20.0,
                    "pointer_threshold": 0.5,
                    "minimum_confidence": 0.5,
                },
            }
        )
    )
    report = tmp_path / "training-report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "hyphae-transformer.rezero-sequence-control-experiment/v1",
                "completed": True,
                "passed": True,
                "scope": "gemma",
                "backbone_unchanged": True,
                "backbone": {
                    "model_id": backbone.identity.model_id,
                    "revision": backbone.identity.revision,
                    "feature_contract": backbone.identity.feature_contract,
                },
                "dataset_id": "gtd_fixture",
                "selected_learning_rate": 0.001,
                "preregistration_sha256": hashlib.sha256(
                    preregistration.read_bytes()
                ).hexdigest(),
                "final": [
                    {
                        "seed": 17,
                        "passed": True,
                        "checkpoint_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        )
    )
    dataset = tmp_path / "dataset-manifest.json"
    dataset.write_text(json.dumps({"dataset_id": "gtd_fixture"}))
    bundle = tmp_path / "rezero-bundle.tar.gz"
    manifest = build_rezero_deployment_bundle(
        output=bundle,
        checkpoint=checkpoint,
        training_report=report,
        preregistration=preregistration,
        dataset_manifest=dataset,
        source_revision="2" * 40,
    )
    second = tmp_path / "rezero-bundle-second.tar.gz"
    build_rezero_deployment_bundle(
        output=second,
        checkpoint=checkpoint,
        training_report=report,
        preregistration=preregistration,
        dataset_manifest=dataset,
        source_revision="2" * 40,
    )
    assert bundle.read_bytes() == second.read_bytes()
    assert inspect_rezero_deployment_bundle(bundle) == manifest
    controller = load_rezero_deployment_bundle(
        bundle,
        expected_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        backbone=backbone,
        device=torch.device("cpu"),
    )
    assert controller.manifest.bundle_id.startswith("rzcb_")
    result = controller.observe(
        query="missing",
        evidence=EvidenceBundle(TenantId("tenant_a"), "0" * 64, "generation", ()),
        host_decision=SufficiencyDecision.ABSENT,
    )
    assert result.selected_handles == ()

    with tarfile.open(bundle, "r:gz") as archive:
        files = {
            member.name: archive.extractfile(member).read()  # type: ignore[union-attr]
            for member in archive.getmembers()
        }
    legacy_manifest = json.loads(files["manifest.json"])
    legacy_manifest.pop("host_control_contract")
    legacy_manifest.pop("action_policy_prior_scale")
    legacy_manifest.pop("action_residual_bound")
    legacy_manifest.pop("pointer_residual_bound")
    legacy_identity = dict(legacy_manifest)
    legacy_identity.pop("bundle_id")
    legacy_manifest["bundle_id"] = "rzcb_" + hashlib.sha256(
        canonical_json(legacy_identity).encode()
    ).hexdigest()
    from celiums_rezero.governed.deployment import _rezero_manifest

    restored = _rezero_manifest(legacy_manifest)
    assert restored.host_control_contract == "host-policy-summary-v1"
    assert restored.action_policy_prior_scale == 0.0
