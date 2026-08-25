from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from celiums_rezero.core.gates import ReZeroGate
from celiums_rezero.governed import (
    ControlAction,
    ControlTarget,
    ControlTrainConfig,
    FixtureBackboneV1,
    GovernedControlHead,
    ReZeroSequenceControlHead,
    TrajectoryStep,
    train_control_head,
)
from celiums_rezero.governed.data import (
    HOST_CONTROL_V2,
    host_control_values,
    load_trajectory_split,
    materialize_governed_batch,
)
from celiums_rezero.governed.evaluation import evaluate_control_head
from celiums_rezero.knowledge.schemas import EvidenceHit, SufficiencyPolicy
from celiums_rezero.lab.serialization import canonical_json


def hit(handle: str, text: str, score: float = 0.95) -> EvidenceHit:
    return EvidenceHit(
        handle=handle,
        source_id="docs",
        source_version="v1",
        text=text,
        score=score,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
    )


def records() -> tuple[TrajectoryStep, ...]:
    items = []
    for index in range(12):
        evidence = (hit(f"passage_{index:016x}", f"supported fact {index}"),)
        items.append(
            TrajectoryStep(
                trajectory_id=f"trajectory_{index:016x}",
                scenario_id=f"scenario_{index:016x}",
                step_index=0,
                query=f"supported query {index}",
                generation_id="generation_fixture",
                evidence=evidence,
                approximate=False,
                conflicting=False,
                blocked=False,
                target=ControlTarget(ControlAction.ANSWER, (evidence[0].handle,)),
                provenance="fixture-v1",
            )
        )
    for index in range(12, 24):
        items.append(
            TrajectoryStep(
                trajectory_id=f"trajectory_{index:016x}",
                scenario_id=f"scenario_{index:016x}",
                step_index=0,
                query=f"missing query {index}",
                generation_id="generation_fixture",
                evidence=(),
                approximate=False,
                conflicting=False,
                blocked=False,
                target=ControlTarget(ControlAction.REQUEST_EVIDENCE),
                provenance="fixture-v1",
            )
        )
    for index in range(24, 36):
        items.append(
            TrajectoryStep(
                trajectory_id=f"trajectory_{index:016x}",
                scenario_id=f"scenario_{index:016x}",
                step_index=0,
                query=f"blocked query {index}",
                generation_id="generation_fixture",
                evidence=(),
                approximate=False,
                conflicting=False,
                blocked=True,
                target=ControlTarget(ControlAction.ABSTAIN),
                provenance="fixture-v1",
            )
        )
    return tuple(items)


def test_control_training_is_deterministic_and_backbone_frozen(tmp_path: Path) -> None:
    backbone = FixtureBackboneV1()
    torch.manual_seed(17)
    first = GovernedControlHead(backbone.hidden_size)
    torch.manual_seed(17)
    second = GovernedControlHead(backbone.hidden_size)
    config = ControlTrainConfig(epochs=80, learning_rate=0.03)
    first_summary = train_control_head(
        backbone, first, records(), config, checkpoint=tmp_path / "first.pt"
    )
    second_summary = train_control_head(
        backbone, second, records(), config, checkpoint=tmp_path / "second.pt"
    )
    assert first_summary.backbone_unchanged and second_summary.backbone_unchanged
    assert first_summary.final_loss == second_summary.final_loss
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    evaluation = evaluate_control_head(backbone, first, records())
    assert evaluation.action_accuracy >= 0.9


def test_trajectory_loader_rejects_tampering(tmp_path: Path) -> None:
    record = records()[0]
    value = json.loads(canonical_json(record))
    value["schema"] = "governed-trajectory-step-v1"
    path = tmp_path / "train.jsonl"
    path.write_text(canonical_json(value) + "\n", encoding="ascii")
    loaded = load_trajectory_split(path, SufficiencyPolicy())
    assert loaded == (record,)
    value["evidence"][0]["text"] = "tampered"
    path.write_text(canonical_json(value) + "\n", encoding="ascii")
    try:
        load_trajectory_split(path, SufficiencyPolicy())
    except ValueError:
        pass
    else:
        raise AssertionError("tampered trajectory evidence was accepted")


def test_conflict_and_blocked_targets_must_abstain() -> None:
    item = records()[-1]
    item.validate_policy(SufficiencyPolicy())
    bad = TrajectoryStep(
        trajectory_id="trajectory_ffffffffffffffff",
        scenario_id="scenario_ffffffffffffffff",
        step_index=0,
        query="blocked",
        generation_id="generation_fixture",
        evidence=(),
        approximate=False,
        conflicting=False,
        blocked=True,
        target=ControlTarget(ControlAction.REQUEST_EVIDENCE),
        provenance="fixture-v1",
    )
    try:
        bad.validate_policy(SufficiencyPolicy())
    except ValueError:
        pass
    else:
        raise AssertionError("blocked target did not require abstention")


def test_control_evaluation_uses_the_head_device() -> None:
    class TrackingBackbone(FixtureBackboneV1):
        def __init__(self) -> None:
            self.devices: list[torch.device] = []

        def encode(self, texts: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
            self.devices.append(device)
            return super().encode(texts, device=device)

    backbone = TrackingBackbone()
    head = GovernedControlHead(backbone.hidden_size)
    evaluate_control_head(backbone, head, records())
    assert backbone.devices
    assert set(backbone.devices) == {next(head.parameters()).device}


def test_materialized_batch_matches_direct_batch() -> None:
    backbone = FixtureBackboneV1()
    direct = materialize_governed_batch(
        records(),
        backbone,
        maximum_evidence_items=8,
        feature_batch_size=len(records()),
        device=torch.device("cpu"),
    )
    chunked = materialize_governed_batch(
        records(),
        backbone,
        maximum_evidence_items=8,
        feature_batch_size=5,
        device=torch.device("cpu"),
    )
    for field in direct.__slots__:
        torch.testing.assert_close(getattr(direct, field), getattr(chunked, field))


def test_score_aware_head_can_separate_pointer_targets() -> None:
    head = GovernedControlHead(
        4,
        normalized_features=True,
        use_evidence_scores=True,
    )
    assert head.evidence_score is not None
    with torch.no_grad():
        head.context.weight.zero_()
        head.evidence.weight.zero_()
        head.evidence_score.weight.fill_(20)
        head.evidence_score.bias.fill_(-14)
    logits = head(
        torch.ones((1, 4)),
        torch.ones((1, 2, 4)),
        torch.ones((1, 2), dtype=torch.bool),
        torch.tensor([[0.95, 0.4]]),
    )
    assert (torch.sigmoid(logits.evidence_logits) >= 0.5).tolist() == [[True, False]]


def test_pointer_policy_prior_separates_sufficient_evidence() -> None:
    head = GovernedControlHead(
        4,
        pointer_policy_score=0.72,
        pointer_policy_scale=20,
    )
    with torch.no_grad():
        head.context.weight.zero_()
        head.evidence.weight.zero_()
    logits = head(
        torch.ones((1, 4)),
        torch.ones((1, 3, 4)),
        torch.ones((1, 3), dtype=torch.bool),
        torch.tensor([[0.95, 0.85, 0.4]]),
    )
    assert (torch.sigmoid(logits.evidence_logits) >= 0.5).tolist() == [
        [True, True, False]
    ]


def test_host_control_features_are_explicit_action_inputs() -> None:
    head = GovernedControlHead(4, use_host_control_features=True)
    with torch.no_grad():
        head.action.weight.zero_()
        head.action.bias.zero_()
        head.action.weight[1, 6] = 10
    logits = head(
        torch.ones((1, 4)),
        torch.ones((1, 1, 4)),
        torch.ones((1, 1), dtype=torch.bool),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    assert logits.action_logits.argmax(-1).item() == 1


def test_rezero_sequence_controller_uses_shared_zero_initialized_gates() -> None:
    head = ReZeroSequenceControlHead(
        8,
        control_size=16,
        n_layers=2,
        n_heads=4,
        maximum_evidence_items=3,
    )
    gates = [module for module in head.modules() if isinstance(module, ReZeroGate)]
    assert len(gates) == 2
    assert all(gate.value.item() == 0 for gate in gates)
    assert all(block.attention_gate is block.mlp_gate for block in head.blocks)


def test_rezero_sequence_controller_produces_bounded_control_logits() -> None:
    head = ReZeroSequenceControlHead(
        8,
        control_size=16,
        n_layers=2,
        n_heads=4,
        maximum_evidence_items=3,
    )
    logits = head(
        torch.ones((2, 8)),
        torch.ones((2, 3, 8)),
        torch.tensor([[True, True, False], [False, False, False]]),
        torch.tensor([[0.95, 0.4, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.95, 2.0], [0.0, 0.0, 1.0, 0.0, 0.0]]
        ),
    )
    assert logits.action_logits.shape == (2, 3)
    assert logits.evidence_logits.shape == (2, 3)
    assert torch.isneginf(logits.evidence_logits[0, 2])
    assert torch.isneginf(logits.evidence_logits[1]).all()
    assert logits.evidence_logits[0, 0] > logits.evidence_logits[0, 1]


def test_rezero_sequence_controller_trains_with_frozen_backbone(tmp_path: Path) -> None:
    backbone = FixtureBackboneV1()
    torch.manual_seed(17)
    head = ReZeroSequenceControlHead(
        backbone.hidden_size,
        control_size=32,
        n_layers=1,
        n_heads=4,
    )
    summary = train_control_head(
        backbone,
        head,
        records(),
        ControlTrainConfig(epochs=2, learning_rate=0.01),
        checkpoint=tmp_path / "rezero-sequence.pt",
    )
    assert summary.backbone_unchanged
    assert summary.final_loss <= summary.initial_loss
    checkpoint = torch.load(tmp_path / "rezero-sequence.pt", weights_only=True)
    weight_decays = {
        float(group["weight_decay"])
        for group in checkpoint["optimizer"]["param_groups"]
    }
    assert weight_decays == {0.0, 0.01}
    assert sum(parameter.numel() for parameter in head.parameters()) < 5_000_000


def test_host_policy_certificate_exposes_sufficiency_without_shadow_labels() -> None:
    policy = SufficiencyPolicy()
    supported = records()[0]
    from celiums_rezero.knowledge.schemas import EvidenceBundle, TenantId

    bundle = EvidenceBundle(
        TenantId("tenant_a"),
        "0" * 64,
        supported.generation_id,
        supported.evidence,
    )
    values = host_control_values(bundle, policy, HOST_CONTROL_V2)
    assert len(values) == 17
    assert values[-5:] == (1.0, 0.0, 0.0, 0.0, 0.0)


def test_rezero_action_policy_prior_maps_certified_decisions() -> None:
    head = ReZeroSequenceControlHead(
        8,
        control_size=16,
        n_layers=1,
        n_heads=4,
        host_control_size=17,
        action_policy_prior_scale=20,
        maximum_evidence_items=2,
    )
    with torch.no_grad():
        head.action.weight.zero_()
        head.action.bias.zero_()
    base = torch.zeros((3, 17))
    base[0, -5] = 1
    base[1, -3] = 1
    base[2, -1] = 1
    logits = head(
        torch.ones((3, 8)),
        torch.ones((3, 2, 8)),
        torch.tensor([[True, False], [False, False], [False, False]]),
        torch.tensor([[0.95, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        base,
    )
    assert logits.action_logits.argmax(-1).tolist() == [0, 1, 2]


def test_bounded_policy_certificates_cannot_be_reversed() -> None:
    head = ReZeroSequenceControlHead(
        8,
        control_size=16,
        n_layers=1,
        n_heads=4,
        host_control_size=17,
        action_policy_prior_scale=20,
        action_residual_bound=1,
        pointer_residual_bound=1,
        maximum_evidence_items=2,
    )
    with torch.no_grad():
        head.action.weight.fill_(-1000)
        head.action.bias.fill_(-1000)
        head.pointer.weight.fill_(-1000)
    certificate = torch.zeros((1, 17))
    certificate[0, -5] = 1
    logits = head(
        torch.ones((1, 8)),
        torch.ones((1, 2, 8)),
        torch.tensor([[True, True]]),
        torch.tensor([[0.95, 0.4]]),
        certificate,
    )
    assert logits.action_logits.argmax(-1).item() == 0
    assert (logits.evidence_logits >= 0).tolist() == [[True, False]]


def test_minimum_loss_checkpoint_restores_selected_state(tmp_path: Path) -> None:
    backbone = FixtureBackboneV1()
    torch.manual_seed(17)
    head = ReZeroSequenceControlHead(backbone.hidden_size, control_size=32, n_layers=1)
    summary = train_control_head(
        backbone,
        head,
        records(),
        ControlTrainConfig(
            epochs=4,
            learning_rate=0.01,
            checkpoint_selection="minimum_training_loss",
        ),
        checkpoint=tmp_path / "minimum.pt",
    )
    assert summary.selected_loss == summary.best_loss
    assert 1 <= summary.selected_epoch <= summary.epochs
    checkpoint = torch.load(tmp_path / "minimum.pt", weights_only=True)
    assert checkpoint["config"]["checkpoint_selection"] == "minimum_training_loss"
