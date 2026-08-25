from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from celiums_rezero.governed.backbone import FixtureBackboneV1
from celiums_rezero.governed.navigation_experiment import (
    decide_navigation_step,
    load_navigation_pilot,
)
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyPolicy,
    TenantId,
)

TENANT = TenantId("navigation_fixture")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hit(handle: str, score: float) -> EvidenceHit:
    text = f"policy body {handle}"
    return EvidenceHit(
        handle=handle,
        source_id="official_docs",
        source_version="v1",
        text=text,
        score=score,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        trusted=True,
        active=True,
    )


def _save_checkpoint(path: Path) -> Path:
    backbone = FixtureBackboneV1()
    from celiums_rezero.governed.hyphaelm import ReZeroNeuroPilot

    pilot = ReZeroNeuroPilot(
        backbone.identity.hidden_size,
        control_size=16,
        n_layers=1,
        n_heads=4,
        maximum_evidence_items=2,
    )
    torch.save(
        {
            "version": 1,
            "head": pilot.state_dict(),
            "backbone": json.dumps(
                {
                    "model_id": backbone.identity.model_id,
                    "revision": backbone.identity.revision,
                    "feature_contract": backbone.identity.feature_contract,
                    "hidden_size": backbone.identity.hidden_size,
                },
                sort_keys=True,
            ),
            "action_order": ["search", "answer", "abstain"],
            "maximum_evidence_items": 2,
        },
        path,
    )
    return path


def test_load_navigation_pilot_rejects_foreign_contracts(tmp_path: Path) -> None:
    good = _save_checkpoint(tmp_path / "good.pt")
    pilot = load_navigation_pilot(
        good,
        hidden_size=FixtureBackboneV1().identity.hidden_size,
        device=torch.device("cpu"),
        control_size=16,
        n_layers=1,
        n_heads=4,
    )
    assert pilot.maximum_evidence_items == 2
    bad = tmp_path / "bad.pt"
    payload = torch.load(good, map_location="cpu", weights_only=False)
    payload["action_order"] = ["answer", "search", "abstain"]
    torch.save(payload, bad)
    try:
        load_navigation_pilot(
            bad,
            hidden_size=FixtureBackboneV1().identity.hidden_size,
            device=torch.device("cpu"),
            control_size=16,
            n_layers=1,
            n_heads=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("navigation checkpoint with swapped actions accepted")


def test_decide_navigation_step_uses_live_certificates(tmp_path: Path) -> None:
    backbone = FixtureBackboneV1()
    device = torch.device("cpu")
    pilot = load_navigation_pilot(
        _save_checkpoint(tmp_path / "nav-step.pt"),
        hidden_size=backbone.identity.hidden_size,
        device=device,
        control_size=16,
        n_layers=1,
        n_heads=4,
    )
    policy = SufficiencyPolicy()
    empty = EvidenceBundle(TENANT, _digest("q0"), "gen", ())
    empty_decision = decide_navigation_step(
        backbone=backbone,
        pilot=pilot,
        query="what is the approved maintenance window?",
        evidence=empty,
        policy=policy,
        search_steps_used=0,
        device=device,
    )
    assert empty_decision.action in {"search", "answer", "abstain"}
    filled = EvidenceBundle(
        TENANT,
        _digest("q1"),
        "gen",
        (_hit("doc_" + "0"*16, 0.9), _hit("doc_" + "f"*16, 0.1)),
    )
    filled_decision = decide_navigation_step(
        backbone=backbone,
        pilot=pilot,
        query="what is the approved maintenance window?",
        evidence=filled,
        policy=policy,
        search_steps_used=1,
        device=device,
    )
    assert set(filled_decision.selected_handles) <= {"doc_" + "0"*16, "doc_" + "f"*16}
