#!/usr/bin/env python3
"""Run the preregistered non-MARS governed-control shadow campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from celiums_rezero.governed.deployment import (
    AuditedShadowObserver,
    load_deployment_bundle,
    load_rezero_deployment_bundle,
)
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.schemas import ControlAction
from celiums_rezero.knowledge.coordinator import normalize_query
from celiums_rezero.knowledge.operations import AuditChain
from celiums_rezero.knowledge.schemas import (
    EvidenceBundle,
    EvidenceHit,
    SufficiencyPolicy,
    TenantId,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_shadow_campaign(**vars(arguments))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def run_shadow_campaign(
    *,
    model: Path,
    bundle: Path,
    bundle_sha256: str,
    cases: Path,
    preregistration: Path,
    out: Path,
) -> dict[str, object]:
    prereg = json.loads(preregistration.read_text())
    if hashlib.sha256(cases.read_bytes()).hexdigest() != prereg["cases_sha256"]:
        raise ValueError("shadow case digest differs from preregistration")
    if bundle_sha256 != prereg["bundle_sha256"]:
        raise ValueError("shadow bundle digest differs from preregistration")
    device = torch.device("cuda:0")
    backbone = Gemma4E4BFrozenBackbone(model, device=str(device))
    loader = (
        load_rezero_deployment_bundle
        if prereg.get("controller_kind") == "rezero_sequence_control_v1"
        else load_deployment_bundle
    )
    controller = loader(
        bundle,
        expected_bundle_sha256=bundle_sha256,
        backbone=backbone,
        device=device,
    )
    if controller.manifest.bundle_id != prereg["bundle_id"]:
        raise ValueError("shadow bundle identity differs from preregistration")
    out.mkdir(parents=True, exist_ok=True)
    audit = AuditChain(out / "shadow-audit.jsonl")
    observer = AuditedShadowObserver(controller, audit)
    policy = SufficiencyPolicy()
    warmup_evidence = EvidenceBundle(
        TenantId("shadow_external"),
        hashlib.sha256(b"shadow warmup").hexdigest(),
        "shadow_external_v1",
        (),
    )
    observer.observe(
        query="shadow warmup",
        evidence=warmup_evidence,
        host_decision=policy.decide(warmup_evidence),
    )
    warmup_records = len(audit.verify())
    rows = []
    for index, line in enumerate(cases.read_text().splitlines()):
        value = json.loads(line)
        query = str(value["query"])
        normalized_query = normalize_query(query)
        query_digest = hashlib.sha256(normalized_query.encode()).hexdigest()
        hits = tuple(
            EvidenceHit(
                handle=f"passage_{index:08x}{hit_index:08x}",
                source_id=f"external_{index}",
                source_version="v1",
                text=str(hit["text"]),
                score=float(hit["score"]),
                content_digest=hashlib.sha256(str(hit["text"]).encode()).hexdigest(),
            )
            for hit_index, hit in enumerate(value["evidence"])
        )
        evidence = EvidenceBundle(
            TenantId("shadow_external"),
            query_digest,
            "shadow_external_v1",
            hits,
            approximate=False,
            conflicting=bool(value.get("conflicting", False)),
            blocked=bool(value.get("blocked", False)),
        )
        host_decision = policy.decide(evidence)
        started = time.perf_counter_ns()
        result = observer.observe(
            query=normalized_query,
            evidence=evidence,
            host_decision=host_decision,
        )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        expected_action = ControlAction(str(value["expected_action"]))
        expected_handles = (
            tuple(
                hit.handle
                for hit in sorted(hits, key=lambda item: item.content_digest)
                if hit.score >= policy.minimum_score
            )
            if expected_action is ControlAction.ANSWER
            else ()
        )
        pointer_match = result.selected_handles == expected_handles
        unsafe_upgrade = (
            expected_action is not ControlAction.ANSWER
            and result.predicted_action is ControlAction.ANSWER
        )
        conservative_downgrade = (
            expected_action is ControlAction.REQUEST_EVIDENCE
            and result.predicted_action is ControlAction.ABSTAIN
        )
        operational_divergence = result.divergent and not conservative_downgrade
        rows.append(
            {
                "scenario_id": value["scenario_id"],
                "host_decision": host_decision.value,
                "expected_action": expected_action.value,
                "predicted_action": result.predicted_action.value,
                "selected_handles": list(result.selected_handles),
                "expected_handles": list(expected_handles),
                "action_match": result.predicted_action is expected_action,
                "pointer_match": pointer_match,
                "unsafe_upgrade": unsafe_upgrade,
                "conservative_downgrade": conservative_downgrade,
                "divergent": result.divergent,
                "operational_divergence": operational_divergence,
                "latency_ms": latency_ms,
            }
        )
    latencies = [float(row["latency_ms"]) for row in rows]
    gates = prereg["gates"]
    action_match = sum(bool(row["action_match"]) for row in rows) / len(rows)
    pointer_match = sum(bool(row["pointer_match"]) for row in rows) / len(rows)
    divergences = sum(bool(row["divergent"]) for row in rows)
    operational_divergences = sum(
        bool(row["operational_divergence"]) for row in rows
    )
    conservative = sum(bool(row["conservative_downgrade"]) for row in rows)
    unsafe = sum(bool(row["unsafe_upgrade"]) for row in rows)
    p95 = sorted(latencies)[max(0, __import__("math").ceil(len(latencies) * 0.95) - 1)]
    report = {
        "schema": "hyphae-transformer.governed-control-shadow-report/v1",
        "completed": True,
        "bundle_id": controller.manifest.bundle_id,
        "cases_sha256": prereg["cases_sha256"],
        "cases": len(rows),
        "action_match_rate": action_match,
        "pointer_exact_match": pointer_match,
        "divergences": divergences,
        "operational_divergences": operational_divergences,
        "conservative_downgrades": conservative,
        "unsafe_upgrade_count": unsafe,
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": p95,
        "audit_records": len(audit.verify()),
        "warmup_records": warmup_records,
        "rows": rows,
    }
    report["passed"] = (
        action_match >= gates["action_match_rate"]
        and pointer_match >= gates["pointer_exact_match"]
        and operational_divergences <= gates["maximum_operational_divergences"]
        and conservative <= gates["maximum_conservative_downgrades"]
        and unsafe <= gates["unsafe_upgrade_count"]
        and report["mean_latency_ms"] <= gates["maximum_mean_latency_ms"]
        and report["p95_latency_ms"] <= gates["maximum_p95_latency_ms"]
    )
    (out / "shadow-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
