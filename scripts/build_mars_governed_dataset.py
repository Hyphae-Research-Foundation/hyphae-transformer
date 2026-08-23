#!/usr/bin/env python3
"""Convert pinned MARS v2 evaluation fixtures into governed trajectory splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from celiums_rezero.governed.schemas import (
    ControlAction,
    ControlTarget,
    DatasetSplit,
    GovernedDatasetManifest,
    TrajectoryStep,
)
from celiums_rezero.knowledge.schemas import EvidenceHit, SufficiencyPolicy
from celiums_rezero.lab.serialization import canonical_json

PINNED_FILES = {
    "dev.jsonl": "375151c56030482b64ad94d2ab82ef82a6e7650318a5777fdfdb08f913730553",
    "evaluation-dev.jsonl": "740d2ddd4d2482ac8a8dda7975a4b2a2c8779efefa6b7b3b5af4b407c022c66f",
    "evaluation-test.jsonl": "02cfa0dedd4949e1c9ccd0b610e260c20401e045b51825a1d0f7af71ff4be103",
    "manifest.json": "a922359bba1d4a0dcd34b048f92728a6fcd83a59b5f1507e0beb00078347f3f0",
    "test.jsonl": "f7716b01e8f123e9e891be0008c4d94f9b9a7ac539e4f2f2591d0598a1951b21",
    "validation-report.json": (
        "1dfba5e19392c3f39b0f8da40c48958dfa2a2ce058056941e773e2f76967dc76"
    ),
}
RECORDS_PER_WORLD = 19
SPLIT_WORLD_COUNTS = {"train": 20, "validation": 10, "adversarial": 10, "test": 20}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    build_dataset(arguments.release, arguments.out)
    return 0


def build_dataset(release: Path, output: Path) -> dict[str, object]:
    release = release.resolve(strict=True)
    if not release.is_dir():
        raise ValueError("MARS release must be a directory")
    for name, expected in PINNED_FILES.items():
        _verify_digest(release / name, expected)

    dev = _load_canonical(release / "dev.jsonl", expected_worlds=40)
    test = _load_canonical(release / "test.jsonl", expected_worlds=20)
    evaluation_dev = _load_evaluation(release / "evaluation-dev.jsonl")
    evaluation_test = _load_evaluation(release / "evaluation-test.jsonl")
    _verify_evaluation_alignment(dev, evaluation_dev)
    _verify_evaluation_alignment(test, evaluation_test)

    dev_worlds = _ordered_worlds(dev)
    test_worlds = _ordered_worlds(test)
    split_worlds = {
        "train": dev_worlds[:20],
        "validation": dev_worlds[20:30],
        "adversarial": dev_worlds[30:40],
        "test": test_worlds,
    }
    if any(len(split_worlds[name]) != count for name, count in SPLIT_WORLD_COUNTS.items()):
        raise ValueError("MARS release does not satisfy the preregistered world partition")
    if len(set().union(*(set(worlds) for worlds in split_worlds.values()))) != 60:
        raise ValueError("MARS world identity leaked across governed splits")

    canonical_by_id = {
        str(record["record_id"]): record for record in (*dev, *test)
    }
    evaluation_by_id = {
        str(record["episode_id"]): record for record in (*evaluation_dev, *evaluation_test)
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_splits: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test", "adversarial"):
        records = _convert_worlds(
            canonical_by_id,
            evaluation_by_id,
            split=split,
            worlds=split_worlds[split],
        )
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(
                canonical_json(
                    {
                        "schema": "governed-trajectory-step-v1",
                        **json.loads(canonical_json(record)),
                    }
                )
                + "\n"
                for record in records
            ),
            encoding="ascii",
        )
        manifest_splits[split] = {
            "path": path.name,
            "records": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    policy = SufficiencyPolicy(
        minimum_score=0.72,
        minimum_margin=0.08,
        minimum_trusted_hits=1,
        allow_approximate=False,
    )
    governed_manifest = GovernedDatasetManifest(
        splits=tuple(
            (name, DatasetSplit(**manifest_splits[name])) for name in sorted(manifest_splits)
        ),
        policy=policy,
        maximum_evidence_items=8,
    )
    manifest: dict[str, object] = {
        "schema": "governed-trajectory-dataset-v1",
        "dataset_id": governed_manifest.dataset_id,
        "maximum_evidence_items": governed_manifest.maximum_evidence_items,
        "policy": json.loads(canonical_json(policy)),
        "source": {
            "license": "CC0-1.0",
            "files": PINNED_FILES,
            "records_per_world": RECORDS_PER_WORLD,
            "score_contract": "supported-order:0.95-minus-0.10;otherwise:0.40",
            "split_world_counts": SPLIT_WORLD_COUNTS,
            "split_world_ids_sha256": {
                name: hashlib.sha256(canonical_json(worlds).encode()).hexdigest()
                for name, worlds in split_worlds.items()
            },
        },
        "splits": manifest_splits,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return manifest


def _load_canonical(path: Path, *, expected_worlds: int) -> tuple[dict[str, Any], ...]:
    records = _load_jsonl(path)
    required = {"record_id", "world_id", "claims", "expected_outcome"}
    if any(not required <= set(record) for record in records):
        raise ValueError(f"canonical MARS records are incomplete: {path}")
    if len(records) != expected_worlds * RECORDS_PER_WORLD:
        raise ValueError(f"canonical MARS record count is invalid: {path}")
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        worlds[str(record["world_id"])].append(record)
    if len(worlds) != expected_worlds or any(
        len(items) != RECORDS_PER_WORLD for items in worlds.values()
    ):
        raise ValueError(f"canonical MARS world grouping is invalid: {path}")
    return records


def _load_evaluation(path: Path) -> tuple[dict[str, Any], ...]:
    records = _load_jsonl(path)
    required = {"episode_id", "backend", "expected_outcome", "instruction"}
    if any(not required <= set(record) for record in records):
        raise ValueError(f"evaluation MARS records are incomplete: {path}")
    return records


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"MARS JSONL record is not an object: {path}")
        records.append(value)
    if not records:
        raise ValueError(f"MARS JSONL input is empty: {path}")
    return tuple(records)


def _verify_evaluation_alignment(
    canonical: tuple[dict[str, Any], ...], evaluation: tuple[dict[str, Any], ...]
) -> None:
    canonical_ids = tuple(str(record["record_id"]) for record in canonical)
    evaluation_ids = tuple(str(record["episode_id"]) for record in evaluation)
    if canonical_ids != evaluation_ids or len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("canonical and evaluation MARS episode identities do not align")


def _ordered_worlds(records: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(record["world_id"]) for record in records))


def _convert_worlds(
    canonical_by_id: dict[str, dict[str, Any]],
    evaluation_by_id: dict[str, dict[str, Any]],
    *,
    split: str,
    worlds: tuple[str, ...],
) -> tuple[TrajectoryStep, ...]:
    output = []
    selected_worlds = set(worlds)
    for episode_id, canonical in canonical_by_id.items():
        world_id = str(canonical["world_id"])
        if world_id not in selected_worlds:
            continue
        value = evaluation_by_id[episode_id]
        evidence_values = value["backend"].get("evidence", [])
        if not isinstance(evidence_values, list):
            raise ValueError("MARS evaluation evidence must be a list")
        support = _supported_source_ids(canonical)
        support_rank = {source_id: index for index, source_id in enumerate(support)}
        evidence = tuple(
            _evidence(item, support_rank=support_rank.get(str(item.get("evidence_id"))))
            for item in evidence_values
            if item.get("model_visible") is True
        )
        handles = tuple(
            hit.handle for hit in evidence if hit.source_id in support and hit.score >= 0.72
        )
        expected = canonical["expected_outcome"]
        if not isinstance(expected, dict):
            raise ValueError("MARS expected outcome must be an object")
        claims = canonical["claims"]
        if not isinstance(claims, list):
            raise ValueError("MARS claims must be a list")
        conflicting = any(claim.get("status") == "disputed" for claim in claims)
        blocked = (
            bool(expected.get("requires_abstention", False))
            or expected.get("action_state") in {"denied", "pending"}
            or value["expected_outcome"].get("action_state")
            in {"denied", "approval_pending"}
        )
        if blocked or conflicting:
            action = ControlAction.ABSTAIN
            handles = ()
        elif handles:
            action = ControlAction.ANSWER
        else:
            action = ControlAction.REQUEST_EVIDENCE
        identity = hashlib.sha256(f"{split}:{world_id}:{episode_id}".encode()).hexdigest()[:16]
        output.append(
            TrajectoryStep(
                trajectory_id=f"trajectory_{identity}",
                scenario_id=f"scenario_{identity}",
                step_index=0,
                query=str(value["instruction"]),
                generation_id="generation_mars_v2",
                evidence=evidence,
                approximate=False,
                conflicting=conflicting,
                blocked=blocked,
                target=ControlTarget(action, handles),
                provenance=f"mars-v2:{split}:{world_id}",
            )
        )
    expected_records = len(worlds) * RECORDS_PER_WORLD
    if len(output) != expected_records:
        raise ValueError(f"governed {split} split has incomplete MARS worlds")
    return tuple(output)


def _supported_source_ids(canonical: dict[str, Any]) -> tuple[str, ...]:
    result = []
    for claim in canonical["claims"]:
        if not isinstance(claim, dict) or claim.get("status") != "supported":
            continue
        support = claim.get("support", [])
        if not isinstance(support, list):
            raise ValueError("MARS claim support must be a list")
        for locator in support:
            source_id = str(locator).split("#", 1)[0]
            if source_id not in result:
                result.append(source_id)
    return tuple(result)


def _evidence(value: dict[str, Any], *, support_rank: int | None) -> EvidenceHit:
    text = str(value["content"])
    digest = value["digest"]
    if not isinstance(digest, dict) or digest.get("algorithm") != "sha256":
        raise ValueError("MARS evidence digest is invalid")
    if hashlib.sha256(text.encode()).hexdigest() != digest.get("value"):
        raise ValueError("MARS evidence content digest does not match")
    source_id = str(value["evidence_id"])
    handle = f"passage_{hashlib.sha256(source_id.encode()).hexdigest()[:16]}"
    return EvidenceHit(
        handle=handle,
        source_id=source_id,
        source_version=str(value["revision"]),
        text=text,
        score=0.4 if support_rank is None else max(0.72, 0.95 - support_rank * 0.1),
        content_digest=str(digest["value"]),
        trusted=True,
        active=True,
    )


def _verify_digest(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned MARS source is absent or unsafe: {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"pinned MARS source digest mismatch: {path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
