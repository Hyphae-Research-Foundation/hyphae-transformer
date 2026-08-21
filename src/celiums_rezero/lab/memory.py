"""Evidence-backed causal memory with conservative confidence aggregation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from celiums_rezero.lab.registry import Registry
from celiums_rezero.lab.schemas import MemoryEntry, MemoryRelation
from celiums_rezero.lab.serialization import read_json


def load_memory(root: Path | str) -> list[MemoryEntry]:
    registry = Registry(root)
    entries: list[MemoryEntry] = []
    for path in sorted(registry.memory.glob("M-*.json")):
        values = read_json(path)
        entries.append(
            MemoryEntry(
                statement=values["statement"],
                relation=MemoryRelation(values["relation"]),
                conditions=values["conditions"],
                evidence_run_ids=tuple(values["evidence_run_ids"]),
                confidence=float(values["confidence"]),
                entry_id=values["entry_id"],
            )
        )
    return entries


def summarize_memory(entries: list[MemoryEntry]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        counts[entry.relation.value] += 1
    return dict(sorted(counts.items()))
