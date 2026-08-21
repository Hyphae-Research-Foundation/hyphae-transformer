"""Append-only filesystem registry for hypotheses and run evidence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

from celiums_rezero.lab.schemas import Hypothesis, MemoryEntry, RunManifest, RunResult
from celiums_rezero.lab.serialization import canonical_json, read_json


class Registry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.hypotheses = self.root / "hypotheses"
        self.runs = self.root / "runs"
        self.memory = self.root / "memory"
        for path in (self.hypotheses, self.runs, self.memory):
            path.mkdir(parents=True, exist_ok=True)

    def register_hypothesis(self, hypothesis: Hypothesis) -> Path:
        assert hypothesis.hypothesis_id is not None
        path = self.hypotheses / f"{hypothesis.hypothesis_id}.json"
        self._write_once(path, hypothesis)
        return path

    def register_run(self, manifest: RunManifest) -> Path:
        assert manifest.run_id is not None
        self._validate_id(manifest.run_id, "R-")
        run_directory = self.runs / manifest.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        path = run_directory / "manifest.json"
        self._write_once(path, manifest)
        return path

    def complete_run(self, result: RunResult) -> Path:
        self._validate_id(result.run_id, "R-")
        run_directory = self.runs / result.run_id
        if not (run_directory / "manifest.json").exists():
            raise FileNotFoundError(f"run {result.run_id} is not registered")
        path = run_directory / "result.json"
        self._write_once(path, result)
        return path

    def add_memory(self, entry: MemoryEntry) -> Path:
        assert entry.entry_id is not None
        path = self.memory / f"{entry.entry_id}.json"
        self._write_once(path, entry)
        return path

    def run_manifest(self, run_id: str) -> dict[str, object]:
        self._validate_id(run_id, "R-")
        return cast(dict[str, object], read_json(self.runs / run_id / "manifest.json"))

    def run_result(self, run_id: str) -> dict[str, object] | None:
        self._validate_id(run_id, "R-")
        path = self.runs / run_id / "result.json"
        return cast(dict[str, object], read_json(path)) if path.exists() else None

    def list_run_ids(self) -> list[str]:
        return sorted(path.name for path in self.runs.iterdir() if path.is_dir())

    def _write_once(self, path: Path, value: object) -> None:
        if path.exists():
            if canonical_json(read_json(path)) != canonical_json(value):
                raise FileExistsError(f"immutable registry record differs: {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(value) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)
        except FileExistsError:
            if canonical_json(read_json(path)) != canonical_json(value):
                raise FileExistsError(f"immutable registry record differs: {path}") from None
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_id(value: str, prefix: str) -> None:
        if not value.startswith(prefix) or not value.removeprefix(prefix).isalnum():
            raise ValueError(f"unsafe registry identifier: {value}")
