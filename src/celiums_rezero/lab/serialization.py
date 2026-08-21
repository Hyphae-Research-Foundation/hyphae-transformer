"""Canonical JSON serialization and stable content identifiers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any


def to_primitive(value: object) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, set):
        return [to_primitive(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(
        to_primitive(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: object, *, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return digest[:length]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        to_primitive(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ) + "\n"
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
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
