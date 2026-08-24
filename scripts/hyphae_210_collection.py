#!/usr/bin/env python3
"""Create the exact Hyphae 2.1.0 conformance collection through Native v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct

from hyphae_sdk.v2 import HyphaeClient, RequestOptions


def framed(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def name(value: str) -> bytes:
    encoded = value.encode()
    return framed(encoded) + framed(encoded)


def header(
    *, kind: int, object_id: int, owner: int, object_name: str, parent: int | None
) -> bytearray:
    value = bytearray(b"HYCOBJ02")
    value.extend((kind, 2))
    value.extend(object_id.to_bytes(16, "little"))
    value.append(owner)
    value.extend(name("main"))
    value.extend(name("public"))
    value.extend(name(object_name))
    value.append(parent is not None)
    if parent is not None:
        value.extend(parent.to_bytes(16, "little"))
    value.extend(struct.pack("<Q", 1))
    return value


def definitions(*, vector_dimensions: int = 2) -> tuple[bytes, ...]:
    if not 1 <= vector_dimensions <= 4096:
        raise ValueError("vector dimensions must be in [1, 4096]")
    database = bytes(
        header(kind=1, object_id=10, owner=0, object_name="database", parent=None)
    )
    schema = bytes(
        header(kind=2, object_id=11, owner=0, object_name="schema", parent=10)
    )
    analyzer = header(
        kind=8,
        object_id=12,
        owner=3,
        object_name="hyphae_transformer_analyzer",
        parent=11,
    )
    analyzer.append(1)
    analyzer.extend(struct.pack("<I", 1))
    analyzer.append(1)
    collection = header(
        kind=7,
        object_id=13,
        owner=3,
        object_name="hyphae_transformer_knowledge",
        parent=11,
    )
    text = b"\x07"
    signed_i64 = b"\x02\x40"
    fields = (
        (1, "body", text, 12, 2),
        (2, "source_id", text, None, 1),
        (3, "source_version", text, None, 1),
        (4, "content_digest", text, None, 1),
        (5, "corpus_generation", text, None, 1),
        (6, "byte_start", signed_i64, None, 1),
        (7, "byte_end", signed_i64, None, 1),
        (8, "chunk_ordinal", signed_i64, None, 1),
    )
    collection.extend(struct.pack("<I", len(fields)))
    for field_id, field_name, logical_type, analyzer_id, lexical_policy in fields:
        collection.extend(struct.pack("<I", field_id))
        collection.extend(name(field_name))
        collection.extend(framed(logical_type))
        collection.append(analyzer_id is not None)
        if analyzer_id is not None:
            collection.extend(analyzer_id.to_bytes(16, "little"))
        collection.extend((1, 1, 2, lexical_policy))
    collection.extend(struct.pack("<I", 1))
    collection.extend(struct.pack("<I", 9))
    collection.extend(name("semantic"))
    collection.append(1)
    collection.extend(struct.pack("<H", vector_dimensions))
    collection.append(3)
    collection.append(1)
    collection.extend(struct.pack("<IHH", 1000, 4, 2))
    return database, schema, bytes(analyzer), bytes(collection)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("--vector-dimensions", type=int, default=2)
    arguments = parser.parse_args()
    endpoint = arguments.endpoint
    values = definitions(vector_dimensions=arguments.vector_dimensions)
    with HyphaeClient.local(endpoint) as client:
        for request_id, definition in enumerate(values, start=1001):
            response = client.catalog(
                "create",
                {"definition": definition},
                options=RequestOptions(request_id=request_id, durability="strict"),
            )
            if response.kind != "catalog_created":
                raise RuntimeError("unexpected Hyphae catalog response")
        described = client.catalog(
            "describe", {"id": 13}, options=RequestOptions(request_id=1100)
        )
        if described.kind != "catalog_definition" or described.value != values[-1]:
            raise RuntimeError("Hyphae canonical collection definition differs")
    print(
        json.dumps(
            {
                "collection_definition_sha256": hashlib.sha256(values[-1]).hexdigest(),
                "vector_dimensions": arguments.vector_dimensions,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
