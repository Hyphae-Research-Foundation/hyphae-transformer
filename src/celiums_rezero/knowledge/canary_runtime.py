"""Deterministic subprocess fixture for the supervised quoted-runtime boundary."""

from __future__ import annotations

import json
import sys

from celiums_rezero.knowledge.model_runtime import RESPONSE_SCHEMA
from celiums_rezero.lab.serialization import canonical_json


def main() -> int:
    request = json.load(sys.stdin)
    passage = request["passages"][0]
    print(
        canonical_json(
            {
                "schema": RESPONSE_SCHEMA,
                "request_id": request["request_id"],
                "identity": request["identity"],
                "decision": "answer",
                "claims": [
                    {
                        "handle": passage["handle"],
                        "quote": passage["text"],
                    }
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
