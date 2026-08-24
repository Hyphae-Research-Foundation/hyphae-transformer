#!/usr/bin/env python3
"""Download the pinned MiniLM embedding model and verify its exact artifact set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from celiums_rezero.knowledge.embedding import (
    MiniLML6V2EmbeddingProvider,
    minilm_l6_v2_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MiniLML6V2EmbeddingProvider.model_id,
        revision=MiniLML6V2EmbeddingProvider.revision,
        local_dir=arguments.out.resolve(),
        allow_patterns=list(MiniLML6V2EmbeddingProvider.required_artifacts),
    )
    print(json.dumps(minilm_l6_v2_preflight(arguments.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
