#!/usr/bin/env python3
"""Download the pinned Gemma 4 E4B checkpoint and verify every required artifact."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MINIMUM_FREE_BYTES = 64 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < MINIMUM_FREE_BYTES:
        raise RuntimeError("less than 64 GiB is free for the pinned Gemma checkpoint")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=output,
        allow_patterns=list(FILES),
    )
    preflight = Path(__file__).with_name("preflight_gemma4_e4b.py")
    return subprocess.run(
        [sys.executable, str(preflight), "--model", str(output)], check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
