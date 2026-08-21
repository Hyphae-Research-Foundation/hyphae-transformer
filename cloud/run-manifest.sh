#!/bin/sh
set -eu

: "${MANIFEST_PATH:?Set MANIFEST_PATH to the mounted manifest JSON}"
: "${REGISTRY_ROOT:?Set REGISTRY_ROOT to the persistent artifact directory}"

exec uv run --no-sync celiums-rezero run-manifest "$MANIFEST_PATH" --registry "$REGISTRY_ROOT"
