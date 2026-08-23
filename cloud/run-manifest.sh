#!/bin/sh
set -eu

: "${MANIFEST_PATH:?Set MANIFEST_PATH to the mounted manifest JSON}"
: "${REGISTRY_ROOT:?Set REGISTRY_ROOT to the persistent artifact directory}"

if [ -n "${DATA_ROOT:-}" ]; then
  exec uv run --no-sync hyphae-transformer run-manifest "$MANIFEST_PATH" \
    --registry "$REGISTRY_ROOT" --data-root "$DATA_ROOT"
fi

exec uv run --no-sync hyphae-transformer run-manifest "$MANIFEST_PATH" --registry "$REGISTRY_ROOT"
