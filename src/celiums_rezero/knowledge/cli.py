"""Minimal operational CLI for tenant knowledge authorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from celiums_rezero.knowledge.runtime import load_tenant_runtime
from celiums_rezero.knowledge.schemas import TenantId
from celiums_rezero.knowledge.store import SQLiteTenantStore


def main() -> int:
    parser = argparse.ArgumentParser(prog="celiums-knowledge")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--database", type=Path, required=True)
    preflight.add_argument("--tenant", required=True)
    worker = commands.add_parser("worker-once")
    worker.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        store = SQLiteTenantStore(arguments.database, tenant=TenantId(arguments.tenant))
        print(json.dumps({"status": "ready", "pragmas": store.pragmas()}, sort_keys=True))
        return 0
    config = json.loads(arguments.config.read_text())
    fields = {"tenant", "database", "adapter", "adapter_config"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError(
            "worker config must contain tenant, database, adapter, and adapter_config"
        )
    adapter_config = config["adapter_config"]
    if not isinstance(adapter_config, dict):
        raise ValueError("worker adapter_config must be an object")
    runtime = load_tenant_runtime(
        adapter=str(config["adapter"]),
        tenant=TenantId(str(config["tenant"])),
        database=Path(str(config["database"])),
        config=adapter_config,
    )
    processed = runtime.run_once()
    if isinstance(processed, bool) or not isinstance(processed, int) or processed < 0:
        raise TypeError("tenant runtime returned an invalid processed-work count")
    print(json.dumps({"status": "completed", "processed": processed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
