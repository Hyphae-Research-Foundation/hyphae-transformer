from __future__ import annotations

import json
from pathlib import Path

import pytest

from celiums_rezero.knowledge.production_runtime import factory
from celiums_rezero.knowledge.schemas import TenantId
from celiums_rezero.knowledge.store import SQLiteTenantStore

ROOT = Path(__file__).resolve().parents[1]


def test_example_production_runtime_config_is_strictly_accepted(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    outer = json.loads((ROOT / "deploy" / "tenant-finalization.example.json").read_text())
    model = outer["adapter_config"]["model"]
    model["executable_sha256"] = "a" * 64
    store = SQLiteTenantStore(
        tmp_path / "jobs.sqlite3",
        tenant=TenantId("tenant_a"),
    )
    runtime = factory(
        tenant=TenantId("tenant_a"),
        store=store,
        config=outer["adapter_config"],
    )
    assert runtime.tenant == TenantId("tenant_a")


def test_production_runtime_rejects_unknown_fields_and_short_lease(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    outer = json.loads((ROOT / "deploy" / "tenant-finalization.example.json").read_text())
    outer["adapter_config"]["model"]["executable_sha256"] = "a" * 64
    store = SQLiteTenantStore(
        tmp_path / "jobs.sqlite3",
        tenant=TenantId("tenant_a"),
    )
    unknown = dict(outer["adapter_config"])
    unknown["ignored"] = True
    with pytest.raises(ValueError, match="fields"):
        factory(tenant=TenantId("tenant_a"), store=store, config=unknown)
    short = dict(outer["adapter_config"])
    short["lease_seconds"] = 1
    with pytest.raises(ValueError, match="lease"):
        factory(tenant=TenantId("tenant_a"), store=store, config=short)
