"""Owner-configured tenant runtime adapter boundary for one-shot workers."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Protocol, cast

from celiums_rezero.knowledge.schemas import TenantId
from celiums_rezero.knowledge.store import SQLiteTenantStore

_ADAPTER_PATTERN = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*$"
)


class TenantWorkerRuntime(Protocol):
    def run_once(self) -> int: ...


def load_tenant_runtime(
    *,
    adapter: str,
    tenant: TenantId,
    database: Path,
    config: dict[str, object],
) -> TenantWorkerRuntime:
    if not _ADAPTER_PATTERN.fullmatch(adapter):
        raise ValueError("tenant adapter must be a canonical module:factory reference")
    module_name, factory_name = adapter.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError("tenant adapter factory is absent or not callable")
    store = SQLiteTenantStore(database, tenant=tenant)
    runtime = factory(tenant=tenant, store=store, config=config)
    if not hasattr(runtime, "run_once") or not callable(runtime.run_once):
        raise TypeError("tenant adapter did not return a one-shot runtime")
    return cast(TenantWorkerRuntime, runtime)
