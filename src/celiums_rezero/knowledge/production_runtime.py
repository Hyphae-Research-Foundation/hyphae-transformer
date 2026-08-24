"""Strict owner-configured assembly for a finalization-only production worker."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.runtime import (
    QUOTED_RUNTIME_VERSION,
    quoted_runtime_manifest_sha256,
)
from celiums_rezero.knowledge.embedding import EmbeddingProvider
from celiums_rezero.knowledge.finalization import (
    DurableFinalizationWorker,
    FinalizationPolicy,
)
from celiums_rezero.knowledge.generation import GenerationAuthority
from celiums_rezero.knowledge.model_runtime import (
    SupervisedFrozenGemmaRuntime,
    SupervisedFrozenRuntimeConfig,
)
from celiums_rezero.knowledge.notifications import (
    SQLiteMailboxConfig,
    SQLiteMailboxNotificationSink,
)
from celiums_rezero.knowledge.orchestration import (
    FrozenModelIdentity,
    GenerationRoutedEvidenceProvider,
    HostGemmaAnswerer,
)
from celiums_rezero.knowledge.publication import PublicationReceiptStore
from celiums_rezero.knowledge.retrieval import (
    HYPHAE_210_RETRIEVAL_PROFILE,
    GenerationRoutedRetriever,
)
from celiums_rezero.knowledge.schemas import SufficiencyPolicy, TenantId
from celiums_rezero.knowledge.store import SQLiteTenantStore


class ManagedHyphaeClient(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(self, *exc_info: object) -> None: ...


@dataclass(frozen=True, slots=True)
class ProductionFinalizationRuntime:
    tenant: TenantId
    store: SQLiteTenantStore
    config: dict[str, object]

    def __post_init__(self) -> None:
        _parse_config(self.config)

    def run_once(self) -> int:
        config = _parse_config(self.config)
        embedder = _provider(
            _text(config["embedder_factory"], "embedder factory"),
            tenant=self.tenant,
            config=_object_value(config["embedder_config"], "embedder config"),
        )
        if not hasattr(embedder, "embed") or not isinstance(
            getattr(embedder, "profile", None), str
        ) or isinstance(getattr(embedder, "dimensions", None), bool) or not isinstance(
            getattr(embedder, "dimensions", None), int
        ):
            raise TypeError("embedder factory returned an invalid provider")
        client_factory = _load_callable(
            _text(config["hyphae_client_factory"], "Hyphae client factory")
        )
        client_context = client_factory(
            tenant=self.tenant,
            config=_object_value(config["hyphae_config"], "Hyphae config"),
        )
        if not hasattr(client_context, "__enter__") or not hasattr(client_context, "__exit__"):
            raise TypeError("Hyphae client factory returned no managed client")
        receipts = PublicationReceiptStore(
            _absolute_path(config["receipt_root"], "receipt root")
        )
        with cast(ManagedHyphaeClient, client_context) as client:
            authority = GenerationAuthority(self.store, receipts=receipts)
            retriever = GenerationRoutedRetriever(
                tenant=self.tenant,
                authority=authority,
                client=cast(Any, client),
                profile=HYPHAE_210_RETRIEVAL_PROFILE,
                embedder=cast(EmbeddingProvider, embedder),
                request_options_factory=_request_options_factory(),
            )
            model = _model(_object_value(config["model"], "model"))
            mailbox = SQLiteMailboxNotificationSink(
                SQLiteMailboxConfig(
                    tenant_id=self.tenant.value,
                    path=_absolute_path(config["mailbox_path"], "mailbox path"),
                    mailbox_id=_text(config["mailbox_id"], "mailbox ID"),
                )
            )
            policy = _finalization_policy(
                _object_value(config["finalization"], "finalization")
            )
            finalizer = DurableFinalizationWorker(
                store=self.store,
                worker_id=_text(config["worker_id"], "worker ID"),
                lease_seconds=_number(config["lease_seconds"], "lease seconds"),
                answerer=HostGemmaAnswerer(
                    runtime=model,
                    evidence=GenerationRoutedEvidenceProvider(retriever=retriever),
                    expected_identity=model.identity,
                    sufficiency=SufficiencyPolicy(),
                ),
                sink=mailbox,
                policy=policy,
            )
            completed = finalizer.run_next()
            return int(completed is not None)


def factory(
    *, tenant: TenantId, store: SQLiteTenantStore, config: dict[str, object]
) -> ProductionFinalizationRuntime:
    return ProductionFinalizationRuntime(tenant, store, config)


_ROOT_FIELDS = {
    "schema",
    "worker_id",
    "lease_seconds",
    "receipt_root",
    "embedder_factory",
    "embedder_config",
    "hyphae_client_factory",
    "hyphae_config",
    "model",
    "mailbox_path",
    "mailbox_id",
    "finalization",
}


def _parse_config(config: dict[str, object]) -> dict[str, object]:
    values = _object(config, _ROOT_FIELDS, "production runtime")
    _text(values["worker_id"], "worker ID")
    _number(values["lease_seconds"], "lease seconds")
    _absolute_path(values["receipt_root"], "receipt root")
    _text(values["embedder_factory"], "embedder factory")
    _object_value(values["embedder_config"], "embedder config")
    _text(values["hyphae_client_factory"], "Hyphae client factory")
    _object_value(values["hyphae_config"], "Hyphae config")
    _absolute_path(values["mailbox_path"], "mailbox path")
    _text(values["mailbox_id"], "mailbox ID")
    _model_config(_object_value(values["model"], "model"))
    policy = _finalization_policy(
        _object_value(values["finalization"], "finalization")
    )
    if _number(values["lease_seconds"], "lease seconds") < max(
        policy.answer_timeout_seconds,
        policy.notification_timeout_seconds,
    ) + policy.lease_safety_seconds:
        raise ValueError("production runtime lease is too short")
    return values


def _model(config: dict[str, object]) -> SupervisedFrozenGemmaRuntime:
    values = _model_config(config)
    bundle_sha256 = _digest(values["bundle_sha256"], "bundle SHA-256")
    manifest = quoted_runtime_manifest_sha256(bundle_sha256)
    identity = FrozenModelIdentity(
        Gemma4E4BFrozenBackbone.model_id,
        Gemma4E4BFrozenBackbone.revision,
        manifest,
        QUOTED_RUNTIME_VERSION,
    )
    return SupervisedFrozenGemmaRuntime(
        SupervisedFrozenRuntimeConfig(
            executable=_absolute_path(values["executable"], "model executable"),
            executable_sha256=_digest(
                values["executable_sha256"], "model executable SHA-256"
            ),
            identity=identity,
            arguments=(
                "--model",
                str(_absolute_path(values["model_path"], "model path")),
                "--bundle",
                str(_absolute_path(values["bundle_path"], "bundle path")),
                "--bundle-sha256",
                bundle_sha256,
                "--runtime-manifest-sha256",
                manifest,
                "--device",
                _text(values["device"], "model device"),
            ),
        )
    )


def _model_config(config: dict[str, object]) -> dict[str, object]:
    fields = {
        "executable",
        "executable_sha256",
        "model_path",
        "bundle_path",
        "bundle_sha256",
        "device",
    }
    values = _object(config, fields, "model")
    _absolute_path(values["executable"], "model executable")
    _digest(values["executable_sha256"], "model executable SHA-256")
    _absolute_path(values["model_path"], "model path")
    _absolute_path(values["bundle_path"], "bundle path")
    _digest(values["bundle_sha256"], "bundle SHA-256")
    _text(values["device"], "model device")
    return values


def _request_options_factory() -> Any:
    try:
        module = importlib.import_module("hyphae_sdk.v2")
    except ModuleNotFoundError as error:
        raise RuntimeError("production runtime requires hyphae-sdk==2.1.0") from error
    request_options = getattr(module, "RequestOptions", None)
    if request_options is None:
        raise RuntimeError("Hyphae SDK has no v2 request options")

    def factory(timeout_seconds: float) -> object:
        return request_options(
            deadline_micros=int((time.time() + timeout_seconds) * 1_000_000)
        )

    return factory


def _provider(reference: str, **kwargs: object) -> object:
    return _load_callable(reference)(**kwargs)


def _load_callable(reference: str) -> Any:
    if reference.count(":") != 1:
        raise ValueError("provider factory must be module:factory")
    module_name, factory_name = reference.split(":", 1)
    factory_value = getattr(importlib.import_module(module_name), factory_name, None)
    if not callable(factory_value):
        raise TypeError("provider factory is absent or not callable")
    return factory_value


def _finalization_policy(config: dict[str, object]) -> FinalizationPolicy:
    fields = {
        "answer_timeout_seconds",
        "notification_timeout_seconds",
        "lease_safety_seconds",
        "retry_base_seconds",
        "retry_max_seconds",
        "max_answer_failures",
        "max_notification_failures",
    }
    values = _object(config, fields, "finalization")
    return FinalizationPolicy(
        answer_timeout_seconds=_number(values["answer_timeout_seconds"], "answer timeout"),
        notification_timeout_seconds=_number(
            values["notification_timeout_seconds"], "notification timeout"
        ),
        lease_safety_seconds=_number(values["lease_safety_seconds"], "lease safety"),
        retry_base_seconds=_number(values["retry_base_seconds"], "retry base"),
        retry_max_seconds=_number(values["retry_max_seconds"], "retry maximum"),
        max_answer_failures=_integer(values["max_answer_failures"], "answer failures"),
        max_notification_failures=_integer(
            values["max_notification_failures"], "notification failures"
        ),
    )


def _object(value: dict[str, object], fields: set[str], name: str) -> dict[str, object]:
    if set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    if name == "production runtime" and value.get("schema") != (
        "hyphae-transformer.production-finalization/v1"
    ):
        raise ValueError("production runtime schema is invalid")
    return value


def _object_value(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _absolute_path(value: object, name: str) -> Path:
    path = Path(_text(value, name))
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be an absolute normalized path")
    return path
