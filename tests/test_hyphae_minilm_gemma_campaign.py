from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.module_from_spec(
    spec := importlib.util.spec_from_file_location(
        "unified_canary", ROOT / "scripts" / "run_hyphae_minilm_gemma_canary.py"
    )
)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_unified_canary_pins_all_component_identities() -> None:
    assert module.HYPHAE_ARCHIVE_SHA256 == (
        "a1e8cf56d9b9a96ee5f230aa4dec92b2541792f7ca4bb40c0dbf761d9ed3e0aa"
    )
    assert module.HYPHAE_BINARY_SHA256 == (
        "a00ea0cfc502ad63d65c42357664f7664f8a8c482fbdeb24a4f5511feceb45d0"
    )
    assert module.HYPHAE_WHEEL_SHA256 == (
        "fd6503abbcac18db9a6705682b80a83904389f146e6dd0c4d17fdef49535a5fb"
    )
    assert module.COLLECTION_SHA256 == (
        "181552f7f9666546db8f09b3e89be98e99f4c4e09be227f6d257da93029ea527"
    )
    assert module.BUNDLE_SHA256 == (
        "93db742ead71c12fa46c62661b12108fdb0a815d3b5fcf180821538dcfc8b9be"
    )


def test_unified_canary_uses_real_components_and_safe_cleanup() -> None:
    source = (ROOT / "scripts" / "run_hyphae_minilm_gemma_canary.py").read_text()
    assert "MiniLML6V2EmbeddingProvider" in source
    assert "DurableAcquisitionWorker" in source
    assert "authority.verify_candidate" in source
    assert "GenerationRoutedRetriever" in source
    assert "SupervisedFrozenGemmaRuntime" in source
    assert "SQLiteMailboxNotificationSink" in source
    assert "shutil.rmtree(arguments.work_root" in source
