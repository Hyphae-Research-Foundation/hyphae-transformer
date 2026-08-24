from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.module_from_spec(
    spec := importlib.util.spec_from_file_location(
        "hyphae_210_certification", ROOT / "scripts" / "hyphae_210_certification.py"
    )
)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_certification_pins_exact_hyphae_tag() -> None:
    assert module.HYPHAE_VERSION == "2.1.0"
    assert module.HYPHAE_TAG_COMMIT == "34b939fc0064b701cc2b34cf6f3a1f07d743638d"
    assert module.HYPHAE_TAG_TREE == "2f2032f79cb26efa6a928aeafeb52023f5dd572e"


def test_certification_runs_generation_routing_canary() -> None:
    source = (ROOT / "scripts" / "hyphae_210_certification.py").read_text()
    conformance = (ROOT / "scripts" / "hyphae_knowledge_conformance.py").read_text()
    assert '"--routing-database"' in source
    assert "authority.verify_candidate" in conformance
    assert "authority.activate" in conformance
    assert "GenerationRoutedRetriever" in conformance
    assert "HYPHAE_210_RETRIEVAL_PROFILE" in conformance
    assert "GenerationRoutedEvidenceProvider" in conformance
    assert "DurableFinalizationWorker" in conformance
    assert "SupervisedFrozenGemmaRuntime" in conformance
    assert "SQLiteMailboxNotificationSink" in conformance
