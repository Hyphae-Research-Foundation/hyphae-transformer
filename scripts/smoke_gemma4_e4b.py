#!/usr/bin/env python3
"""Run the preregistered MI355X one-step batch-size smoke for frozen Gemma E4B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from celiums_rezero.governed.data import load_governed_dataset, make_batch
from celiums_rezero.governed.gemma4 import Gemma4E4BFrozenBackbone
from celiums_rezero.governed.model import GovernedControlHead
from celiums_rezero.governed.schemas import (
    DatasetSplit,
    GovernedDataset,
    GovernedDatasetManifest,
)
from celiums_rezero.knowledge.schemas import SufficiencyPolicy

EXPECTED_DATASET_ID = "gtd_b7161eb4c1cf007dca96741ad8acffbe25cd9b6b46681fd48f19f21fde29332f"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--max-vram-gib", type=float, required=True)
    arguments = parser.parse_args()
    if arguments.batch_sizes != [1, 2, 4, 8] or arguments.max_vram_gib != 240:
        raise ValueError("smoke settings differ from the preregistration")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("the Gemma E4B smoke requires a ROCm GPU")
    if torch.cuda.device_count() != 1 or "MI355" not in torch.cuda.get_device_name(0):
        raise RuntimeError("the Gemma E4B smoke requires one AMD MI355X")

    dataset = _load_dataset(arguments.dataset)
    device = torch.device("cuda:0")
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats(device)
    backbone = Gemma4E4BFrozenBackbone(arguments.model, device=str(device))
    state_before = backbone.state_fingerprint()
    results = []
    for batch_size in arguments.batch_sizes:
        torch.manual_seed(17)
        torch.cuda.reset_peak_memory_stats(device)
        head = GovernedControlHead(backbone.hidden_size).to(device)
        batch = make_batch(
            dataset.train[:batch_size],
            backbone,
            maximum_evidence_items=dataset.manifest.maximum_evidence_items,
            device=device,
        )
        logits = head(batch.context, batch.evidence, batch.evidence_mask)
        action_loss = nn.functional.cross_entropy(logits.action_logits, batch.action_targets)
        finite_pointers = logits.evidence_logits.masked_fill(~batch.evidence_mask, 0)
        pointer_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                finite_pointers[batch.evidence_mask],
                batch.pointer_targets[batch.evidence_mask],
            )
            if batch.evidence_mask.any()
            else torch.zeros((), device=device)
        )
        loss = action_loss + pointer_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Gemma E4B smoke loss is non-finite")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ):
            raise RuntimeError("Gemma E4B smoke gradient is non-finite")
        peak = torch.cuda.max_memory_allocated(device)
        results.append(
            {
                "batch_size": batch_size,
                "loss": float(loss.detach()),
                "peak_vram_bytes": peak,
            }
        )
        del batch, head, logits, loss
        torch.cuda.empty_cache()
    peak = max(int(result["peak_vram_bytes"]) for result in results)
    maximum = int(arguments.max_vram_gib * 1024**3)
    if peak > maximum:
        raise RuntimeError(f"Gemma E4B smoke peak VRAM {peak} exceeds {maximum}")
    if backbone.state_fingerprint() != state_before:
        raise RuntimeError("Gemma E4B backbone changed during smoke")
    payload = {
        "schema": "hyphae-transformer.gemma4-e4b-smoke/v1",
        "model_id": backbone.model_id,
        "model_revision": backbone.revision,
        "dataset_id": dataset.manifest.dataset_id,
        "gpu": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "batch_results": results,
        "peak_vram_bytes": peak,
        "max_vram_bytes": maximum,
        "backbone_unchanged": True,
        "passed": True,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_dataset(root: Path) -> GovernedDataset:
    values = json.loads((root / "manifest.json").read_text())
    if values.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("Gemma smoke dataset ID differs from preregistration")
    manifest = GovernedDatasetManifest(
        splits=tuple(
            (name, DatasetSplit(**item)) for name, item in sorted(values["splits"].items())
        ),
        policy=SufficiencyPolicy(**values["policy"]),
        maximum_evidence_items=values["maximum_evidence_items"],
        dataset_id=values["dataset_id"],
    )
    return load_governed_dataset(root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
