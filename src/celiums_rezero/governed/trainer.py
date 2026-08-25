"""Deterministic head-only training over frozen trajectory features."""

from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import torch
from torch import nn

from celiums_rezero.core.gates import is_gate_parameter
from celiums_rezero.core.optim import build_optimizer_groups
from celiums_rezero.governed.backbone import FrozenTextBackbone
from celiums_rezero.governed.data import GovernedBatch, make_batch
from celiums_rezero.governed.schemas import TrajectoryStep
from celiums_rezero.lab.serialization import canonical_json


@dataclass(frozen=True, slots=True)
class ControlTrainConfig:
    epochs: int = 50
    learning_rate: float = 0.01
    evidence_loss_weight: float = 1.0
    gradient_clip: float = 1.0
    seed: int = 17
    device: str = "cpu"
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    pointer_loss_scope: str = "all"

    def __post_init__(self) -> None:
        values = (self.learning_rate, self.evidence_loss_weight, self.gradient_clip)
        if self.epochs < 1 or any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("control training configuration is invalid")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("control optimizer is invalid")
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("control weight decay is invalid")
        if self.pointer_loss_scope not in {"all", "answer"}:
            raise ValueError("control pointer loss scope is invalid")


@dataclass(frozen=True, slots=True)
class ControlTrainSummary:
    initial_loss: float
    final_loss: float
    best_loss: float
    epochs: int
    checkpoint_sha256: str
    backbone_unchanged: bool


def train_control_head(
    backbone: FrozenTextBackbone,
    head: nn.Module,
    records: tuple[TrajectoryStep, ...],
    config: ControlTrainConfig,
    *,
    checkpoint: Path,
    maximum_evidence_items: int = 8,
    deadline: float | None = None,
    batch: GovernedBatch | None = None,
) -> ControlTrainSummary:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config.device)
    head.to(device).train()
    has_residual_gates = any(is_gate_parameter(parameter) for parameter in head.parameters())
    optimizer = (
        torch.optim.AdamW(
            build_optimizer_groups(
                head,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.optimizer == "adamw" and has_residual_gates
        else torch.optim.AdamW(
            head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        if config.optimizer == "adamw"
        else torch.optim.SGD(head.parameters(), lr=config.learning_rate)
    )
    before = canonical_json(backbone.identity)
    state_before = backbone.state_fingerprint()
    losses: list[float] = []
    if batch is None:
        batch = make_batch(
            records,
            backbone,
            maximum_evidence_items=maximum_evidence_items,
            device=device,
        )
    elif batch.action_targets.shape != (len(records),) or batch.context.device != device:
        raise ValueError("precomputed governed training batch is invalid")
    for _ in range(config.epochs):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("governed training wall-time budget exceeded")
        optimizer.zero_grad(set_to_none=True)
        logits = head(
            batch.context,
            batch.evidence,
            batch.evidence_mask,
            batch.evidence_scores,
            batch.host_control_features,
        )
        action_loss = nn.functional.cross_entropy(logits.action_logits, batch.action_targets)
        finite = logits.evidence_logits.masked_fill(~batch.evidence_mask, 0)
        pointer_rows = batch.evidence_mask
        if config.pointer_loss_scope == "answer":
            pointer_rows = pointer_rows & (batch.action_targets == 0).unsqueeze(-1)
        pointer_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                finite[pointer_rows], batch.pointer_targets[pointer_rows]
            )
            if pointer_rows.any()
            else torch.zeros((), device=device)
        )
        loss = action_loss + config.evidence_loss_weight * pointer_loss
        if not torch.isfinite(loss):
            raise RuntimeError("control training loss is non-finite")
        torch.autograd.backward(loss)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ):
            raise RuntimeError("control head gradient is non-finite")
        nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in head.parameters()):
            raise RuntimeError("control head parameter is non-finite")
        losses.append(float(loss.detach()))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "version": 1,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "backbone": before,
            "backbone_state": state_before,
            "maximum_evidence_items": maximum_evidence_items,
            "action_order": ["answer", "request_evidence", "abstain"],
            "record_digest": hashlib.sha256(
                canonical_json(records).encode()
            ).hexdigest(),
        },
        temporary,
    )
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return ControlTrainSummary(
        initial_loss=losses[0],
        final_loss=losses[-1],
        best_loss=min(losses),
        epochs=config.epochs,
        checkpoint_sha256=digest,
        backbone_unchanged=(
            canonical_json(backbone.identity) == before
            and backbone.state_fingerprint() == state_before
        ),
    )
