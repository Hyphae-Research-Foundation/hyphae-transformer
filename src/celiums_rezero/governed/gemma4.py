"""Pinned local-only Gemma 4 E4B feature backbone adapter."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import ClassVar, cast

import torch

from celiums_rezero.governed.backbone import PinnedBackboneIdentity


class Gemma4E4BFrozenBackbone:
    model_id = "google/gemma-4-E4B-it"
    revision = "ee0ef6023621cff504d758262d4e04895a5af4a2"
    config_sha256 = "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
    tokenizer_config_sha256 = (
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
    )
    model_safetensors_sha256 = (
        "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
    )
    model_safetensors_bytes = 15_992_595_884
    required_artifacts: ClassVar[dict[str, tuple[int | None, str]]] = {
        "chat_template.jinja": (
            None,
            "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
        ),
        "config.json": (None, config_sha256),
        "generation_config.json": (
            None,
            "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
        ),
        "model.safetensors": (model_safetensors_bytes, model_safetensors_sha256),
        "processor_config.json": (
            None,
            "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
        ),
        "tokenizer.json": (
            32_169_626,
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        ),
        "tokenizer_config.json": (None, tokenizer_config_sha256),
    }
    hidden_size = 2560
    hidden_layer = 41

    def __init__(self, model_path: Path, *, device: str = "cuda:0") -> None:
        self.model_path = model_path.resolve(strict=True)
        self.device = torch.device(device)
        self._verify_metadata()
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise RuntimeError("Gemma backbone requires transformers==5.14.1") from error
        if getattr(transformers, "__version__", None) != "5.14.1":
            raise RuntimeError("Gemma backbone requires transformers==5.14.1 exactly")
        model_class = transformers.AutoModelForMultimodalLM
        tokenizer_class = transformers.AutoTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        self.model = model_class.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.identity = PinnedBackboneIdentity(
            family="gemma4",
            model_id=self.model_id,
            revision=self.revision,
            artifact_manifest_sha256=self._artifact_manifest_digest(),
            tokenizer_manifest_sha256=self.tokenizer_config_sha256,
            runtime_version="transformers-5.14.1",
            feature_contract="gemma4-e4b-it-layer41-final-valid-token-f32-v1",
            hidden_size=self.hidden_size,
        )
        self._state = self._state_digest()

    def encode(self, texts: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        if device != self.device:
            raise ValueError("Gemma features must be requested on the pinned model device")
        if not texts:
            raise ValueError("Gemma feature request cannot be empty")
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if int(batch["input_ids"].shape[1]) > 512:
            raise ValueError("Gemma governed prompt exceeds the 512-token preregistration")
        batch = {name: value.to(self.device) for name, value in batch.items()}
        with torch.no_grad():
            output = self.model(**batch, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[self.hidden_layer + 1]
            positions = torch.arange(hidden.shape[1], device=self.device).expand_as(
                batch["attention_mask"]
            )
            index = positions.masked_fill(batch["attention_mask"] == 0, -1).max(-1).values
            rows = torch.arange(hidden.shape[0], device=self.device)
            features = hidden[rows, index].float().detach().clone()
        if self._state_digest() != self._state:
            raise RuntimeError("Gemma backbone state changed during feature extraction")
        return cast(torch.Tensor, features)

    def state_fingerprint(self) -> str:
        return self._state_digest()

    def _verify_metadata(self) -> None:
        for name, (expected_bytes, expected_digest) in self.required_artifacts.items():
            path = self.model_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Gemma artifact is absent or unsafe: {name}")
            if expected_bytes is not None and path.stat().st_size != expected_bytes:
                raise ValueError(f"Gemma artifact size mismatch: {name}")
            if _sha256(path) != expected_digest:
                raise ValueError(f"Gemma artifact digest mismatch: {name}")

    def _artifact_manifest_digest(self) -> str:
        payload = json.dumps(self.required_artifacts, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _state_digest(self) -> str:
        digest = hashlib.sha256()
        for index, (name, parameter) in enumerate(self.model.named_parameters()):
            digest.update(
                f"parameter:{name}:{parameter.data_ptr()}:{parameter._version}:"
                f"{parameter.numel()}:{parameter.dtype}:{parameter.requires_grad}".encode()
            )
            if index % 97 == 0 or index < 4:
                tensor = parameter.detach().reshape(-1)
                sample = tensor[: min(256, tensor.numel())].float().cpu().numpy().tobytes()
                digest.update(sample)
        for name, buffer in self.model.named_buffers():
            digest.update(
                f"buffer:{name}:{buffer.data_ptr()}:{buffer._version}:"
                f"{buffer.numel()}:{buffer.dtype}".encode()
            )
        return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
