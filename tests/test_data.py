from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from celiums_rezero.data.bytes import ByteTokenizer, load_byte_corpus
from celiums_rezero.data.corpus import ContinuousSequenceSource, evaluate_corpus
from celiums_rezero.data.prepare import prepare_public_corpus, wikitext2_paths
from celiums_rezero.data.synthetic import SyntheticSequenceSource
from celiums_rezero.transformer.config import ModelConfig
from celiums_rezero.transformer.model import ReZeroLM


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "ReZero: identidad primero."
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokens[0] == tokenizer.bos_token_id
    assert tokens[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(tokens) == text


def test_synthetic_source_repeats_for_same_seed() -> None:
    first = SyntheticSequenceSource(vocab_size=64, sequence_length=16, seed=3)
    second = SyntheticSequenceSource(vocab_size=64, sequence_length=16, seed=3)
    first_inputs, first_targets = first.batch(2, device="cpu")
    second_inputs, second_targets = second.batch(2, device="cpu")
    torch.testing.assert_close(first_inputs, second_inputs)
    torch.testing.assert_close(first_targets, second_targets)
    torch.testing.assert_close(first_inputs[:, 1:], first_targets[:, :-1])


def test_continuous_source_repeats_and_preserves_next_token_alignment() -> None:
    tokens = torch.arange(100, dtype=torch.long)
    first = ContinuousSequenceSource(tokens, sequence_length=8, seed=3)
    second = ContinuousSequenceSource(tokens, sequence_length=8, seed=3)
    first_inputs, first_targets = first.batch(4, device="cpu")
    second_inputs, second_targets = second.batch(4, device="cpu")
    torch.testing.assert_close(first_inputs, second_inputs)
    torch.testing.assert_close(first_targets, second_targets)
    torch.testing.assert_close(first_inputs[:, 1:], first_targets[:, :-1])


def test_corpus_evaluation_covers_each_transition_once() -> None:
    model = ReZeroLM(
        ModelConfig(
            vocab_size=32,
            max_sequence_length=4,
            n_layers=1,
            d_model=16,
            n_heads=2,
            d_ff=32,
        )
    )
    evaluation = evaluate_corpus(model, torch.arange(10, dtype=torch.long), batch_size=2)
    assert evaluation.tokens == 9
    assert evaluation.nll > 0
    assert evaluation.bits_per_token > 0


def test_corpus_evaluation_restores_model_mode_after_timeout() -> None:
    model = ReZeroLM(
        ModelConfig(
            vocab_size=32,
            max_sequence_length=4,
            n_layers=1,
            d_model=16,
            n_heads=2,
            d_ff=32,
        )
    ).train()
    with pytest.raises(TimeoutError):
        evaluate_corpus(
            model,
            torch.arange(10, dtype=torch.long),
            deadline=0.0,
        )
    assert model.training


def test_prepare_wikitext2_downloads_and_verifies_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = {
        "train.txt": b"train split",
        "valid.txt": b"validation split",
        "test.txt": b"test split",
    }
    checksums = {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }

    def fake_download(url: str, destination: Path, *, expected_sha256: str | None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents[destination.name])
        assert expected_sha256 == checksums[destination.name]
        return destination

    monkeypatch.setattr("celiums_rezero.data.prepare.WIKITEXT2_SPLITS", checksums)
    monkeypatch.setattr("celiums_rezero.data.prepare.download_file", fake_download)
    assert prepare_public_corpus("wikitext2", tmp_path) == list(wikitext2_paths(tmp_path))

    (tmp_path / "wikitext2" / "valid.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_public_corpus("wikitext2", tmp_path)


def test_byte_corpus_supports_ranges_and_raw_byte_protocol(tmp_path: Path) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(bytes(range(10)))
    shifted = load_byte_corpus(path, start=2, limit=4)
    raw = load_byte_corpus(path, start=2, limit=4, byte_offset=0)
    torch.testing.assert_close(shifted, torch.tensor([5, 6, 7, 8]))
    torch.testing.assert_close(raw, torch.tensor([2, 3, 4, 5]))
