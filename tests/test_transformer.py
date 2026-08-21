from __future__ import annotations

import pytest
import torch

from celiums_rezero.core.diagnostics import collect_gate_stats
from celiums_rezero.transformer.block import TransformerBlock, residual_gate_count
from celiums_rezero.transformer.config import ModelConfig, ResidualStrategy
from celiums_rezero.transformer.model import ReZeroLM


def tiny_config(strategy: ResidualStrategy) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_sequence_length=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        residual_strategy=strategy,
    )


@pytest.mark.parametrize("strategy", list(ResidualStrategy))
def test_all_strategies_forward_backward(strategy: ResidualStrategy) -> None:
    config = tiny_config(strategy)
    model = ReZeroLM(config)
    tokens = torch.randint(3, config.vocab_size, (2, 16))
    output = model(tokens, tokens.roll(-1, dims=1))
    assert output.logits.shape == (2, 16, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (ResidualStrategy.PRE_RMS, 0),
        (ResidualStrategy.REZERO_CANONICAL, 1),
        (ResidualStrategy.REZERO_SPLIT, 2),
        (ResidualStrategy.REZERO_RMS_SHARED, 1),
        (ResidualStrategy.CRZ_RMS, 2),
    ],
)
def test_strategy_gate_topology(strategy: ResidualStrategy, expected: int) -> None:
    block = TransformerBlock(tiny_config(strategy))
    assert residual_gate_count(block) == expected
    assert len(collect_gate_stats(block)) == expected


@pytest.mark.parametrize(
    "strategy",
    [
        ResidualStrategy.REZERO_CANONICAL,
        ResidualStrategy.REZERO_SPLIT,
        ResidualStrategy.REZERO_RMS_SHARED,
        ResidualStrategy.CRZ_RMS,
    ],
)
def test_gated_transformer_block_is_exact_identity(strategy: ResidualStrategy) -> None:
    block = TransformerBlock(tiny_config(strategy)).eval()
    inputs = torch.randn(2, 8, 32)
    output = block(inputs)
    torch.testing.assert_close(output, inputs, rtol=0, atol=0)


def test_first_lm_backward_has_gate_but_not_branch_gradients() -> None:
    model = ReZeroLM(tiny_config(ResidualStrategy.CRZ_RMS))
    tokens = torch.randint(3, model.config.vocab_size, (2, 16))
    output = model(tokens, tokens.roll(-1, dims=1))
    assert output.loss is not None
    output.loss.backward()
    gate_stats = collect_gate_stats(model)
    assert gate_stats and all(stat.gradient is not None for stat in gate_stats)
    first = model.blocks[0]
    assert first.attention.out_proj.weight.grad is not None
    torch.testing.assert_close(
        first.attention.out_proj.weight.grad,
        torch.zeros_like(first.attention.out_proj.weight.grad),
        rtol=0,
        atol=0,
    )


def test_generation_is_deterministic_at_zero_temperature() -> None:
    model = ReZeroLM(tiny_config(ResidualStrategy.CRZ_RMS)).eval()
    prompt = torch.tensor([[3, 4, 5]])
    first = model.generate(prompt, max_new_tokens=4, temperature=0)
    second = model.generate(prompt, max_new_tokens=4, temperature=0)
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_incremental_cache_matches_full_forward(n_kv_heads: int) -> None:
    config = tiny_config(ResidualStrategy.PRE_RMS)
    config = ModelConfig(**{**config.to_dict(), "n_kv_heads": n_kv_heads})
    model = ReZeroLM(config).eval()
    tokens = torch.randint(3, config.vocab_size, (2, 12))
    full_logits = model(tokens).logits

    cached_logits = []
    cache = None
    for position in range(tokens.shape[1]):
        output = model(
            tokens[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cached_logits.append(output.logits)
        cache = output.past_key_values
    torch.testing.assert_close(torch.cat(cached_logits, dim=1), full_logits, rtol=1e-4, atol=1e-5)
    assert cache is not None
    for keys, values in cache:
        assert keys.shape == values.shape == (2, n_kv_heads, 12, config.head_dim)


def test_chunked_cache_matches_full_forward() -> None:
    model = ReZeroLM(tiny_config(ResidualStrategy.PRE_RMS)).eval()
    tokens = torch.randint(3, model.config.vocab_size, (2, 13))
    prefix = model(tokens[:, :7], use_cache=True)
    continuation = model(
        tokens[:, 7:],
        past_key_values=prefix.past_key_values,
        use_cache=True,
    )
    torch.testing.assert_close(
        continuation.logits,
        model(tokens).logits[:, 7:],
        rtol=1e-4,
        atol=1e-5,
    )


def test_cached_generation_matches_uncached_across_context_limit() -> None:
    model = ReZeroLM(tiny_config(ResidualStrategy.PRE_RMS)).eval()
    prompt = torch.randint(3, model.config.vocab_size, (2, 13))
    cached = model.generate(prompt, max_new_tokens=7, temperature=0, use_cache=True)
    uncached = model.generate(prompt, max_new_tokens=7, temperature=0, use_cache=False)
    torch.testing.assert_close(cached, uncached)


def test_invalid_cache_usage_fails_early() -> None:
    model = ReZeroLM(tiny_config(ResidualStrategy.PRE_RMS)).eval()
    tokens = torch.randint(3, model.config.vocab_size, (1, 4))
    output = model(tokens, use_cache=True)
    assert output.past_key_values is not None
    with pytest.raises(ValueError, match="requires use_cache"):
        model(tokens[:, :1], past_key_values=output.past_key_values)
    with pytest.raises(ValueError, match="targets cannot"):
        model(tokens, tokens, use_cache=True)


def test_invalid_config_fails_early() -> None:
    with pytest.raises(ValueError, match="n_heads must divide"):
        ModelConfig(d_model=30, n_heads=8)
