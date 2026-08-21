# Celiums ReZero

Celiums ReZero is a from-scratch PyTorch framework for testing identity-initialized
residual learning in modern decoder-only transformers. It contains two coupled
products:

- **Core**: reusable ReZero gates, a language model, training utilities, and signal
  propagation diagnostics.
- **Lab**: typed hypotheses, staged experiments, budgets, evidence-backed memory,
  immutable run manifests, and reproducible reports.

The project is inspired by *ReZero is All You Need: Fast Convergence at Large Depth*
but is an independent implementation. It is not affiliated with or endorsed by the
paper's authors, UC San Diego, Ai2, Asta, or Practical NLP.

## Residual Strategies

All strategies use the same attention, MLP, tokenizer, data, and training pipeline.

| Strategy | Branch normalization | Gates |
|---|---|---|
| `pre_rms` | RMSNorm | fixed residual weight 1 |
| `rezero_canonical` | none | one shared scalar per block |
| `rezero_split` | none | separate attention and MLP scalars |
| `rezero_rms_shared` | RMSNorm | one shared scalar per block |
| `crz_rms` | RMSNorm | separate attention and MLP scalars |

The primary Celiums hypothesis is `crz_rms`:

```text
u       = x + alpha_attn * Attention(RMSNorm(x))
x_next  = u + alpha_mlp  * SwiGLU(RMSNorm(u))
```

Both gates start at zero. The branch output projections are deliberately non-zero so
the gates receive a useful first-step gradient.

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run celiums-rezero smoke-model --strategy crz_rms
uv run celiums-rezero smoke-lab --root runs/smoke
uv run celiums-rezero prepare-data wikitext2
uv run celiums-rezero pilot-wikitext2 --device auto --seeds 7 17 29
uv run celiums-rezero prepare-data enwiki8
uv run celiums-rezero pilot-enwiki8 --device auto --seeds 7 17 29
```

The WikiText-2 command downloads checksum-pinned splits from a fixed
`pytorch/examples` revision. The pilot uses lossless byte tokens and records validation
and test negative log likelihood plus bits per byte for every residual strategy. Runs
are idempotent by manifest ID, and each campaign writes aggregate JSON and HTML with
per-strategy means, standard deviations, failed-seed rates, and effects versus
`pre_rms`. The default campaign stage is `pilot`; use `--stage mini_pilot` for
correctness-only runs.

Training runs write an atomic `latest.pt` checkpoint and append-only `history.jsonl`
with losses, gradient norms, and gate value/gradient trajectories. Re-executing an
incomplete manifest resumes model, optimizer, data-source, and random-number state.
Completed manifests are idempotent.

`pilot-enwiki8` uses the canonical 90M/5M/5M ranges and a 256-symbol raw-byte
protocol. `run-manifest` executes a strict, versioned manifest through an allowlisted
runner, which is also the entry point used by the included container image.

Autoregressive generation uses per-layer grouped-query KV caches until a sliding
context window must be rebuilt; `use_cache=False` remains available as a reference
path.

The current local conclusion is narrower than the project name: RMS-normalized ReZero
outperformed Pre-RMS in the tested short pilots, while separate attention/MLP gates
did not clear the preregistered minimum versus one shared gate. See
[`docs/research-program.md`](docs/research-program.md) for the evidence and stop
decision.

## Research Contract

- Claims are registered before full experiments.
- Mini-pilots verify correctness before expensive runs.
- Baselines receive equal tuning and token budgets.
- Gates are excluded from weight decay by default and have an explicit learning-rate
  group.
- Results report seeds, tokens, parameters, FLOPs, memory, wall time, and failures.
- Negative and inconclusive results are first-class evidence.
- No cloud run is promoted without passing local invariants and a declared budget.

See [`docs/research-program.md`](docs/research-program.md) for the initial research
program and [`docs/attribution.md`](docs/attribution.md) for provenance.

## License

MIT. See [`LICENSE`](LICENSE).
