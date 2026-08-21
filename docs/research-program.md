# Initial Research Program

## Question

Under which modern training conditions does identity-initialized residual control
improve optimization, stability, and compute efficiency?

## First Campaign

The first campaign compares five residual strategies on identical decoder-only
models. WikiText-2 is the rapid local benchmark and enwiki8 is the historical bridge.
FineWeb-Edu is deferred until local promotion criteria pass.

Primary questions:

1. Does canonical ReZero still beat a modern Pre-RMSNorm baseline?
2. Is any gain caused by zero residual initialization or by removing normalization?
3. Do separate attention and MLP gates improve utilization and stability?
4. Does branch-local RMSNorm preserve identity while reducing numerical failures?
5. What gate learning rate and decay policy are robust across depth?
6. Do near-zero gates identify redundant layers or optimization starvation?

## Promotion Ladder

```text
static validation -> mini-pilot -> pilot -> full local -> full cloud
```

- **Mini-pilot**: one seed, a few steps, invariant and finite-value checks.
- **Pilot**: one or two seeds, fixed token subset, early effect-size estimate.
- **Full local**: at least three seeds under equal tuning budgets.
- **Full cloud**: parameter/FLOP-matched 30M, 60M, and 150M runs after explicit
  promotion from local evidence.

## Primary Metrics

- validation negative log likelihood;
- tokens and wall time to a preregistered threshold;
- failed-seed and non-finite rate;
- gate value and gradient trajectories;
- branch-to-skip RMS ratio;
- activation and gradient RMS by depth;
- effective representation rank and small-model Jacobian spectrum;
- throughput, peak accelerator memory, FLOPs, and device-hours.

## Stop Rules

- Stop immediately on non-finite model state.
- Do not promote a candidate that only wins under a larger tuning budget.
- Treat effects below the preregistered minimum as inconclusive.
- Preserve every completed manifest and report, including negative results.

## Current Local Evidence

The controlled 12-layer WikiText-2 byte pilot (`500` steps, three seeds) found:

| Strategy | Validation NLL, mean | Standard deviation | Effect vs Pre-RMS |
|---|---:|---:|---:|
| `crz_rms` | 1.7412 | 0.0269 | +2.42% |
| `rezero_rms_shared` | 1.7439 | 0.0193 | +2.27% |
| `pre_rms` | 1.7845 | 0.0238 | baseline |

The difference between split and shared gates is only about 0.15% of the Pre-RMS
NLL, below the preregistered 1% minimum. Current evidence therefore supports
branch-local RMSNorm plus zero residual initialization, but does not establish that
separate attention and MLP gates contribute materially.

A dedicated equal-budget comparison using `rezero_rms_shared` as the preregistered
baseline confirmed a +0.153% relative effect for `crz_rms`, with an inconclusive
verdict against the 1% minimum. The split-gate hypothesis is not promoted.

The canonical raw-byte enwiki8 pilot (`200` steps, three seeds) produced means of
2.5233 for `crz_rms`, 2.5094 for `rezero_rms_shared`, and 2.5608 for `pre_rms`.
Shared gates led this short rung, so `crz_rms` must not be promoted as uniquely
superior without a longer direct ablation.
