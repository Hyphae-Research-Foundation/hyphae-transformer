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

## V0.2 Analysis Protocol

The next campaign preregisters `rezero_rms_shared` against `pre_rms`. Corpus
manifests are location-independent and bind a prepared data root only at execution.
Validation NLL is sampled on a fixed step grid, tokens-to-threshold uses the first
observed crossing without interpolation, and comparisons are paired by seed. Point
effects are promoted only when their paired 95% Student-t confidence interval clears
the minimum effect.

### WikiText-2 1M-Token Result

The preregistered 12-layer campaign completed 1,000 steps and 1,024,000 training
tokens for seeds 7, 17, and 29 with no failures.

| Strategy | Validation NLL | 95% CI | Tokens/s |
|---|---:|---:|---:|
| `rezero_rms_shared` | 1.5435 | [1.5257, 1.5613] | 7,489 |
| `pre_rms` | 1.5738 | [1.5505, 1.5970] | 8,016 |

The paired relative NLL effect was +1.92%, but its 95% confidence interval was
[-0.66%, +4.51%]. It did not clear the preregistered +1% minimum with confidence, so
the verdict is **inconclusive**. Shared-gate ReZero used about 6.6% more peak memory
and had about 6.6% lower throughput in this configuration.

At the 1.8 NLL threshold, both conditions reached the target for all seeds. Observed
training-token crossings were `(512k, 614.4k, 512k)` for `pre_rms` and
`(512k, 409.6k, 512k)` for `rezero_rms_shared`. The paired mean improvement was
12.5%, but the 95% confidence interval was [-41.3%, +66.3%], also inconclusive.

This rung does not justify cloud promotion. The next efficient action is increasing
local seed count or repeating at a deeper model before paying for larger hardware.

### Preregistered Eight-Seed Extension

Because the three-seed point effect was above the 1% minimum but its interval was
wide, the next rung fixes eight total seeds before additional computation. It reuses
the immutable runs for seeds 7, 17, and 29 and adds seeds 41, 53, 67, 79, and 97 with
the identical 12-layer, 1,024,000-token configuration. All five new seeds must run
unless the declared failure budget is exceeded. The primary verdict remains based on
the paired final validation-NLL effect and its 95% Student-t interval; the 1.8-NLL
crossing is secondary.

### Eight-Seed Extension Result

All five additional seeds completed, producing eight paired observations and no
failures. Final validation NLL was 1.5490 for `rezero_rms_shared` and 1.5672 for
`pre_rms`. The paired relative improvement was +1.16%, with a 95% confidence
interval of [+0.17%, +2.15%]. The interval excludes zero but its lower bound remains
below the preregistered +1% minimum, so the formal verdict is **inconclusive**.

For the secondary 1.8-NLL threshold, the paired mean token improvement was +4.88%
with a 95% interval of [-6.66%, +16.41%], also inconclusive. Shared-gate ReZero
retained its quality signal but remained slower and more memory-intensive.

The extension falsifies the claim at the declared practical-effect threshold for this
12-layer, one-million-token condition. Additional seeds at the same condition are not
authorized. Future work should test whether depth changes the effect rather than
continuing to narrow this interval.

### Preregistered 24-Layer Rung

The depth axis is tested next with 24 layers while preserving the 12-layer campaign's
width, batch size, sequence length, optimizer settings, one-million-token budget,
validation grid, threshold, and seeds 7, 17, and 29. The primary verdict remains the
paired final validation-NLL effect against the +1% practical minimum. All six runs
must complete unless the declared failure budget is exceeded; interim curves do not
authorize optional stopping.

### 24-Layer Result

The exact preregistered campaign completed all six runs with no failures. Final
validation NLL was 1.5273 for `rezero_rms_shared` and 1.5504 for `pre_rms`. The
paired relative improvement was +1.49%, with a 95% confidence interval of
[+0.14%, +2.84%]. The direction is supported, but the lower bound remains below the
preregistered +1% practical minimum, so the primary verdict is **inconclusive**.

The secondary threshold result was decisive on the fixed 100-step observation grid:
all `pre_rms` seeds first crossed 1.8 NLL at 512,000 tokens, while all
`rezero_rms_shared` seeds crossed at 409,600 tokens. This is a paired 20% reduction,
with a collapsed 95% interval of [20%, 20%]. Shared-gate ReZero also used about 7.2%
more peak memory; throughput was similar in the final valid run.

Two earlier attempts are explicitly invalid evidence. The first exceeded the default
100 MB artifact budget, which was below the preregistered 500 MB allowance. The
second reused corrected budget values but exposed CUDA checkpoint RNG tensors being
restored into CPU generators after tool timeout. Both attempts remain immutable in
the local registry, and the defects were fixed through reviewed PRs before executing
the fresh preregistered-budget campaign.

The depth result strengthens the optimization-speed signal but still does not satisfy
the primary final-NLL criterion. The next depth rung should be preregistered at 48
layers; cloud promotion remains deferred until local memory/runtime feasibility is
measured.

### Preregistered 48-Layer Rung

The 48-layer rung preserves the 24-layer campaign's width, optimizer, batch size,
sequence length, one-million-token budget, validation grid, threshold, and seeds. A
local CUDA feasibility gate runs both strategies for 10 steps at the exact full-run
shape before any full campaign starts. The gate requires finite state and peak memory
below 7.5 GB for both conditions. If it passes, all six full runs must complete unless
the declared failure budget is exceeded; interim effects do not authorize stopping.

### 48-Layer Result

The feasibility gate passed for both strategies at the exact full-run shape. Peak
memory was 0.98 GB for `pre_rms` and 1.05 GB for `rezero_rms_shared`, well below the
7.5 GB stop limit, and both 10-step runs remained finite.

All six full runs then completed without failure. Final validation NLL was 1.5207 for
`rezero_rms_shared` and 1.5381 for `pre_rms`. The paired relative improvement was
+1.13%, with a 95% confidence interval of [+0.59%, +1.68%]. The interval excludes
zero but its lower bound remains below the preregistered +1% minimum, so the primary
verdict is **inconclusive**.

The secondary threshold signal replicated: every `pre_rms` seed first crossed 1.8
NLL at 512,000 tokens, while every shared-gate seed crossed at 409,600 tokens. The
paired reduction is 20%, with a collapsed interval of [20%, 20%]. Shared-gate ReZero
used about 7.7% more peak memory and delivered about 9.7% lower throughput.

Across 24 and 48 layers, shared-gate ReZero consistently reaches the 1.8-NLL target
20% earlier, while final-NLL practical superiority remains just below the declared
confidence threshold. Local feasibility is established; a cloud scale-up is now
scientifically defensible only if it targets a larger parameter or sequence-length
rung rather than repeating the same depth condition.

### Preregistered 30M Cloud Rung

The first cloud rung scales width to 512 with 12 layers, producing approximately
31.6M parameters while preserving 1,024,000 training tokens per condition-seed run.
It uses sequence length 256, batch size 8, validation every 50 steps, and seeds 7, 17,
and 29. Before provisioning, both strategies must pass a local exact-shape 10-step
CUDA gate below 7.5 GB peak memory.

If promoted, all six runs execute sequentially on one ephemeral RTX 4000 Ada 20 GB
Droplet in `tor1`. The hard lifetime is three hours and the maximum infrastructure
cost is $2.28 at the recorded $0.76/hour rate. The Droplet must be destroyed after
artifact retrieval on both success and failure paths. Interim effects do not authorize
optional stopping.

### 30M Cloud Result

The local exact-shape feasibility gate passed. Peak memory was 1.39 GB for
`pre_rms` and 1.49 GB for `rezero_rms_shared`, both well below the 7.5 GB stop
limit. The promoted campaign then completed all six runs on one ephemeral
DigitalOcean RTX 4000 Ada 20 GB Droplet in `tor1` with no failed seeds.

Final validation NLL was 1.4979 for `rezero_rms_shared` and 1.5324 for `pre_rms`.
The paired relative improvement was +2.25%, with a 95% confidence interval of
[-0.79%, +5.30%]. The interval crosses zero and the +1% practical minimum, so the
primary verdict is **inconclusive**.

For the 1.8-NLL threshold, `pre_rms` crossed at 512,000 tokens for all seeds, while
shared-gate ReZero crossed at `(512k, 409.6k, 409.6k)`. The paired mean reduction was
13.33%, with a wide 95% interval of [-15.35%, +42.02%], also inconclusive.

The Droplet existed for approximately 0.786 hours at the recorded $0.76/hour rate,
for an estimated infrastructure cost of $0.60 against the $2.28 maximum. Evidence
was retrieved before deletion, and the Droplet was confirmed absent afterward. No
Celiums ReZero cloud resource remains active.

This first parameter-scale rung shows a larger point estimate but insufficient
precision. It does not authorize a larger 60M cloud rung. The next efficient action is
to improve artifact upload/runner automation and decide whether additional 30M seeds
are worth preregistering under a new budget.

### Preregistered 30M Eight-Seed Extension

The initial cloud point estimate is extended to eight total seeds by adding seeds 41,
53, 67, 79, and 97 under the identical 30M configuration. Ten new runs execute on a
single RTX 4000 Ada Droplet through the fail-safe executor. The hard lifetime is two
hours and the extension cost ceiling is $1.52 at $0.76/hour. All new runs must execute
unless the declared failure or cloud budget is exceeded; interim effects do not
authorize stopping. The combined analysis reuses the original three seeds and remains
paired by seed.

The first extension launch attempt created no resource and spent $0 because
DigitalOcean returned `Size is not available in this region` for the preregistered
RTX 4000 Ada in `tor1`. No result data was produced. Any alternate accelerator or
cost ceiling requires a prospective amendment before another launch.
