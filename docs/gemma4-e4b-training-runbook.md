# Gemma 4 E4B Governed-Control Training Runbook

The selected parent is `google/gemma-4-E4B-it` at immutable revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`. The model is Apache-2.0 and public, but
the local checkpoint is not present. This repository trains only the governed control
head; Gemma parameters and buffers remain frozen.

## Hardware Decision

Create a dedicated single AMD MI355X spot Droplet; do not reuse `hyin-mi355x`, which is
serving unrelated work:

- size `gpu-mi355x1-288gb-spot`;
- 288 GiB VRAM;
- 24 vCPU, 256 GiB RAM;
- $4.50/hour, with an authorized hard budget of $36 / 8 hours;
- 720 GiB boot plus 5 TiB scratch.

Do not provision the eight-GPU node before measuring the x1 smoke. The x8 MI355X spot
node costs $36/hour and is authorized only when the single-GPU batch smoke is finite but
cannot satisfy the effective-batch requirement, or when measured distributed throughput
predicts lower total dollar cost. The current head-only design is expected to fit x1;
x8 is a scaling option, not a prerequisite.

## Stop Gates

Stop before paid training if any artifact digest, model topology, tokenizer/template,
MARS source digest, split-leakage gate, ROCm/PyTorch/Transformers pin, backbone freeze,
or fixture evaluation fails. Stop the x1 smoke on non-finite state or peak allocation
above 240 GiB. Destroy spot resources after evidence retrieval; powered-off GPU Droplets
remain billable.

After the smoke passes, run the three preregistered seeds for exactly three epochs with
AdamW, learning rate `0.001`, evidence-loss weight `1.0`, and gradient clipping `1.0`.
Evaluate the held-out test and adversarial splits only after training. Do not use those
metrics for checkpoint selection. Any seed that misses a preregistered gate falsifies the
claim; do not average a failed seed away.

## Current External Blocks

- Gemma E4B weights/tokenizer are not cached locally or on a dedicated campaign host.
- `transformers==5.14.1` and the dedicated ROCm runtime remain to be verified on that host.
- The occupied host has suitable scratch, but is explicitly excluded from this campaign.
- MARS v2 canonical dev/test and compiled evaluation dev/test data are present locally
  with pinned digests and are converted without modifying their source.

The governed converter partitions complete 19-record MARS worlds. It uses 20 dev worlds
for train, 10 separate dev worlds for validation, all 20 test worlds for test, and the
final 10 dev worlds for adversarial evaluation. Source episode/world identities remain
in the deterministic derived IDs; no world may cross splits.

The dedicated executor verifies one MI355X, ROCm availability, exact model artifacts,
`transformers==5.14.1`, and the governed dataset before running batch sizes 1, 2, 4, and
8. Its paid-lifetime clock starts as soon as DigitalOcean returns the new Droplet and
also covers bootstrap and evidence retrieval, with five minutes reserved for deletion.
PyTorch addresses the ROCm device as `cuda:0`; `rocm` identifies the runtime, not a
valid `torch.device` string. The plan remains deliberately invalid until `revision` is
replaced with the 40-character merged commit that contains this preregistration and
executor.
