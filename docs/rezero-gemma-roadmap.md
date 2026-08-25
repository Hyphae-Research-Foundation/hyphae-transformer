# ReZero-Gemma Integration Roadmap

## Recovered Baseline

The repository has two validated but previously separate lines of work:

- `rezero_rms_shared` improved final validation NLL by 2.33% with a paired 95%
  interval of `[1.59%, 3.06%]` in the 30M eight-seed campaign and reduced observed
  tokens to the 1.8-NLL threshold by 17.5%.
- The frozen-Gemma governed control v3 bundle passed held-out and adversarial gates,
  and its corrected external shadow v2 campaign passed with no unsafe upgrades or
  operational divergences.

The current production-shaped path uses a linear action and pointer head. Gemma,
MiniLM, and Hyphae are pinned dependencies; none is trained by that path.

## First Integration Slice

`ReZeroSequenceControlHead` is the first direct bridge between those lines. It accepts
the same frozen Gemma context and evidence features as control v3, projects them into a
bounded sequence, and processes that sequence with the promoted
`rezero_rms_shared` transformer blocks. It preserves the existing three-action and
evidence-pointer boundary, host-owned sufficiency features, and fixed score prior.

This slice intentionally does not yet implement autonomous navigation. Its purpose is
to test one necessary question before adding a tool loop:

> Do the validated shared-gate ReZero blocks add useful evidence-interaction capacity
> without weakening the safety properties already established by control v3?

The prospective comparison is fixed in
[`experiments/canonical/gemma4_e4b_rezero_sequence_control_v1.json`](../experiments/canonical/gemma4_e4b_rezero_sequence_control_v1.json).

## Promotion Order

1. Run the preregistered frozen-feature comparison against control v3.
2. Require every seed to pass held-out and adversarial safety gates.
3. Package the candidate with the same content-addressed deployment contract.
4. Run corrected external shadow cases without authority to affect host decisions.
5. Only after those gates pass, define a separate preregistration for a bounded
   multi-step Hyphae navigation environment.
6. Evaluate LoRA on Gemma only if the external ReZero controller cannot meet the
   preregistered navigation objective. Full Gemma fine-tuning is not authorized by
   this roadmap.

At every stage, tenant routing, credentials, source policy, publication, generation
activation, and notification remain host-owned.

## First MI355X Smoke Attempt

The first paid smoke produced no model observation. The pinned ROCm runtime rejected
`torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))` before Gemma loading or
ReZero evaluation. The Droplet was deleted and confirmed absent. The failed attempt and
its content-addressed logs are recorded in
[`experiments/results/gemma4_e4b_rezero_smoke_attempt_v1.json`](../experiments/results/gemma4_e4b_rezero_smoke_attempt_v1.json).
The API call is corrected prospectively to use the already active device, matching the
previously successful Gemma smoke convention.

The second paid attempt also produced no model observation. The plan contained a
mistyped and nonexistent merge SHA, so checkout failed before environment setup. Its evidence is recorded in
[`experiments/results/gemma4_e4b_rezero_smoke_attempt_v2.json`](../experiments/results/gemma4_e4b_rezero_smoke_attempt_v2.json).
The executor now verifies locally that the exact revision can be fetched before creating
a paid resource, and the remote bootstrap fetches that revision explicitly before
checkout.
