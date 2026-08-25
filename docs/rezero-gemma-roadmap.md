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

The third attempt passed. On one MI355X, frozen Gemma features drove a 2,919,188
parameter shared-gate ReZero controller through finite forward and backward passes.
Both residual-gate gradients were finite, gates were excluded from weight decay, the
Gemma fingerprint remained unchanged, and peak allocated VRAM was 18,002,620,928 bytes
against the 240 GiB limit. The Droplet was deleted and confirmed absent. The
content-addressed result is
[`experiments/results/gemma4_e4b_rezero_smoke_v1.json`](../experiments/results/gemma4_e4b_rezero_smoke_v1.json).

This result satisfies the smoke promotion gate for the preregistered frozen-feature
training search. It does not report validation, test, or adversarial model quality.

## Frozen-Gemma Training Result

The preregistered search completed all nine train/validation runs for learning rates
`0.001`, `0.003`, and `0.01` across seeds 17, 29, and 43. Every validation run passed
with zero unsafe answers. The ranking selected `0.001` because it achieved evidence
exact match `0.99999994` on validation, ahead of `0.99090904` for both larger learning
rates; final training loss was therefore not used to override the earlier ranking
criterion.

The selected recipe was evaluated once on test and adversarial splits. Every seed
passed all gates with action accuracy 1.0, abstention recall 1.0, answer recall within
float32 precision of 1.0, and zero unsafe answers. Test evidence exact match ranged
from `0.99545449` to `0.99999994`; adversarial exact match ranged from `0.99090904` to
`0.99999994`. Gemma remained unchanged throughout.

The content-addressed result is recorded in
[`experiments/results/gemma4_e4b_rezero_sequence_control_v1.json`](../experiments/results/gemma4_e4b_rezero_sequence_control_v1.json).
This promotes the ReZero controller to deterministic packaging and external shadow
evaluation. It still does not authorize Gemma fine-tuning or multi-step navigation.

## ReZero External Shadow Result

The selected seed-17 controller was packaged deterministically as bundle
`rzcb_25c0219bd57b00231f046cc75667314175a58c60ade89b4c5fc103bcd92469c6`
and evaluated on the unchanged corrected non-MARS cases. The shadow completed but
failed its preregistered gates. Action match was `0.75` and pointer exact match was
`0.8333`; there were zero unsafe upgrades, but two supported scenarios were downgraded
to abstention and counted as operational divergences. The previously known low-score
case also produced one allowed conservative abstention. Latency passed comfortably.

The negative result is preserved in
[`experiments/results/gemma4_e4b_rezero_shadow_external_v1.json`](../experiments/results/gemma4_e4b_rezero_shadow_external_v1.json).
The candidate is not promoted to a host-affecting runtime or navigation. The next
experiment must address external supported-case generalization using train and
validation data only; relaxing the shadow gates after observation is not permitted.

The v2 hypothesis addresses the missing information directly. The model will receive a
host-owned 17-value sufficiency certificate containing the policy thresholds, top and
second retrieval scores, score margin, count above threshold, approximation state, and
the one-hot result of `SufficiencyPolicy.decide`. A fixed action prior maps `supported`
to `answer`, partial/absent to `request_evidence`, and conflict/blocked to `abstain`;
the learned ReZero path remains a residual correction. These values are computed from
the ordinary request and root-owned policy, not from shadow labels. The prospective
protocol is
[`experiments/canonical/gemma4_e4b_rezero_sequence_control_v2.json`](../experiments/canonical/gemma4_e4b_rezero_sequence_control_v2.json).

The v2 search selected learning rate `0.0003`; every validation, test, and adversarial
run passed with zero unsafe answers and an unchanged Gemma fingerprint. This authorizes
only a repeat of the unchanged external shadow using the new content-addressed bundle.

That shadow remained negative. The action certificate corrected the absent low-score
case, but the learned pointer residual still overrode policy support: two supported
cases selected no pointer and were forced to abstain, while one distractor was selected
in another answer. The result is preserved in
[`experiments/results/gemma4_e4b_rezero_shadow_external_v2.json`](../experiments/results/gemma4_e4b_rezero_shadow_external_v2.json).
The next protocol will certify support per evidence item and bound the semantic pointer
residual so it cannot reverse that host validation.

The v3 protocol bounds learned action and pointer residuals to `[-1, 1]` while host
certificates contribute magnitude 20. Therefore learned semantics can rank within a
certified class but cannot reverse action or evidence-support signs. The prospective
protocol is
[`experiments/canonical/gemma4_e4b_rezero_sequence_control_v3.json`](../experiments/canonical/gemma4_e4b_rezero_sequence_control_v3.json).
