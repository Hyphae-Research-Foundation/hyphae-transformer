# Governed Model Training V1

## Claim

This slice trains only a small action and evidence-pointer control head over a frozen
feature backbone. It does not train acquired knowledge into model weights, grant model
authority, or establish production Gemma quality.

Actions are `answer`, `request_evidence`, and `abstain`. The host remains authoritative
for tenant routing, source policy, sufficiency, conflict, generation, publication and
notification. Pointer logits can select only evidence supplied in the current request;
opaque handles are never generated as vocabulary tokens.

## Preregistered Fixture Protocol

- Frozen fixture backbone: `utf8-signed-ngram-l2-f32-v1`, 128 dimensions.
- Trainable parameters: one three-way action classifier and low-rank evidence pointer.
- Seed: 17.
- Optimizer: AdamW.
- Full-batch deterministic CPU training.
- No early stopping based on test or adversarial data.
- Primary fixture gates: action accuracy >= 0.90, answer recall >= 0.80,
  abstention recall = 1.00, answer evidence exact-set match >= 0.90, unsupported answer
  rate = 0.
- Any backbone mutation, target-policy disagreement, unknown handle, digest mismatch,
  split leakage or non-reproducible checkpoint invalidates the run.

## Gemma Boundary

A real Gemma feature extractor must implement `FrozenTextBackbone` with a pinned local
model revision, full model/tokenizer artifact manifests, fixed hidden layer and pooling
contract, eval mode, no gradients and finite detached float32 output. Gemma weights,
tokenizer, license acceptance and runtime are not bundled here. Until independently
annotated Gemma trajectories and hardware-specific evaluation pass, this fixture proves
only the training/checkpoint/governance contract and should run in shadow mode.
