# ADR 0007: adapt collated replay batches to RLlib SAC

- Status: Accepted
- Date: 2026-07-24

## Context

Phase 3 ends with bounded, contiguous NumPy batches built from uniformly
sampled transitions. The runtime now needs to update SAC without inheriting
RLlib's synchronous `Algorithm` control plane or copying SAC loss code.

RLlib 2.56.1 accepts a prebuilt `MultiAgentBatch`, but its stock SAC learner
checkpoint omits two pieces of state needed for deterministic continuation:

- the current trainable `log_alpha` value;
- the per-module sampled timestep of the last target-network update.

The second omission matters whenever `target_network_update_freq` is nonzero.
Restoring only target weights is insufficient because the next update could run
the target schedule at a different time.

## Decision

`SACLearnerAdapter` owns one local RLlib `LearnerGroup`. It requires PyTorch,
the RLModule/Learner and ConnectorV2 API stacks, and `num_learners=0`. GPU
placement remains the responsibility of the future one-GPU `LearnerHost` Ray
actor. Phase 4 intentionally supports exactly one `default_policy` module;
multi-module collation remains part of the hierarchy phase.

The flat SAC input schema is:

- `obs`;
- `new_obs`;
- `actions`;
- `rewards`;
- `terminateds`;
- `truncateds`;
- `n_step`, defaulting to one;
- `weights`, defaulting to one.

The adapter validates aligned leading dimensions and scalar reward/termination
columns, normalizes RLlib-specific dtypes, and wraps the columns in one
`SampleBatch` under `default_policy`, then in a `MultiAgentBatch`. RLlib remains
responsible for tensor conversion, device transfer, forward passes, SAC losses,
optimizer steps, and target-network updates.

Update calls receive absolute sampled environment and agent-step counters. The
learning-start gate follows RLlib's `count_steps_by` setting, while both lifetime
counters are passed to the learner. In particular, RLlib continues to evaluate
its target-network schedule from sampled environment steps.

An initial inference publication has module version zero. After that, weights
are published every configured number of successful learner updates. A
`WeightsDescriptor` carries the member ID, per-module versions, learner update
count, publication timestamp, and inference-only module state.

The versioned adapter checkpoint contains:

- full RLlib module, target-network, optimizer, and sequence state;
- current SAC `log_alpha`;
- the target-update timestep per module;
- sampled-step and learner-update counters;
- publication interval state and module versions;
- the last actually published inference state.
- PyTorch CPU and visible-CUDA RNG states;
- an algorithm-configuration and observation/action-space compatibility
  contract.

RLlib's diagnostic metrics logger is deliberately excluded from the checkpoint:
some metric windows retain live autograd tensors and are not a safe persistence
boundary. Runtime lifetime counters needed for scheduling are stored explicitly;
loss windows restart after restore.

`CheckpointableSACTorchLearner` subclasses RLlib's learner only to add and
restore the two missing state entries. It does not override SAC loss, gradient,
optimizer, or target-update logic.

## Consequences

- Fixed-batch updates can be compared directly with stock RLlib SAC.
- Learning start and target updates use the same sampled-step semantics as
  RLlib's new API stack.
- Restoring the adapter preserves the next optimizer, temperature, target, and
  publication decisions, including the next stochastic SAC sample.
- The last publication is checkpointed separately from current learner weights;
  this matters when a checkpoint falls between publication intervals.
- Weight state is immutable by contract and copied at publication/checkpoint
  boundaries.
- `published_at_monotonic` is diagnostic within one process lifetime and must
  not be used to order publications across restore.
- RNG restoration is process-global, matching the one-learner-per-`LearnerHost`
  actor topology.
- Explicit pinned-memory and CUDA prefetch behavior remains part of the future
  `LearnerHost` integration, not this algorithm adapter.

## Rejected alternatives

### Call the SAC learner connector with synthetic episodes

Rejected for the hot path because Phase 3 has already materialized independent
transitions. Recreating one Python episode object per sampled transition adds
allocation without changing the learner input.

### Reimplement SAC loss or target updates

Rejected because it duplicates algorithm logic that RLlib already owns and
would weaken the fixed-version parity gate.

### Restore target weights but not the last target-update timestep

Rejected because the next update can then execute a different schedule from an
uninterrupted control run.

### Regenerate the last publication from current learner weights on restore

Rejected because learner weights may have advanced since the last publication.
Reusing the old version for newer weights would break behavior-version
provenance.
