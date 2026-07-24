# ADR 0001: compose RLlib components inside a Tune Trainable

- Status: Accepted for Phase 0
- Date: 2026-07-24

## Context

The runtime needs explicit ownership of rollout actors, one authoritative replay
service, learner-local replay views, pending calls, weight versions, and
checkpoint boundaries.

RLlib 2.56.1 offers reusable `RLModule`, `LearnerGroup`, learner connectors, and
`SingleAgentEnvRunner` APIs. A standard SAC `Algorithm`, however, eagerly creates
its own `EnvRunnerGroup`, replay buffer, `LearnerGroup`, synchronization path,
and synchronous training step.

Subclassing `Algorithm` would therefore either retain an unused control plane or
require overriding a large portion of its setup and training lifecycle. Both
options obscure resource ownership and make the future population topology
harder to reason about.

## Decision

Use a thin `ray.tune.Trainable` as the future member controller and compose
RLlib components explicitly.

The controller will own lifecycle and reporting. RLlib continues to own:

- SAC modules and losses;
- learner connectors;
- optimizers and target-network updates;
- episode and environment-runner abstractions.

The project will own:

- replay ingest, retention, snapshot, and delta;
- learner-local materialized replay;
- the asynchronous event pump and backpressure;
- weight publication and versioning;
- member and population topology.

Phase 0 does not implement this controller. It only verifies the component
construction and state boundaries needed by the later implementation.

## Compatibility findings

### One environment versus vector environments

`SingleAgentEnvRunner` internally wraps its environment in Gymnasium's
single-element vector adapter even when `num_envs_per_env_runner=1`. The runtime
will expose and support exactly one logical environment per rollout actor. It
will not expose vector-env configuration or depend on batching across multiple
environment instances.

### SAC temperature checkpoint gap

RLlib 2.56.1 `Learner.get_state()` includes RLModule, optimizer, metrics, and
weight-sequence state. `SACTorchLearner` keeps the trainable `curr_log_alpha`
temperature outside the RLModule. PyTorch optimizer state does not contain the
current parameter value, so stock learner state is insufficient for a complete
SAC restore.

The Phase 0 harness demonstrated a minimal state-only override that adds and
restores `curr_log_alpha`. Phase 4 productionizes that adapter, also preserves
the target-update cursor, and keeps a parity test against the pinned RLlib
version. The override copies no SAC loss, optimizer, or target-update code.

### Dependency baseline

Ray 2.56.1's own ML requirements pin PyTorch 2.7.0. Its GPU image uses a newer
CUDA build than the target host driver supports. The project therefore keeps
PyTorch 2.7.0 but selects the CUDA 11.8 wheel for the devcontainer. NVIDIA driver
550 is backward-compatible with that runtime, and RTX 3090 supports it.

CPU CI selects the CPU wheel from the same PyTorch release. `uv.lock` contains
both mutually exclusive resolutions.

## Consequences

- Resource ownership remains explicit.
- Tune metrics and checkpoint lifecycle remain available without inheriting
  SAC's synchronous execution plan.
- The project depends on alpha-status RLlib component APIs, so the compatibility
  harness is a required upgrade gate.
- A small SAC checkpoint adapter is required.
- The internal one-element vector wrapper is tolerated but never becomes a
  public runtime feature.

## Rejected alternatives

### Subclass `SAC`

Rejected because it creates the replay, runner group, learner group, and
synchronous training loop that this project must replace.

### Subclass `Algorithm` and suppress setup piecemeal

Rejected because too much inherited lifecycle assumes Algorithm-owned
components. The resulting class would be more fragile than a small composed
Trainable.

### Reimplement SAC

Rejected. It adds algorithmic risk and provides no value for the runtime's
actual contribution.
