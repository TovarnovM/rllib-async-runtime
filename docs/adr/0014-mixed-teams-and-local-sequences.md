# ADR 0014: Mixed teams and learner-local recurrent sequences

## Context

Sparse hierarchical decisions can span different numbers of physical ticks.
The old sparse codec contains aggregated rewards and cannot reconstruct a
return under a new discount. A mixed DQN/SAC team also needs independent
sampling, warm-up, optimizer and target clocks without separate replay copies.
Pinned RLlib 2.56.1 provides losses, optimizers, target networks and module
interfaces, but its default SAC forward drops recurrent inputs and its loss
does not mask padded sequence positions.

## Decision

Keep authoritative commit, deduplication and eviction at whole episodes.
`RawMultiModuleEpisodeCodec` introduces schema 2 with physical start time,
positive duration, one raw reward per tick, explicit state reset and local
subtask termination. The old codecs keep their existing semantics. A legacy
sparse reward sum cannot be migrated to a physical-time trace.

The temporal collector records the actual environment step before RLlib
aggregates rewards. Temporal environments expose integer `world_t`; a call
advances zero ticks for routing or one tick for simulation. A routing call
cannot pay a reward or create a zero-duration training transition. A final
observation is required to close each real decision. Worker invocations in
the example are separate finite local tasks; their completion resets worker
memory without terminating the manager's task.

Graph padding needed by Gym observation spaces is removed at episode packing.
Observations remain raw; changing encoder parameters never rewrites replay.
Structural startup validation is opt-in. Temporal and protocol invariants
necessary for correctness remain enforced at their ownership boundaries.

`FastReplay` keeps one local replica. Its immutable sequence index stores
coordinates into each agent/reset history. Sampling is uniform over real
learning starts, including short tails, and holds an episode reference as a
read lease. Full prefix, learning segment and n-step continuation are local
views. They are not separately committed or transported.

`SequenceConfig(learning_length=..., burn_in=None)` reconstructs the full
available prefix. A numeric `burn_in` is explicitly approximate, including
zero. Algorithm `n_step` counts decisions; the prepared RLlib `n_step` column
contains their summed physical duration so stock losses use the correct
`gamma ** duration`. Raw tick rewards provide the corresponding discounted
return. Truncation bootstraps from the actual final observation; true task
termination suppresses bootstrap. A sampling window boundary is not terminal.

Use ragged histories and flatten only valid learning positions into stock
losses. There is no synthetic padding to mask out. Stateless modules do not
construct a recurrent prefix or history tensor. Independent GRUs live inside
the existing encoder ownership boundaries. Actor, Q1, Q2 and targets each
reconstruct memory using their own current parameters. Continuous critics
append the candidate action after observation-memory encoding, so resampling
an actor action does not rewrite critic history.

`SingleMemberAsyncTeam(rollout_config, module_configs, runtime_config, ...)`
uses the existing controller event loop. Rollout policies declare spaces and
agent mapping. Each module's ordinary DQNConfig/SACConfig owns its model,
batch size, discount, horizon, warm-up and intensity. The runtime resolves
identical inference and learner specs before constructing actors.

One learner host owns one local adapter per module and one replay replica.
Only the enclosing Ray actor reserves CPU/GPU resources. A deterministic
bounded round robin serves ready trainable modules; frozen modules never
receive learner updates. Independent learners avoid advancing unrelated
targets or optimizer schedules during a partial team update.

Mixed training intensity is valid learned positions per new compatible
decision (`None` means 1). Prefix and continuation do not spend this budget.
Only successful replay synchronization adds unique data; duplicate commits
and empty deltas do not. Warm-up does not accrue historical training debt.
Budget, learned positions, optimizer counters and scheduler position persist
through checkpoints. A stalled rollout eventually exhausts its finite budget.

Exploration uses module decision counts with completed-episode totals
piggybacked on existing rollout calls. This is a delayed member-wide estimate,
not a synchronously updated distributed counter. Manager epsilon does not
advance with worker decisions. Target clocks use each consumer's ingested
compatible decisions.

Publication combines inference state for the full team under one epoch.
Epoch and per-module optimizer counts have different meanings. Rollout and
evaluation construct inference-only modules, and install complete bundles
between episodes. No new per-action counter service or payload fetch by the
controller is introduced.

Checkpoints keep one replay plus all learner/target/optimizer states, sampler
RNG, process CPU/CUDA RNG, budgets, clocks and the scheduler frontier. Restore
process RNG after constructing and restoring every local learner. The direct
batching mode has no prefetched selection frontier to discard. Environment
state is not saved mid-episode; restored runners use a new generation, as in
the existing runtime.

## Scope and consequences

The new team path requires direct batching (`batch_queue_capacity=0`), finite
episodes and explicit module spaces. Stock `use_lstm`/`use_attention` flags
are not a replacement for the implemented recurrent modules. Full-prefix
cost grows with history length; bounded burn-in trades that cost for an
explicit approximation. Neither mode requires physical streaming commits.

Keep the previous single-policy SAC/DQN entry points and checkpoint contracts.
The old hierarchy example remains a demonstration of its original delayed
choice semantics. The new causal hierarchy and hidden-cue examples use the
full shared-replay runtime and expose frozen/joint and memory modes.

No new dependency, distributed learner or second event loop is introduced.
PER, Noisy DQN, mixed population/PBT and measured learning-quality/performance
campaigns remain the next stage. Functional smoke and numerical parity are
reported separately from learning-quality results and real GPU validation.
