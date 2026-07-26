# ADR 0012: preserve sparse hierarchy turns in module-specific replay

- Status: Accepted
- Date: 2026-07-26

## Context

Phase 9 must exercise three heterogeneous policies without turning the example
into a second reinforcement-learning framework. A manager chooses one of two
workers every few environment steps. The manager has a natural `Discrete(2)`
action space, while both workers use continuous actions. An inactive policy
must not acquire a fabricated observation, action, reward, or replay row.

The pinned RLlib 2.56.1 compatibility spike constructed one heterogeneous
`MultiRLModule` with all three SAC modules and completed a learner update. This
removes the need for a continuous proxy action or a project-owned DQN learner.

The existing replay contract stores whole episodes and samples uniformly over
transitions. A hierarchy therefore needs explicit sparse timeline metadata and
direct per-module views without changing authoritative commit, FIFO eviction,
snapshot, or delta semantics.

## Decision

`HierarchySwitchEnv` is a finite integration environment with stable
one-to-one agent/module IDs:

- `manager` acts at environment timesteps divisible by `manager_period`;
- exactly one worker acts at every environment timestep;
- the manager's choice takes effect on the following worker turn;
- the inactive worker is absent from the observation and action dictionaries;
- termination emits final observations for all participating agents.

`MultiModuleEpisodeRunner` delegates policy execution and episode construction
to RLlib's `MultiAgentEnvRunner`. It converts only the action turns that RLlib
actually recorded. Each stored transition carries `env_t`, per-agent
`agent_t`, `agent_id`, and `module_id`. The runner rejects incomplete episodes,
misaligned sparse turns, missing weight sequence metadata, and episodes beyond
its configured finite bound.

The rollout publication contains all module states at one synchronized version.
RLlib's `MultiAgentEnvRunner` exposes one `WEIGHTS_SEQ_NO` for the complete
module state, so Phase 9 deliberately rejects a publication that mixes module
versions. Independent per-module rollout publication would require a different
RLlib runner boundary and is not implied by this example.

`MultiModuleEpisodeCodec` groups immutable encoded transitions by module while
retaining sparse timeline metadata outside the encoded learner data. It is a
trusted-local pickle codec, like the existing flat reference codec. Validation
requires:

- contiguous `agent_t` values starting at zero for every agent;
- strictly increasing `env_t` per agent and no empty environment timestep;
- one stable module per agent;
- exact agreement between payload modules and behavior-version keys;
- exact envelope transition, environment-step, and byte totals.

The authoritative `EpisodeStore` remains codec-agnostic. `FastReplay` builds
one cumulative uniform-sampling index per module alongside its global index.
These indexes contain episode references and cumulative lengths, not copied
transitions. Snapshot publication, asynchronous delta rebuild, stale-build
rejection, FIFO eviction, and reader leases are unchanged.
`ReferenceFastReplay` supplies the corresponding correctness oracle.

`MultiModuleBatchCollator` groups sampled transitions by `module_id` and
removes `env_t`, `agent_t`, and identity metadata before producing the stock
SAC columns. `SACLearnerAdapter.update_modules()` then submits one
heterogeneous `MultiAgentBatch` to the existing RLlib `LearnerGroup`; the
project still owns no SAC loss.

Replay checkpoints continue to serialize only authoritative episode state.
The learner checkpoint already stores the complete heterogeneous
`MultiRLModule`, targets, optimizers, temperatures, counters, publication, and
RNG state. `FastReplay` module indexes remain derived and are rebuilt after
restore.

## Consequences

- All three modules participate without fake inactive-worker transitions.
- Manager frequency is observable from `env_t`, while each agent's training
  sequence remains contiguous in `agent_t`.
- Each module is sampled uniformly over its own retained transitions.
- Module indexes follow the same snapshot/delta/FIFO lifecycle as the global
  replay view.
- Discrete manager and continuous workers use stock RLlib SAC in the pinned
  dependency set.
- The example proves the hierarchy/replay/learner integration boundary; it is
  not a production-ready universal hierarchical controller.
- Generic time-limit synthesis for arbitrary sparse multi-agent environments,
  DQN managers, asynchronous per-module weight publication, and graph
  observations remain outside Phase 9.

## Rejected alternatives

### Emit no-op transitions for the inactive worker

Rejected because it changes the environment's action semantics, biases replay,
and makes module batch counts depend on an implementation artifact.

### Encode the manager choice as a continuous threshold

Rejected because pinned RLlib supports the real heterogeneous action spaces and
the proxy would hide the principal compatibility risk.

### Add an AsyncDQN manager

Rejected because Phase 9 is an integration example and the project has no DQN
runtime, replay-priority, or checkpoint contract.

### Build copied transition lists for every module

Rejected because every delta would multiply retained payload memory and weaken
the reader-safe materialized-view ownership model.
