# Architecture boundary

The project is a thin Ray-native runtime around RLlib components, not a new
implementation of SAC.

The central design is:

> Authoritative episode store + learner-specific materialized replay views.

Phase 0 verified the RLlib seams that later phases depend on:

- a single-environment `SingleAgentEnvRunner` can return one complete episode;
- a local SAC `LearnerGroup` can update from replay-sampled episodes;
- inference weights can be installed between complete episodes;
- module, target-network, optimizer, and SAC temperature state can be restored;
- a standard RLlib `Algorithm` eagerly owns a control plane that the custom
  runtime must instead compose explicitly.

Phase 1 adds the process-boundary contracts and a deterministic in-process
correctness oracle:

- immutable, schema-versioned whole-episode envelopes;
- idempotent FIFO storage bounded by transitions and approximate bytes;
- atomic add/evict transactions, snapshots, deltas, and resync cursors;
- a materialized reference replay that samples uniformly over transitions.

Phase 2 adds the authoritative process boundary:

- one synchronous Ray actor serializes every store operation;
- actor methods remain finite and perform no batch sampling;
- replay checkpoints preserve retention configuration, cursor, journal and its
  base manifest, payload manifest, commit-ordered identity/retention metadata,
  and metrics;
- save uses checksummed atomic replacement on a POSIX filesystem;
- restore replays and validates exact FIFO evictions before replacing live
  state.

Retained episode payload and the mutation journal are bounded. Exact
full-generation conflict detection and journal-base validation keep one ID,
fingerprint, transition count, and approximate byte size per committed episode,
so total metadata is not yet bounded. This limitation is measured explicitly
rather than hidden behind the FIFO payload capacity.

Phase 3A adds the correctness-first learner-local materialization boundary:

- a snapshot constructs one immutable payload manifest and cumulative
  transition index;
- each delta is fully validated against a candidate manifest before one
  replacement view is published;
- stale or foreign cursors leave the old view untouched and require snapshot
  bootstrap;
- sampling captures one view, selects a uniform flattened transition offset,
  and performs no Ray RPC;
- manifest/index rebuilds reuse immutable episode envelopes rather than copying
  payloads.

Phase 3B moves index construction and batch preparation off the consumer path:

- the latest validated target manifest/cursor is distinct from the immutable
  active sampling view;
- one coalescing worker publishes only the latest completed index revision;
- sampling captures a strong view lease, so an in-flight reader retains any
  payload evicted during a concurrent swap;
- stale and failed builds never replace the active view;
- a flat collator produces contiguous NumPy columns;
- one bounded batch producer provides explicit lifecycle, queue backpressure,
  data-wait, delta-lag, and rebuild metrics.

Phase 3B does not add a second payload graph. Active and retired views contain
only tuples of references to immutable episode envelopes, and Python ownership
defers reclamation until readers release those views.

Phase 4 adds the algorithm boundary:

- a validated flat transition schema becomes one RLlib `MultiAgentBatch`;
- one local `LearnerGroup` continues to own SAC loss, optimizer, temperature,
  and target-network updates;
- absolute sampled-step counters drive learning start and RLlib's target
  schedule;
- inference weights are published at a bounded update interval with monotonic
  per-module versions;
- member state restores module, targets, optimizers, SAC temperature,
  target-update cursors, runtime counters, the last published weights, and
  CPU/CUDA RNG state under a checked config/space contract.

The production learner subclass extends state serialization only. It does not
override SAC loss or target-update behavior. The project still contains no
hierarchy or graph encoder.

Phase 5 adds the rollout boundary:

- each actor owns one logical environment and returns one complete episode per
  finite RPC;
- module weights are installed only before sampling, and RLlib's per-transition
  sequence metadata must remain equal to the installed version;
- whole episodes are converted to the existing flat transition codec and retain
  generation-safe, idempotent identities;
- 4–16 actors progress independently through `ray.wait`, without an episode
  barrier;
- a commit slot is reserved before sampling and released only after replay
  acknowledgement, making the high watermark strict;
- high/low hysteresis applies backpressure only at episode boundaries;
- policy-version lag is measured against the latest publication when an episode
  completes.

Phase 6 adds the first complete single-member control plane:

- a finite-call learner actor owns one `FastReplay`, one bounded batch producer,
  and one local RLlib SAC `LearnerGroup`;
- each learner tick requests at most one bounded replay delta and can consume
  several already-local batches;
- the controller keeps at most one learner tick pending while rollout and
  evaluation actors progress independently;
- evaluation actors receive one frozen publication for a complete round and
  have no authoritative replay handle;
- pause prevents new episodes at boundaries, drain completes existing
  sample/commit calls and queued learner work, and stop kills every owned actor;
- Tune reporting exposes controller, rollout, authoritative replay,
  learner-local replay, batching, learner, and evaluation state.

This boundary intentionally does not coordinate a recoverable member
checkpoint. Rebuilding the learner-local view and recreating actors after
restore remain Phase 7.

See [ADR 0001](adr/0001-runtime-boundary.md) for the orchestration decision and
[ADR 0002](adr/0002-episode-replay-quantum.md) and
[ADR 0003](adr/0003-authoritative-and-reference-replay.md) for replay semantics.
[ADR 0004](adr/0004-replay-actor-checkpoint.md) records the actor and checkpoint
boundary, and [ADR 0005](adr/0005-fast-replay-materialized-view.md) records the
Phase 3A learner-local view. [ADR 0006](adr/0006-reader-safe-rebuild-and-batch-pipeline.md)
records background publication and the bounded batch pipeline.
[ADR 0007](adr/0007-rllib-sac-learner-adapter.md) records the SAC adapter,
batch schema, publication, and checkpoint boundary.
[ADR 0008](adr/0008-episode-rollout-and-version-sync.md) records whole-episode
collection, version installation, restart identity, and bounded asynchronous
coordination.
[ADR 0009](adr/0009-single-member-async-sac.md) records learner-host ownership,
the event pump, frozen evaluation, reporting, and graceful lifecycle.
See [the implementation plan](IMPLEMENTATION_PLAN.md) for phase gates.
