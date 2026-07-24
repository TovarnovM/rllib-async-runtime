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

Index construction is still synchronous and linear in retained episodes.
Background rebuild, bounded batch queues, collators, and hot-path metrics remain
Phase 3B work. The project still contains no asynchronous execution loop,
hierarchy, or graph encoder.

See [ADR 0001](adr/0001-runtime-boundary.md) for the orchestration decision and
[ADR 0002](adr/0002-episode-replay-quantum.md) and
[ADR 0003](adr/0003-authoritative-and-reference-replay.md) for replay semantics.
[ADR 0004](adr/0004-replay-actor-checkpoint.md) records the actor and checkpoint
boundary, and [ADR 0005](adr/0005-fast-replay-materialized-view.md) records the
Phase 3A learner-local view.
See [the implementation plan](IMPLEMENTATION_PLAN.md) for phase gates.
