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
  base manifest, payload manifest, exact deduplication state, and metrics;
- save uses checksummed atomic replacement on a POSIX filesystem;
- restore replays and validates exact FIFO evictions before replacing live
  state.

Retained episode payload and the mutation journal are bounded. Exact
full-generation conflict detection keeps one ID and fingerprint per committed
episode, so total deduplication metadata is not yet bounded. This limitation is
measured explicitly rather than hidden behind the FIFO payload capacity.

The project still contains no optimized learner-local replay, asynchronous
execution loop, hierarchy, or graph encoder.

See [ADR 0001](adr/0001-runtime-boundary.md) for the orchestration decision and
[ADR 0002](adr/0002-episode-replay-quantum.md) and
[ADR 0003](adr/0003-authoritative-and-reference-replay.md) for replay semantics.
[ADR 0004](adr/0004-replay-actor-checkpoint.md) records the actor and checkpoint
boundary.
See [the implementation plan](IMPLEMENTATION_PLAN.md) for phase gates.
