# ADR 0003: separate authoritative storage from materialized replay views

- Status: Accepted
- Date: 2026-07-24

## Context

Population members share rollout data but need independent sampling state,
future priorities, sequence settings, and batch pipelines. Performing one RPC
per sampled transition would put the authoritative Ray actor on the learner hot
path. Giving every learner an unrelated replay buffer would lose a common
retention and recovery boundary.

The optimized representation is not known yet. Adopting RLlib episodes, packed
NumPy arrays, or object-store blobs before measuring the real workload would
mix correctness with a premature storage decision.

## Decision

Use one authoritative `EpisodeStore` and one materialized replay view per
learner.

The authoritative side owns:

- idempotent commit;
- FIFO retention by transitions and approximate bytes;
- generation plus monotonic mutation cursor;
- bounded mutation journal;
- snapshots and chunked deltas.

Each delta transaction contains one added episode and all evicted episode IDs
caused by that commit. A cursor from another generation, from the future, or
older than the retained journal requires a full snapshot.

Phase 1 provides `ReferenceFastReplay`, a deterministic in-process correctness
oracle. It applies each delta atomically and samples a uniform integer over the
flattened transition range. Episode selection is therefore proportional to
episode length.

The reference `FlatEpisodeCodec` stores each transition as immutable pickle
bytes. This protects the store from caller mutation and makes process-boundary
round trips explicit. It is trusted-internal data only and is deliberately not
the promised high-performance representation.

Phase 3 may replace the learner-local index and payload layout, but the
optimized implementation must remain equivalent to this reference model under
randomized add/evict/delta tests.

## Consequences

- Learner sampling performs no per-transition RPC.
- Replay lag is possible and must be measured.
- Journal compaction has an explicit full-resync path.
- The immutable reference representation is slower and may use more memory
  than packed arrays.
- A later codec can wrap RLlib episodes or packed arrays without changing
  commit, cursor, retention, and synchronization semantics.

## Rejected alternatives

### Sample remotely from one replay actor

Rejected because learner throughput would depend on repeated RPC, Python object
serialization, and shared sampling state.

### Copy independent replay buffers into each member

Rejected because ingest, retention, deduplication, and recovery would diverge.

### Treat pickle bytes as the final optimized layout

Rejected. The Phase 1 representation is a correctness oracle, not a performance
claim.
