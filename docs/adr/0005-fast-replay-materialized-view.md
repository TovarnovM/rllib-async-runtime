# ADR 0005: publish one correctness-first learner-local replay view

- Status: Accepted
- Date: 2026-07-24

## Context

Learners need local transition sampling without putting the authoritative Ray
actor on every GPU update. Snapshot and delta application must not expose a
cursor, payload manifest, and sampling index from different mutations. A stale
cursor must also have one explicit recovery path.

The final hot-path representation and concurrency profile are not yet measured.
Introducing a background executor, reader epochs, and a batch queue in the same
change as the first production learner-local replay would make failures harder
to distinguish from synchronization errors.

## Decision

Phase 3A introduces `FastReplay` as a single-owner learner-local view. It:

- bootstraps from a complete `ReplaySnapshot`;
- validates every transaction in a `ReplayDelta` against a candidate manifest;
- rejects foreign, discontinuous, and compacted cursor updates without changing
  the current view;
- materializes a cumulative transition index in authoritative FIFO order;
- selects a uniform integer over all retained transitions;
- publishes cursor, manifest, byte total, and sampling index together through
  one immutable replacement view.

Episode envelopes and their immutable payloads are reused by reference.
Snapshot and delta rebuilds copy only the ordered manifest and cumulative
sampling index. A sampling call captures one view for both coordinate selection
and payload lookup, so an atomic replacement cannot mix old coordinates with a
new manifest.

The implementation remains behaviorally equivalent to
`ReferenceFastReplay`. Randomized tests cover chunked deltas, journal
compaction, snapshot resync, FIFO eviction, duplicate delivery on the
authoritative side, and exact sampled-coordinate parity for equal RNG state.

## Consequences

- Learner sampling performs no per-transition RPC.
- Invalid snapshots and deltas leave the published local view unchanged.
- A `full_resync_required` delta is not partially interpreted; the owner must
  fetch and load a complete authoritative snapshot.
- Uniform sampling is by transition, not by episode.
- Rebuild cost is linear in retained episodes and runs synchronously in Phase
  3A.
- `FastReplay` assumes one writer. Background rebuilding and concurrent-reader
  lifetime management remain Phase 3B work.
- The local replay is derived state and is rebuilt after recovery rather than
  checkpointed independently.

## Rejected alternatives

### Delegate production behavior to `ReferenceFastReplay`

Rejected because the reference model rebuilds the cumulative index for every
sample. It remains an independent correctness oracle.

### Mutate a shared manifest before rebuilding the index

Rejected because exceptions or concurrent sampling could observe partial delta
application or an index that references an evicted payload.

### Add background rebuild and batching immediately

Rejected for Phase 3A. Those mechanisms require their own concurrency,
backpressure, and reader-lifetime gates and belong in Phase 3B.
