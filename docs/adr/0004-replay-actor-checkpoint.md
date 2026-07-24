# ADR 0004: serialize authoritative replay and validate checkpoint restore

- Status: Accepted
- Date: 2026-07-24

## Context

Concurrent rollout producers must not expose a partially committed episode or
observe a manifest between addition and its FIFO evictions. Recovery must
preserve the cursor, retained delta suffix, and deduplication identity fixed in
Phase 1. Pickling a live actor or rebuilding only its current manifest would
either couple recovery to Ray internals or reintroduce delayed episodes after
eviction.

## Decision

Use one synchronous Ray actor with `max_concurrency=1` as the exclusive owner of
an `EpisodeStore`. The actor exposes only finite commit, snapshot, delta, stats,
save, and load calls. It performs no learner batch sampling.

The versioned authoritative state contains:

- codec ID and schema version;
- transition, approximate-byte, and journal capacities;
- store generation and mutation sequence;
- retained episodes in FIFO order;
- the retained mutation-journal suffix and its lightweight base manifest;
- fingerprints for every successfully committed episode ID in commit order;
- commit, duplicate, rejection, conflict, and eviction counters.

Checkpoint save serializes that immutable state with trusted-local pickle,
prefixes it with a format marker and SHA-256 checksum, writes and `fsync`s a
temporary file in the destination directory, then atomically replaces the
target and `fsync`s the directory. The checksum detects accidental corruption;
it is not a signature and does not make pickle safe for untrusted input.

Checkpoint load authenticates the byte checksum, deserializes a candidate
state, validates its codec, capacities, manifest, fingerprints, contiguous
journal suffix, and metric invariants. It replays the journal from its saved
base manifest, verifies each exact FIFO eviction against transition and byte
capacity, and requires the result to equal the retained manifest before
constructing a separate `EpisodeStore`. The actor swaps its live store only
after every check succeeds.

## Consequences

- Concurrent producer calls have one unambiguous serialization order.
- A commit acknowledgement maps to exactly one mutation cursor or to an
  explicit duplicate/rejection.
- Checkpoint restore preserves bounded deltas and delayed-retry idempotency,
  including for episodes whose payload has already been evicted.
- Save and load block the actor while performing local filesystem I/O. This is
  acceptable for the Phase 2 correctness implementation and must be profiled
  before large replay checkpoints.
- The filesystem must be local or shared POSIX-compatible storage visible to
  the actor.
- Retained training payload and journal memory are bounded by configuration.
- Exact conflict detection after arbitrary delayed retries requires retaining
  one episode ID and 32-byte digest per committed episode for the entire store
  generation. Therefore total actor memory does not strictly stabilize under
  infinite unique ingest. `deduplication_entries` exposes this growth. Before a
  production-scale claim, the runtime must choose an explicit generation
  rotation or bounded retry-window/watermark contract; silently discarding old
  fingerprints would weaken the accepted Phase 1 semantics.

## Rejected alternatives

### Use an async or concurrent actor

Rejected for this phase because it would require a second lock/transaction
layer around a store already designed for one owner.

### Pickle the live Ray actor

Rejected because Ray process state and actor handles are not the replay
checkpoint contract, and partial restore validation would be impossible.

### Save only retained episodes

Rejected because cursor deltas and deduplication after eviction would become
incorrect after restart.

### Delete old fingerprints to bound memory

Rejected without a separately accepted retry-horizon contract. It would either
accept stale training data again or stop detecting conflicting reuse of an old
episode ID.
