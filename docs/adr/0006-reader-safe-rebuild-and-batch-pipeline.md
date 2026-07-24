# ADR 0006: use revisioned index publication and a bounded batch producer

- Status: Accepted
- Date: 2026-07-24

## Context

Phase 3A made snapshot and delta materialization correct, but rebuilt the
sampling index synchronously. A learner must keep sampling while a newer replay
revision is prepared, without allowing an old rebuild to overwrite newer data
or allowing eviction to invalidate an in-flight reader. The later GPU loop also
needs ready CPU batches without an unbounded producer queue.

This phase still precedes the RLlib SAC adapter. It must define concurrency,
ownership, backpressure, and lifecycle without guessing the final learner batch
schema.

## Decision

`FastReplay` separates two immutable points in time:

- the **target state** is the latest validated snapshot/delta state and owns the
  synchronization cursor and ordered payload manifest;
- the **active view** is the revision currently used for sampling and contains
  an aligned tuple of episode references plus the cumulative transition index.

`cursor`, `episode_ids`, totals, and `get_snapshot()` describe the accepted
target state. `active_cursor` describes the sampler. Their difference is
reported as delta lag.

One background thread coalesces target revisions. It builds from a tuple of
immutable episode records and publishes with one reference swap only when the
captured revision is still the latest. A completed stale build is discarded.
Snapshot bootstrap/resync remains synchronous so it establishes a usable view
before returning.

Sampling captures one strong reference to the active view. That reference is a
read lease: after a swap, an old view and any payload evicted from the target
manifest remain alive until the in-flight call completes. No manual epoch table
is required for Python-owned payloads.

Background failures leave the previous active view usable and are surfaced by
`wait_for_idle()`. `close()` prevents new work and, by default, drains the
latest rebuild before joining the worker.

The batch side consists of:

- `FlatBatchCollator`, which requires flat mappings with identical string keys
  and stacks numeric or boolean leaves into contiguous NumPy columns;
- `BatchProducer`, one explicit background producer with a bounded FIFO queue;
- lifecycle operations `start`, `pause`, `resume`, `drain`, and `stop`;
- consumer data-wait, queue high-watermark, queue-full backpressure, production,
  consumption, drop, and failure metrics.

The producer samples only from the active view. A batch begun before an index
swap may therefore complete from the old view; batches begun after the swap
capture the new view.

## Consequences

- Sampling continues while an index is rebuilt.
- No stale rebuild can roll the sampler cursor backward.
- FIFO eviction does not invalidate in-flight readers.
- Accepted sync state may lead the active sampler briefly; this lag is explicit
  and observable.
- Payload objects are reused by reference. Rebuilds allocate index tuples and
  reference tuples, not a second payload graph.
- Queue capacity is a hard upper bound. A faster producer waits and records
  backpressure instead of accumulating batches.
- An empty replay causes bounded producer waiting rather than a busy failure.
- The flat collator deliberately rejects nested, ragged, string, and arbitrary
  object columns. Graph and multi-module collation remain later phases.
- Phase 3B NumPy CPU batches are not pinned and are not yet converted to
  RLlib's exact learner input. Phase 4 owns exact SAC conversion; explicit
  pinning and CUDA prefetch remain part of the later `LearnerHost` integration.

## Rejected alternatives

### Publish every completed rebuild

Rejected because a slower old rebuild could overwrite a newer active view.

### Mutate one shared index in place

Rejected because readers could observe mixed cumulative lengths and payload
references.

### Delete evicted payloads immediately

Rejected because readers that already captured the old index still need those
episode objects.

### Use an unbounded queue

Rejected because producer/consumer rate mismatch would become unbounded memory
growth rather than measurable backpressure.

### Collate directly to RLlib SAC tensors

Rejected because the production SAC adapter and its exact learner input contract
are Phase 4 work.
