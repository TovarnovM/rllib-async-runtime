# ADR 0009: compose one bounded asynchronous SAC member

- Status: Accepted
- Date: 2026-07-25

## Context

Phases 1–5 provide finite components but not a training control plane. Whole
episodes can be committed to authoritative replay, a learner-local view can
synchronize and build bounded batches, the SAC adapter can update and publish
weights, and rollout actors can progress without an episode barrier.

Connecting them must not reintroduce RLlib's synchronous `Algorithm` control
plane, make one replay RPC per gradient update, allow unbounded object
references, mix evaluation and training data, or leave actors alive after a
trial stops.

Checkpoint/recovery is a separate concern. The replay and learner already have
component state formats, but coordinating their restore with actor recreation
and local replay rebuilding belongs to Phase 7.

## Decision

`AsyncSACTrainable` is a thin Tune controller around
`SingleMemberAsyncSAC`. It composes RLlib components rather than subclassing
`Algorithm`.

One `LearnerHostActor` owns:

- one `FastReplay`;
- one bounded `BatchProducer`;
- one `SACLearnerAdapter` and its local RLlib `LearnerGroup`;
- replay-sync, update, publication, and local metrics counters.

Actor methods remain finite. A learner `tick()` requests at most one bounded
delta from authoritative replay, then consumes up to a configured number of
already-local batches. Therefore several SAC updates can follow one replay RPC.
Snapshot resync remains explicit when the journal no longer covers the local
cursor.

The controller maintains at most one pending learner tick. The Phase 5 rollout
group retains its strict bound:

```text
pending episode samples + pending commits <= commit high watermark
```

One evaluation round starts at most one call on each evaluation actor. Its
bound is the configured evaluation episode count. The complete controller bound
is consequently:

```text
commit high watermark + evaluation episode count + one learner tick
```

Evaluation actors receive a single copied `WeightsDescriptor` for the round,
sample with exploration disabled, and never receive a replay actor handle.
Training may continue while the frozen round is in flight.

Lifecycle is explicit:

1. `pause` stops scheduling new rollout episodes at episode boundaries and
   pauses batch production after the current safe point;
2. `drain` completes pending sample/commit and evaluation calls, synchronizes
   the learner-local replay to the current authoritative cursor, consumes the
   finite queued batches, and forces any due final publication;
3. `resume` restarts batch production and episode scheduling;
4. `stop` performs a graceful drain by default, then stops or kills every
   learner, rollout, evaluation, and replay actor owned by the member.

Every Tune result contains separate controller, rollout, authoritative replay,
learner-local replay, batching, learner, and evaluation metric trees.

## Consequences

- The first complete training run is available without copying SAC loss or
  using `SAC.training_step()`.
- Authoritative replay remains the source of truth, while learner updates use
  local materialized data.
- Weight propagation remains episode-boundary-only.
- Evaluation cannot accidentally add data to training replay.
- Pending long-lived RPCs and queues have explicit configured bounds.
- Tune placement requests include every non-zero-resource child actor.
- CPU NumPy batches remain the correctness implementation; pinned memory and
  CUDA prefetch are deferred performance work.
- Runtime checkpoint/restore is intentionally unavailable until Phase 7.

## Rejected alternatives

### Run the learner loop forever inside one actor RPC

Rejected because pause, reporting, failure surfacing, and resource ownership
would become implicit and difficult to bound.

### Synchronize replay before every gradient update

Rejected because it turns the authoritative actor into the learner hot path.
One bounded synchronization followed by several local updates preserves the
materialized-view design.

### Reuse training rollout actors for evaluation

Rejected because it would either interrupt data collection or risk committing
evaluation episodes. Dedicated actors make frozen weights and replay isolation
structural.

### Add coordinated checkpoint restore in the same phase

Rejected because actor recreation, authoritative snapshot restore, duplicate
redelivery, and derived `FastReplay` rebuilding form a separate failure
protocol covered by Phase 7.
