# ADR 0002: use whole episodes as the replay mutation quantum

- Status: Accepted
- Date: 2026-07-24

## Context

Rollout actors must publish data without exposing partially collected episodes.
Replay retention, behavior-weight provenance, snapshots, deltas, and later
checkpoint recovery also need one shared identity and atomicity boundary.

Transition-level commits would require a second protocol for episode completion,
make deduplication ambiguous after retries, and allow one logical episode to use
multiple behavior-weight versions without an explicit boundary.

## Decision

`EpisodeEnvelope` is the only replay commit value. A complete terminated or
time-limit-truncated episode contains:

- an idempotent `episode_id`;
- producer, runner, generation, and local sequence identity;
- one behavior version per participating module;
- environment and agent step counts;
- schema and approximate byte metadata;
- an immutable codec-owned payload.

Commit, FIFO eviction, snapshot, delta, and checkpoint operations act on whole
episodes. One successful commit produces one monotonically numbered replay
transaction containing the addition and every eviction caused by it.

The recommended episode ID is:

```text
member_id/runner_id/runner_generation/local_episode_seq
```

Reusing an ID for identical content is an idempotent no-op. Reusing it for
different content is an explicit conflict. The authoritative store retains a
compact content fingerprint independently of the live FIFO manifest, so both
guarantees continue to hold after the episode payload is evicted.

## Consequences

- Retried commits cannot duplicate training data.
- Deduplication metadata lives for the complete store generation and must be
  included in authoritative replay checkpoints.
- One episode has unambiguous behavior-weight provenance.
- Learners never observe a partial episode or a half-applied eviction set.
- Retention may exceed an exact transition target by less than one episode only
  during transaction staging; the committed state always satisfies both hard
  capacities.
- Very long or infinite environments require a configured episode time limit.

Transition sampling remains uniform after materialization; whole-episode commit
does not imply uniform episode sampling.

## Rejected alternatives

### Commit individual transitions

Rejected because completion, retries, behavior versions, and recovery would
need additional distributed state.

### Commit arbitrary rollout fragments

Deferred. Sequence fragments may become useful for recurrent training, but they
must first define burn-in, overlap, identity, and version semantics. They are
not a simpler initial correctness boundary.
