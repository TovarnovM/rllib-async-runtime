# ADR 0008: collect whole episodes with boundary-only weight updates

- Status: Accepted
- Date: 2026-07-24

## Context

Phase 4 publishes immutable-by-contract inference weights with monotonic
per-module versions. The runtime now needs multiple independent environment
actors to generate replay data without introducing an episode barrier,
mixing behavior versions inside an episode, or accumulating unbounded commit
RPCs.

Retries and actor replacement also need deterministic identities. A restarted
runner cannot safely continue its old local sequence unless that sequence is
checkpointed, which is deferred to the recovery phase.

## Decision

`EpisodeRunner` owns exactly one RLlib `SingleAgentEnvRunner` and collects
exactly one complete single-agent episode per call. The configured environment
is wrapped in a Gymnasium `TimeLimit`; a returned episode is rejected unless it
is terminated or truncated within `max_episode_steps`.

The runner installs a `WeightsDescriptor` only before starting a sampling call.
Phase 5 supports exactly one `default_policy` module. The descriptor version is
installed as RLlib's weight sequence number, and every transition in the
returned episode is checked against that number. A stale publication is an
idempotent no-op; reusing one version for a different learner-update
publication is an error.

The RLlib episode is encoded into the existing flat replay schema with
observation, next-observation, action, reward, terminated, and truncated
columns. The immutable `EpisodeEnvelope` records the descriptor version
actually used.

Episode identity is:

```text
member_id/runner_id/runner_generation/local_episode_seq
```

Member and runner IDs must be individual path segments. A successful
collection advances `local_episode_seq`. Replacing an actor increments
`runner_generation` and resets its local sequence to zero, so the first
post-restart identity cannot collide with an earlier generation.

`AsyncRolloutGroup` owns 4–16 actors and exposes a finite, non-blocking
`poll()` operation. It keeps at most one sampling call active per actor and
uses `ray.wait` over independently completing sample and commit RPCs; there is
no all-runner wait.

Before starting an episode, the group reserves one slot that remains occupied
until that episode's replay commit is acknowledged. Therefore:

```text
pending sample calls + pending episode commits <= high watermark
```

This is deliberately more conservative than limiting commit object references
after collection. It makes the commit bound strict even when several active
actors finish at once. Once the high watermark blocks an episode boundary,
sampling resumes only after the outstanding count reaches the low watermark.

Policy-version lag is measured when a completed episode returns as the latest
published version minus the behavior version recorded by that episode. The
group keeps bounded percentile windows for lag and episode duration.

Duplicate delivery remains safe because the group preserves the envelope ID
and the authoritative replay actor already implements idempotent commit.

## Consequences

- One episode cannot contain two versions of `default_policy`.
- A fast actor can collect and commit while slower actors remain inside their
  own sampling calls.
- Backpressure never interrupts an environment mid-episode.
- Pending commit work is bounded before it is created.
- Restarted actors have collision-free identities without persisting their old
  local sequence.
- Weight propagation may intentionally lag by complete episodes; that lag is
  measured rather than hidden.
- Phase 5 remains a rollout subsystem. Phase 6 will connect its finite polling
  API to replay synchronization, batch consumption, learner updates, and Tune
  reporting.

## Rejected alternatives

### Update module state during an episode

Rejected because replay provenance would no longer identify the behavior
policy that produced each transition.

### Wait for all actors before committing a rollout round

Rejected because the slowest environment would impose a global episode barrier
and waste independent actor capacity.

### Apply backpressure only after commit RPCs exceed the limit

Rejected because already-running episodes can finish concurrently and overshoot
the bound. Reserving a commit slot before sampling makes the limit strict.

### Reuse the pre-restart generation and local sequence

Rejected because an uncheckpointed actor cannot know whether its last result
was committed before failure.
