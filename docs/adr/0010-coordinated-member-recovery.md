# ADR 0010: recover a member from authoritative replay

- Status: Accepted
- Date: 2026-07-25

## Context

Phase 6 can stop every asynchronous layer at an episode boundary, while
Phases 2 and 4 can separately persist authoritative replay and complete SAC
learner state. A usable Tune checkpoint still needs one ordering protocol. It
must not pair learner state with a different replay cursor, serialize the
derived `FastReplay`, reuse pre-crash episode identities, depend on an actor's
local filesystem path, or resume training before the local sampling index is
ready.

The runtime does not promise cluster-wide exactly-once execution. An episode
may have been committed even if its producer did not observe the
acknowledgement, and all work after the last successful checkpoint may be lost
after an unrecoverable process failure.

## Decision

`SingleMemberAsyncSAC.save_checkpoint()` first drains the member:

1. rollout actors finish their current complete episodes and replay commits;
2. evaluation finishes its current frozen round;
3. the learner reaches the authoritative replay cursor and publishes its
   latest due weights;
4. the batch producer pauses, and queued derived batches beyond the cumulative
   training-intensity budget are discarded rather than converted into extra
   updates.

No rollout, commit, evaluation, or learner RPC remains pending at the
checkpoint boundary.

The checkpoint directory must be new for that publication (it may already
exist, as Tune requires, but must not contain either runtime checkpoint file).
The runtime checks this before draining. The directory then contains two
files:

```text
replay.snapshot
member.snapshot
```

The replay actor exports its complete `EpisodeStoreState` through Ray. The
controller writes the replay snapshot in the Tune checkpoint directory, so the
actor and controller do not need a shared local path. The member file is
written last and contains:

- the strict runtime configuration contract;
- complete `SACLearnerAdapter` state, including module, critic, target,
  optimizer, SAC temperature, target-update cursor, publication, sampled-step,
  and CPU/CUDA RNG state;
- the batch sampler RNG state, but no queued batch;
- cumulative controller, rollout, and evaluation state;
- the authoritative replay cursor;
- each rollout runner's last generation.

The learner serializes its tensor-bearing state to an opaque byte payload
inside the learner process. The CPU controller never deserializes CUDA tensors;
restore passes that payload to the newly GPU-assigned learner before decoding
it there.

Both files use checksummed, atomic file replacement. Restore authenticates both
payloads and rejects a member/replay cursor mismatch. Paths inside the member
file are fixed relative names, so Tune may relocate the directory. Reusing a
previous checkpoint directory is rejected: publishing replay over an older
snapshot before publishing the new member file would otherwise weaken
crash-consistency.

Restore creates new components in this order:

1. create a fresh replay actor and validate/install `EpisodeStoreState`;
2. create the learner host, synchronously materialize `FastReplay` from the
   restored authoritative snapshot, and restore SAC and sampler state;
3. obtain the last published inference weights;
4. recreate rollout and evaluation actors;
5. start asynchronous scheduling only after every prior step succeeds.

`FastReplay`, its index, rebuild thread, and batch queue are never serialized.
Normal replay deltas continue from the restored snapshot cursor after new
episodes arrive.

Every restored rollout actor starts at:

```text
saved runner generation + 1, local episode sequence 0
```

This prevents collisions even when the pre-crash controller did not know
whether a runner's last result had reached replay. Re-delivery of an episode
already represented in the snapshot remains safe because the authoritative
deduplication fingerprints are restored with the store.

`AsyncSACTrainable` implements Tune's directory-based `save_checkpoint()` and
`load_checkpoint()` hooks. Tune iteration metadata remains Tune-owned; the
runtime checkpoint owns member-internal counters and actor state.

## Loss boundary

A completed checkpoint contains every episode committed before its drain
finished and every learner update admitted by the cumulative update budget at
that boundary. Restoring that checkpoint loses:

- episodes committed after that checkpoint;
- learner updates and publications performed after that checkpoint;
- incomplete episodes and ephemeral sampled batches that existed only after
  that checkpoint.

Therefore replay loss is bounded by the interval between the last successful
checkpoint and the crash, not by replay capacity. Relative to the contents of a
successfully returned checkpoint, expected replay loss is zero. A partially
written newer directory is rejected; it does not weaken an older successful
Tune checkpoint.

## Consequences

- Tune can continue a single-member trial without serializing `FastReplay`.
- Learner update budgeting and sampled-step counters remain cumulative across
  restore.
- Evaluation history and the next evaluation threshold remain continuous.
- Recovery transfers one complete authoritative replay state through Ray and
  rewrites it in each member checkpoint; incremental or object-store-native
  replay checkpoints remain future performance work.
- Checkpoints contain trusted Python pickle state. Checksums detect accidental
  corruption but do not make untrusted files safe to load.
- Phase 8 may separate population replay checkpoint cadence from individual
  member checkpoints, but it must preserve the same cursor contract.

## Rejected alternatives

### Serialize `FastReplay`

Rejected because it duplicates authoritative payload, persists a derived index
and thread state, and creates two possible sources of truth.

### Reuse saved runner generations

Rejected because local episode sequence is intentionally not checkpointed and
the acknowledgement state of a pre-crash episode can be unknown.

### Let each actor write directly into Tune's local directory

Rejected because actor-local paths need not refer to the controller node on a
Ray cluster.

### Save while rollout and learner calls remain active

Rejected because no write ordering can make independent replay, learner, and
controller files represent one unambiguous asynchronous point in time.
