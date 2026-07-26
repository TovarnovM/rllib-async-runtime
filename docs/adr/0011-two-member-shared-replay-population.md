# ADR 0011: share one replay across two fixed Tune members

- Status: Accepted
- Date: 2026-07-26

## Context

Phase 7 gives one member complete ownership of its replay actor and writes that
replay into every Tune checkpoint. Repeating this topology for two Tune trials
would either create two unrelated authoritative stores or copy the same replay
into both member checkpoints. It would also let stopping either trial destroy
state still needed by the other.

Phase 8 must prove population topology and recovery without introducing PBT
selection semantics. The two members need independent learner actors,
optimizers, weights, sampling RNG, and `FastReplay` views while both ingesting
and learning from one authoritative uniform replay.

## Decision

`PopulationLauncher` accepts exactly two fixed `PopulationMemberSpec` values.
Member IDs must be unique, and the retention, journal, and replay CPU settings
must match. The launcher creates exactly one `ReplayActor` with a unique Ray
name, the current Ray namespace, and `lifetime="detached"`.

Tune receives the actor name and namespace, not a serialized actor handle.
Each trial resolves the same named actor during setup. Its placement group
therefore omits replay resources and contains only that trial's controller,
learner, rollout, and evaluation resources. The launcher expands one nested
grid-search choice into exactly two trials, sets `max_concurrent_trials=2`,
disables actor reuse, and installs no scheduler. There is no exploit/explore
operation in Phase 8.

`SingleMemberAsyncSAC` distinguishes owned and external replay:

- a standalone member creates, checkpoints, and stops its own replay exactly
  as in Phase 7;
- a population member receives the external actor, never kills it, and writes
  only `member.snapshot` when Tune checkpoints the trial;
- stopping or failing one member therefore cannot stop the other member or the
  shared replay.

Each learner continues to own its own `FastReplay`, batch producer,
`SACLearnerAdapter`, optimizer state, RNG state, and weight namespace.
Authoritative replay metrics report retained episode and transition counts
grouped by `producer_member_id`. Learner-local metrics report the same
composition for the immutable active sampling view, not its potentially newer
target state. This makes the shared-data contract observable without adding a
sampling policy beyond uniform transition sampling.

Population persistence is a separate bundle:

```text
population.snapshot
replay.snapshot
members/
  member-0/member.snapshot
  member-1/member.snapshot
```

Both Tune trials must produce a member checkpoint. After the trials have
terminated and their rollout actors can no longer mutate replay, the launcher
fetches the shared authoritative state once and publishes the population
manifest last. Every file is checksummed and atomically replaced inside a new
checkpoint directory. A partial bundle has no valid final manifest and is
rejected.

The shared replay cursor may be newer than an individual member cursor because
the other trial can commit episodes after that member's cut. Restore accepts
this only when both cursors belong to the same store generation and the shared
cursor is greater than or equal to the member cursor. This is safe: the shared
state is a later authoritative retention view that the member would observe
after catching up; it may include additions and FIFO evictions, but no learner
updates are inferred from those mutations. A member newer than or foreign to
the shared snapshot is rejected. The launcher restores replay once, then
creates both Tune trials from their independent member states. Derived
`FastReplay` views are rebuilt from the newer shared snapshot and are still
never serialized.

Detached lifetime is a recovery boundary, not permission to leak actors.
`PopulationLauncher.close()` explicitly kills the replay after both trials no
longer need it.

## Consequences

- Two Tune members can update concurrently on two GPUs while retaining
  independent learner and optimizer state.
- Both members ingest into and materialize the same retained episode set.
- Member placement groups no longer reserve or own duplicate replay actors.
- One member can stop without affecting the other member or authoritative
  replay.
- Population checkpoints contain one replay payload regardless of member
  count.
- The Phase 8 checkpoint API publishes a stable bundle after a completed Tune
  run. Periodic live population checkpoints would need an explicit
  cross-trial coordination protocol and are deferred.
- True PBT, lineage-aware replay mixing, priorities, and exploit/explore remain
  future work.

## Rejected alternatives

### Put replay in every Tune trial checkpoint

Rejected because it duplicates payload, creates conflicting recovery owners,
and scales checkpoint cost with population size.

### Pass a captured actor handle in the persistent Tune config

Rejected because a handle to a dead pre-recovery actor cannot identify the
newly restored detached replay. Name plus namespace is a stable lookup
contract.

### Require all member cursors to equal the final replay cursor

Rejected because it adds a global barrier solely for checkpointing. A
validated later state from the same authoritative replay is the retention view
the member would observe after synchronization and does not weaken
learner-state correctness.

### Put both members inside one Tune trial

Rejected because it would hide independent trial resources, reporting,
checkpoint state, and the topology needed by a later PBT scheduler.
