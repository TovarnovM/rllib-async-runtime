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
  base manifest, payload manifest, commit-ordered identity/retention metadata,
  and metrics;
- save uses checksummed atomic replacement on a POSIX filesystem;
- restore replays and validates exact FIFO evictions before replacing live
  state.

Retained episode payload and the mutation journal are bounded. Exact
full-generation conflict detection and journal-base validation keep one ID,
fingerprint, transition count, and approximate byte size per committed episode,
so total metadata is not yet bounded. This limitation is measured explicitly
rather than hidden behind the FIFO payload capacity.

Phase 3A adds the correctness-first learner-local materialization boundary:

- a snapshot constructs one immutable payload manifest and cumulative
  transition index;
- each delta is fully validated against a candidate manifest before one
  replacement view is published;
- stale or foreign cursors leave the old view untouched and require snapshot
  bootstrap;
- sampling captures one view, selects a uniform flattened transition offset,
  and performs no Ray RPC;
- manifest/index rebuilds reuse immutable episode envelopes rather than copying
  payloads.

Phase 3B moves index construction and batch preparation off the consumer path:

- the latest validated target manifest/cursor is distinct from the immutable
  active sampling view;
- one coalescing worker publishes only the latest completed index revision;
- sampling captures a strong view lease, so an in-flight reader retains any
  payload evicted during a concurrent swap;
- stale and failed builds never replace the active view;
- a flat collator produces contiguous NumPy columns;
- one bounded batch producer provides explicit lifecycle, queue backpressure,
  data-wait, delta-lag, and rebuild metrics.

Phase 3B does not add a second payload graph. Active and retired views contain
only tuples of references to immutable episode envelopes, and Python ownership
defers reclamation until readers release those views.

Phase 4 adds the algorithm boundary:

- a validated flat transition schema becomes one RLlib `MultiAgentBatch`;
- one local `LearnerGroup` continues to own SAC loss, optimizer, temperature,
  and target-network updates;
- absolute sampled-step counters drive learning start and RLlib's target
  schedule;
- inference weights are published at a bounded update interval with monotonic
  per-module versions;
- member state restores module, targets, optimizers, SAC temperature,
  target-update cursors, runtime counters, the last published weights, and
  CPU/CUDA RNG state under a checked config/space contract.

The production learner subclass extends state serialization only. It does not
override SAC loss or target-update behavior. Phase 4 itself assumes flat
single-module batches; Phase 9 reuses the same learner boundary for a
heterogeneous multi-module batch without changing SAC loss.

Phase 5 adds the rollout boundary:

- each actor owns one logical environment and returns one complete episode per
  finite RPC;
- module weights are installed only before sampling, and RLlib's per-transition
  sequence metadata must remain equal to the installed version;
- whole episodes are converted to the existing flat transition codec and retain
  generation-safe, idempotent identities;
- 1–16 actors progress independently through `ray.wait`, without an episode
  barrier; one actor exists for the explicit Phase 11 baseline;
- a commit slot is reserved before sampling and released only after replay
  acknowledgement, making the high watermark strict;
- high/low hysteresis applies backpressure only at episode boundaries;
- policy-version lag is measured against the latest publication when an episode
  completes.

Phase 6 adds the first complete single-member control plane:

- a finite-call learner actor owns one `FastReplay`, one bounded batch producer,
  and one local RLlib SAC `LearnerGroup`;
- each learner tick requests at most one bounded replay delta and can consume
  several already-local batches;
- cumulative sampled-step progress and `SACConfig.training_intensity` bound the
  total learner updates, while `learner_updates_per_tick` only caps one RPC;
- the controller keeps at most one learner tick pending while rollout and
  evaluation actors progress independently;
- evaluation actors receive one frozen publication for a complete round and
  have no authoritative replay handle;
- pause prevents new episodes at boundaries, drain completes existing
  sample/commit calls and queued learner work, and stop kills every owned actor;
- Tune reporting exposes controller, rollout, authoritative replay,
  learner-local replay, batching, learner, and evaluation state.

Phase 7 adds the recovery boundary:

- checkpoint drain finishes complete episode commits, evaluation, replay sync,
  and only the learner updates admitted by the cumulative intensity budget;
- authoritative replay and member state are separate checksummed files, with
  the member file published last and tied to the replay cursor;
- the controller persists replay state returned through Ray, avoiding an
  actor-local filesystem dependency;
- learner, controller, rollout, evaluation, publication, and RNG state are
  restored, while `FastReplay`, its index, and its batch queue are rebuilt;
- each recreated rollout actor increments its saved generation before sequence
  zero, preventing post-crash episode ID collisions;
- Tune directory checkpoints remain relocatable and reject partial or
  mismatched member/replay state.

Recovery preserves everything represented by the last successful checkpoint.
Episodes and learner work performed after it may be lost, so the loss bound is
the checkpoint interval. Cluster-wide exactly-once execution is not claimed.

Phase 8 adds the population ownership boundary:

- one launcher creates a uniquely named detached authoritative replay actor;
- exactly two fixed-config Tune trials resolve that actor by name and namespace
  without reserving or owning duplicate replay resources;
- each member retains an independent learner actor, optimizer, weight
  namespace, batch pipeline, RNG state, and `FastReplay`;
- authoritative metrics expose retained composition and learner-local metrics
  expose active sampling-view composition by `producer_member_id`;
- member stop and failure paths never kill externally owned replay;
- Tune member checkpoints contain only member state, while one population
  bundle publishes replay once and ties both member cursors to it;
- population restore permits a shared cursor newer than a member cut in the
  same generation, rebuilds both derived views, and rejects an older or foreign
  replay;
- no PBT scheduler, exploit/explore, priorities, or lineage-aware mixing is
  present.

Phase 9 adds the sparse hierarchy boundary:

- a finite manager/worker environment exposes only the manager on its fixed
  cadence and exactly one active worker on every environment step;
- RLlib's `MultiAgentEnvRunner` remains responsible for policy execution and
  `MultiAgentEpisode` construction;
- rollout conversion records only real action turns and preserves `env_t`,
  per-agent `agent_t`, agent/module identity, and behavior versions;
- one immutable multi-module episode payload remains the authoritative
  commit/eviction/checkpoint unit;
- learner-local replay adds cumulative module indexes beside the global index,
  without copying transition payloads or changing snapshot/delta publication;
- the multi-module collator strips provenance only after sampling and sends one
  heterogeneous `MultiAgentBatch` through the existing stock SAC learner;
- all module rollout weights advance as one synchronized publication because
  the pinned RLlib runner exposes one weight sequence number;
- the example does not add DQN, generic hierarchy orchestration, or graph
  encoders.

Phase 10 adds the shared ego-graph boundary:

- all homogeneous logical agents map to one `shared_graph` RLModule and one
  synchronized module version;
- the Gymnasium boundary uses bounded padded graph arrays plus live
  node/edge counts, while `GraphEpisodeCodec` removes padding before replay;
- `GraphEpisodePayload` reuses the existing multi-module authoritative,
  retention, snapshot/delta, checkpoint, and module-index lifecycle;
- `GraphBatchCollator` concatenates variable node sets, shifts edge indices,
  builds `graph_ptr`, and aligns controlled nodes and optional graph leaves;
- `SharedGraphSACCatalog` replaces only RLlib's observation encoders;
  stock SAC still owns actor/critic networks, losses, optimizers, target
  updates, and state;
- the pure-PyTorch encoder performs batched ego-graph message passing with
  `index_add_`; PyTorch Geometric is not required;
- the example does not add a centralized environment-wide graph pass,
  continuous graph SAC, or masked-action semantics.

Phase 11 adds measurement without changing algorithm ownership:

- `batch_queue_capacity=0` keeps the same replay sampler and checkpointed RNG
  state but constructs a batch synchronously on the learner call path;
- positive queue capacity retains the existing single bounded producer thread,
  making direct and queued modes an explicit A/B boundary;
- cumulative data-wait, batch-build, and learner-update timing is reported by
  the components that own those operations;
- end-to-end gates compare pending RPC and queue high-water marks with their
  configured bounds and compare both replay views with transition/byte
  capacity;
- stock RLlib SAC is a baseline only for the equivalent one-member topology;
  no artificial stock two-member shared-replay result is manufactured;
- benchmark JSON records parameters, environment, Git state, invariant gates,
  and optional profiles, while noisy throughput remains an observation rather
  than a CI threshold.

The learner timing wraps RLlib's update call but deliberately adds no CUDA
synchronization to the hot path. Target-accelerator attribution and the Phase 8
two-GPU gate therefore remain explicit hardware evidence in `debt.md`, not an
architectural claim inferred from CPU CI.

See [ADR 0001](adr/0001-runtime-boundary.md) for the orchestration decision and
[ADR 0002](adr/0002-episode-replay-quantum.md) and
[ADR 0003](adr/0003-authoritative-and-reference-replay.md) for replay semantics.
[ADR 0004](adr/0004-replay-actor-checkpoint.md) records the actor and checkpoint
boundary, and [ADR 0005](adr/0005-fast-replay-materialized-view.md) records the
Phase 3A learner-local view. [ADR 0006](adr/0006-reader-safe-rebuild-and-batch-pipeline.md)
records background publication and the bounded batch pipeline.
[ADR 0007](adr/0007-rllib-sac-learner-adapter.md) records the SAC adapter,
batch schema, publication, and checkpoint boundary.
[ADR 0008](adr/0008-episode-rollout-and-version-sync.md) records whole-episode
collection, version installation, restart identity, and bounded asynchronous
coordination.
[ADR 0009](adr/0009-single-member-async-sac.md) records learner-host ownership,
the event pump, frozen evaluation, reporting, and graceful lifecycle.
[ADR 0010](adr/0010-coordinated-member-recovery.md) records coordinated
checkpoint ordering, replay reconstruction, runner recreation, Tune hooks, and
the explicit loss boundary.
[ADR 0011](adr/0011-two-member-shared-replay-population.md) records external
replay ownership, fixed two-trial Tune topology, producer composition metrics,
and single-copy population checkpoints.
[ADR 0012](adr/0012-sparse-hierarchy-and-module-replay.md) records sparse
manager/worker turns, heterogeneous SAC compatibility, module-specific replay
indexes, synchronized publications, and checkpoint derivation.
[ADR 0013](adr/0013-shared-ego-graph-policy.md) records padded rollout
transport, variable-size replay collation, the shared module boundary, and the
pure-PyTorch graph encoder.
See [the performance gate](PERFORMANCE.md) for measurement semantics and
[the implementation plan](IMPLEMENTATION_PLAN.md) for phase gates.
