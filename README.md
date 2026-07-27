# rllib-async-runtime

Experimental Ray-native asynchronous off-policy runtime built on RLlib.

> Experimental project built on Ray/RLlib; not an official Ray project.

The project has completed its bootstrap through single-member recovery.
Phase 8 adds two concurrent fixed-config Tune members sharing one authoritative
replay; its dedicated two-GPU acceptance test remains recorded in
[`debt.md`](debt.md). Phase 9 adds a sparse hierarchy example with one
discrete manager, two continuous workers, module-specific replay views, and a
heterogeneous stock RLlib SAC learner. Phase 10 adds homogeneous logical agents
using one shared ego-GNN SAC module, variable-size graph replay, and packed
graph batches. Phase 11 adds deterministic component and end-to-end performance
harnesses, a true no-prefetch comparison mode, boundedness gates, and measured
batch/learner timing. Its target-GPU evidence remains open in
[`debt.md`](debt.md). The project also contains deterministic replay oracles, a
serialized Ray `ReplayActor`, atomic trusted-local replay checkpoints,
reader-safe background `FastReplay` index publication, a bounded local batch
pipeline, version-aware asynchronous rollout, replay-isolated evaluation, and
a Tune-compatible end-to-end event pump. It does **not** yet implement PBT
exploit/explore or centralized full-graph inference.

## Development contract

Local development, dependency installation, linting, tests, and debugging are
supported only inside the repository-owned devcontainer. Do not create or use a
host Python virtual environment for this project.

Host prerequisites:

- Docker with Dev Containers support;
- Git and an editor with Dev Containers support;
- NVIDIA driver and NVIDIA Container Toolkit.

The target workstation has two RTX 3090 GPUs and driver 550.144.03. The
devcontainer uses Python 3.11, Ray/RLlib 2.56.1, and PyTorch 2.7.0 with its
CUDA 11.8 wheel. CUDA 11.8 is deliberately selected because it is compatible
with the existing driver while preserving Ray's tested PyTorch release.

Open the repository and choose **Rebuild and Reopen in Container**. The
container automatically runs:

```bash
uv sync --locked --extra cu118 --group dev
```

All remaining commands in this README run inside the devcontainer:

```bash
uv run --locked --extra cu118 --group dev ruff check .
uv run --locked --extra cu118 --group dev ruff format --check .
uv run --locked --extra cu118 --group dev \
  pytest -m "not gpu and not cluster and not stress"
```

The CPU-only GitHub Actions job uses the same lock file with the mutually
exclusive `cpu` PyTorch extra. Keep the selected extra on the `uv run` command,
not only on the preceding `uv sync`: Ray 2.56 propagates the original `uv run`
arguments to worker processes, and omitting the extra would create workers
without PyTorch.

Target-hardware GPU validation is deliberately separate from the normal
development loop. Its exact prerequisites, commands, artifact layout, and
acceptance criteria are maintained in [`debt.md`](debt.md).

## Phase 0 compatibility gate

The tests verify:

- construction of a SAC `RLModule`;
- a local SAC `LearnerGroup` update from a fixed replay sample;
- finite SAC losses;
- module, target-network, optimizer, and SAC temperature state round-trip;
- one complete episode from one logical environment;
- weight installation between episodes;
- one GPU visible inside a one-GPU Ray learner actor;
- the orchestration boundary between RLlib `Algorithm` and a composed Tune
  `Trainable`.

RLlib 2.56.1 does not include the current SAC `log_alpha` value in its stock
learner state. Phase 4 productionizes the minimal state adapter identified by
this gate. [ADR 0001](docs/adr/0001-runtime-boundary.md) records the component
ownership decision.

## Phase 1 replay contract

Phase 1 provides:

- immutable, schema-versioned whole-episode envelopes;
- idempotent commit that survives FIFO eviction, with explicit duplicate conflicts;
- FIFO retention by transition and approximate-byte capacities;
- atomic snapshot/delta synchronization with stale-cursor resync;
- a deterministic learner-local reference view;
- uniform sampling over transitions rather than episodes.

`FlatEpisodeCodec` uses immutable pickle bytes as a correctness-first reference
format. It accepts trusted internal data only and is not the planned
high-performance learner representation. Future Ray and optimized replay
implementations must remain behaviorally equivalent to this reference under
the randomized model tests.

## Phase 2 authoritative replay actor

Phase 2 provides:

- one synchronous Ray actor as the exclusive owner of `EpisodeStore`;
- serialized commit, snapshot, delta, stats, save, and restore operations;
- commit, duplicate, rejection, conflict, eviction, journal, and dedup metrics;
- versioned replay state containing retention configuration, cursor, journal,
  the journal base manifest, retained payloads, and commit-ordered
  deduplication fingerprints with transition/byte retention metadata;
- checksummed atomic checkpoint replacement and validate-before-swap restore;
- a 16-producer concurrency gate and a sustained FIFO-retention stress test.

Replay checkpoints use pickle for trusted local Python state and must never be
loaded from untrusted sources. The checksum detects accidental corruption; it
does not authenticate a checkpoint.

Exact duplicate/conflict detection and journal-base validation after FIFO
eviction require retaining one episode ID, SHA-256 fingerprint, transition
count, and approximate byte size for every successful commit in the current
store generation. Retained training payload and the delta journal are bounded,
but this metadata is intentionally monotonic and exposed through
`deduplication_entries`. A production-scale retry-horizon or generation-rotation
policy remains required before claiming fully bounded process memory.

## Phase 3 learner-local replay

Phase 3A provides:

- snapshot bootstrap and explicit snapshot resync after a stale cursor;
- atomic application of validated delta transactions;
- one immutable materialized view containing payload references, cursor, and
  cumulative transition index;
- uniform transition sampling with no per-sample Ray RPC;
- randomized behavioral equivalence against `ReferenceFastReplay`.

Phase 3B adds:

- a logical target replay state separated from the immutable active sampling
  view;
- coalesced background index rebuilding with stale-build rejection;
- atomic publication and reader leases that retain evicted payloads for
  in-flight samples;
- explicit delta-lag, rebuild-time, and rebuild-failure metrics;
- `FlatBatchCollator` for contiguous numeric/boolean NumPy columns;
- a bounded `BatchProducer` with backpressure, data-wait metrics, and explicit
  `start/pause/resume/drain/stop` lifecycle.

The accepted synchronization cursor may briefly lead `active_cursor` while a
new index is built. Sampling that already captured an old view may finish on
that view; after publication, new calls use only the new view.

The pipeline returns generic CPU NumPy batches; the Phase 4 adapter owns the
validated conversion to RLlib's exact SAC learner input.

## Phase 4 SAC learner adapter

Phase 4 provides:

- `SACLearnerAdapter` around one local RLlib `LearnerGroup`;
- exact `SampleBatch`/`MultiAgentBatch` construction from flat SAC transitions;
- learning-start gating from absolute sampled-step counters;
- RLlib-owned target-network scheduling;
- interval-based, versioned inference-weight publication;
- complete in-memory member state including optimizers, target networks,
  current SAC temperature, target-update cursors, counters, and the last
  published weights, plus CPU/CUDA RNG state and config/space compatibility;
- fixed-batch parity and post-restore next-update tests against stock RLlib SAC.

The adapter contains no SAC loss implementation. The Phase 6 `LearnerHost`
still uses bounded CPU NumPy batches; explicit pinned-memory and CUDA prefetch
remain performance work rather than part of the correctness path.

## Phase 5 episode rollout and version sync

Phase 5 provides:

- one RLlib environment per rollout actor and one complete episode per call;
- explicit `max_episode_steps` time-limit truncation;
- weight installation only between episodes, with per-transition RLlib
  sequence validation;
- immutable flat replay envelopes carrying the behavior version actually used;
- generation- and sequence-based idempotent episode IDs;
- an asynchronous 1–16 actor group with no global episode barrier, where one
  actor is the explicit Phase 11 baseline rather than the production default;
- strict high/low commit-slot watermarks and boundary-only backpressure;
- bounded policy-lag and episode-duration metrics;
- explicit actor replacement that advances `runner_generation`.

The group exposes finite polling, pause/resume/drain, and weight-publication
operations. Phase 6 composes these operations without changing the Phase 5
episode and version contracts.

## Phase 6 single-member Async SAC

Phase 6 provides:

- one finite-call `LearnerHost` actor owning `FastReplay`, bounded batch
  production, and the stock RLlib SAC learner adapter;
- chunked replay synchronization followed by multiple learner-local updates,
  so the hot path does not perform an authoritative replay RPC per batch;
- cumulative learner-update budgeting from newly sampled steps, including
  `SACConfig.training_intensity`, without warm-up catch-up or stalled-rollout
  over-training;
- a Tune `Trainable` event pump with at most one pending learner call and
  explicit bounds for rollout, commit, and evaluation calls;
- separate frozen-weight evaluation actors that never receive a replay handle;
- graceful rollout-boundary pause, drain, resume, and actor shutdown;
- Tune results containing controller, rollout, authoritative replay,
  learner-local replay, batching, learner, and evaluation metrics;
- runnable Pendulum correctness and synthetic throughput examples.

Run the correctness example with:

```bash
uv run --locked --extra cu118 --group dev \
  python examples/async_sac_pendulum.py --num-gpus 0
```

Use `--require-improvement` to make the command fail unless the recorded
evaluation history improves by the requested margin. For a cheap orchestration
measurement:

```bash
uv run --locked --extra cu118 --group dev \
  python examples/async_sac_throughput.py --runner-count 8
```

## Phase 7 checkpoint and recovery

Phase 7 provides:

- a relocatable, checksummed checkpoint directory containing separate
  authoritative replay and member files;
- episode-boundary drain before persistence, with cumulative
  training-intensity preserved;
- complete learner, controller, rollout, evaluation, publication, and
  learner/batch-sampler RNG state;
- reconstruction of `FastReplay` from authoritative replay rather than
  serialization of its payload/index/thread state;
- recreation of every rollout actor at `saved_generation + 1`, with a
  deterministic generation-derived rollout RNG stream for seeded runs;
- safe duplicate episode re-delivery using restored replay deduplication state;
- standard directory-based Tune `save_checkpoint()` and `load_checkpoint()`
  hooks;
- controlled-crash and Tune-continuation integration gates.

The replay snapshot is transferred from its actor through Ray and persisted by
the controller, so the actor does not need direct access to Tune's local
checkpoint path. Checkpoints use trusted Python pickle state and must never be
loaded from untrusted sources.

A successful checkpoint includes all episodes committed before its drain
completed. Restoring it can lose episodes and learner work produced after that
checkpoint; the maximum loss is therefore determined by checkpoint cadence.
There is no loss relative to the contents of the successfully returned
checkpoint itself.

Direct runtime use:

```python
checkpoint = runtime.save_checkpoint("/path/to/new-empty-checkpoint-dir")
restored = SingleMemberAsyncSAC.from_checkpoint(
    sac_config,
    runtime_config,
    checkpoint.directory,
)
restored.start()
```

The destination directory must already exist but must not contain
`member.snapshot` or `replay.snapshot`; use a new directory for every
checkpoint publication.

With `AsyncSACTrainable`, Tune calls the same coordinated save/restore protocol
through its standard checkpoint lifecycle.

## Phase 8 two-member population

Phase 8 provides:

- one uniquely named detached `ReplayActor` created and owned by
  `PopulationLauncher`;
- exactly two fixed-config Tune trials with no PBT scheduler and no actor
  reuse;
- one independent learner actor, optimizer, weight namespace, batch pipeline,
  and `FastReplay` per member;
- replay placement resources reserved once by the launcher rather than once
  per trial;
- authoritative retained and learner active-view composition metrics grouped
  by `producer_member_id`;
- external-replay lifecycle semantics, so stopping one member leaves the other
  member and replay alive;
- a population checkpoint containing one `replay.snapshot`, two independent
  member snapshots, and one checksummed manifest;
- restore from a shared replay snapshot that may be newer than an individual
  member cut, but never older or from another replay generation;
- CPU integration coverage and a real two-GPU gate for the target two-RTX-3090
  workstation.

Run a short CPU topology smoke inside the devcontainer:

```bash
uv run --locked --extra cu118 --group dev \
  python examples/population_two_members.py \
  --stop-timesteps 2000 \
  --num-gpus-per-member 0
```

Each Tune trial writes only `member.snapshot`. After both trials terminate,
`PopulationLauncher.save_checkpoint()` publishes the shared replay once.
Periodic checkpoints while both trials remain live require a future
cross-trial coordination protocol and are not claimed by Phase 8.
The corresponding target-hardware example, still-open validation command, and
required evidence are in [`debt.md`](debt.md).

## Phase 9 sparse hierarchy example

Phase 9 provides:

- a finite `MultiAgentEnv` where a `Discrete(2)` manager selects one of two
  continuous workers every three environment steps;
- sparse observation/action turns: the inactive worker produces no fabricated
  replay transition;
- an RLlib `MultiAgentEnvRunner` adapter preserving `env_t`, per-agent
  `agent_t`, `agent_id`, `module_id`, and behavior-weight versions;
- `MultiModuleEpisodeCodec` and uniform module-specific indexes in both
  `ReferenceFastReplay` and `FastReplay`;
- a collator and `SACLearnerAdapter` path that update all three heterogeneous
  stock SAC modules in one `MultiAgentBatch`;
- delta/FIFO-eviction coverage and a bounded replay plus learner
  checkpoint/restore smoke test.

Run the CPU example inside the devcontainer:

```bash
uv run --locked --extra cu118 --group dev \
  python examples/hierarchy_three_policies.py --episodes 20
```

The three rollout module versions advance together because the pinned RLlib
multi-agent runner installs one sequence number for a complete module state.
The example proves the sparse hierarchy/replay/learner boundary; it does not
claim a generic production hierarchy framework or a DQN manager.

## Phase 10 shared ego-GNN example

Phase 10 provides:

- four homogeneous logical agents mapped to one `shared_graph` SAC module;
- variable-size ego-graphs with one to four nodes and no policy per agent;
- bounded padded arrays only at the static Gymnasium transport boundary, with
  padding removed before replay;
- `GraphEpisodeCodec` plus the existing module-specific authoritative and
  `FastReplay` lifecycle;
- `GraphBatchCollator` output containing concatenated node features, shifted
  edge indices, `graph_ptr`, controlled-node indices, and optional graph
  leaves;
- a pure-PyTorch mean-aggregation encoder wired through RLlib's custom catalog
  extension point;
- stock RLlib SAC losses, optimizers, target networks, updates, and checkpoint
  state;
- coverage for empty edges, one-node and mixed-size graphs, GNN gradients,
  delta synchronization, and learner plus replay checkpoint restore.

Run the CPU example inside the devcontainer:

```bash
uv run --locked --extra cu118 --group dev \
  python examples/shared_gnn_multiagent.py --episodes 20
```

The shared state is across logical agents. Stock SAC still owns separate actor,
critic, twin-critic, and target encoders inside that one module. The example
does not claim one centralized graph forward for the whole environment,
continuous graph SAC, or masked-action semantics.

## Phase 11 performance gate

Phase 11 provides:

- deterministic flat and variable-size ego-GNN replay workloads;
- authoritative ingest and learner-local sampling/collation benchmarks;
- direct batch construction with `batch_queue_capacity=0`, providing a real
  no-background-batching baseline;
- an end-to-end matrix for stock RLlib SAC, direct runtime, and queued runtime,
  including 1/4/8/16 runners for one member and 1/4/8/12 for the two-member
  shared-replay topology, short/long episodes, multiple batch sizes, and
  update-to-data ratios;
- JSON reports with environment and Git revision metadata plus optional
  `cProfile` artifacts;
- deterministic gates for pending RPCs, queue capacity, replay transition/byte
  capacity, failures, and the presence of measured data-wait, batch-build, and
  learner-update timings.

Run the cheap CPU component smoke tests with:

```bash
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.replay_ingest --episodes 1000
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.fast_replay_sampling --batches 200
```

These commands validate instrumentation and invariants; their throughput is not
a portable performance claim. The full CPU matrix, report schema, profiling
guidance, and known limitations are in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md). Phase 11 remains acceptance-open
until the prepared target-GPU matrix in [`debt.md`](debt.md) is executed and
its evidence is reviewed.

## Architecture

The agreed core is:

> Authoritative episode store + learner-specific materialized replay views.

Whole episodes are the unit of commit, eviction, synchronization, and behavior
weight versioning. SAC continues to sample uniformly by transition inside the
learner-local replay.

Read the [architecture boundary](docs/ARCHITECTURE.md) and
[implementation plan](docs/IMPLEMENTATION_PLAN.md) before extending the
runtime.

## License

Apache-2.0.
