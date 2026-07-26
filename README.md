# rllib-async-runtime

Experimental Ray-native asynchronous off-policy runtime built on RLlib.

> Experimental project built on Ray/RLlib; not an official Ray project.

The project has completed its bootstrap through single-member recovery.
It contains the RLlib compatibility gates, deterministic replay reference
model, a serialized Ray `ReplayActor`, atomic trusted-local replay checkpoints,
reader-safe background `FastReplay` index publication, a bounded local batch
pipeline, a checkpoint-complete local SAC adapter, version-aware asynchronous
rollout, replay-isolated evaluation, and a Tune-compatible end-to-end event
pump, and coordinated Tune checkpoint/recovery. It does **not** yet launch a
population or implement hierarchy and graph encoders.

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
uv run --locked --extra cu118 --group dev pytest -m gpu tests/gpu
```

The CPU-only GitHub Actions job uses the same lock file with the mutually
exclusive `cpu` PyTorch extra. Keep the selected extra on the `uv run` command,
not only on the preceding `uv sync`: Ray 2.56 propagates the original `uv run`
arguments to worker processes, and omitting the extra would create workers
without PyTorch.

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
- an asynchronous 4–16 actor group with no global episode barrier;
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
  python examples/async_sac_pendulum.py --num-gpus 1
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
