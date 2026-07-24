# rllib-async-runtime

Experimental Ray-native asynchronous off-policy runtime built on RLlib.

> Experimental project built on Ray/RLlib; not an official Ray project.

The project has completed its bootstrap, authoritative replay, and
learner-local replay phases. It currently contains the RLlib compatibility
gates, deterministic replay reference model, a serialized Ray `ReplayActor`,
atomic trusted-local replay checkpoints, reader-safe background `FastReplay`
index publication, and a bounded local batch pipeline. It does **not** yet
implement the SAC learner adapter, rollout execution loop, hierarchy, or graph
encoders.

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
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not gpu and not cluster and not stress"
uv run pytest -m gpu tests/gpu
```

The CPU-only GitHub Actions job uses the same lock file with the mutually
exclusive `cpu` PyTorch extra.

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
learner state. The compatibility harness contains a minimal test-only
state adapter, and [ADR 0001](docs/adr/0001-runtime-boundary.md) records the
required production boundary.

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

The pipeline still returns generic CPU NumPy batches. Pinned tensors and the
exact RLlib SAC learner input are Phase 4 concerns.

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
