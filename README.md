# rllib-async-runtime

Experimental Ray-native asynchronous off-policy runtime built on RLlib.

> Experimental project built on Ray/RLlib; not an official Ray project.

The project has completed its bootstrap and in-process replay-contract phases.
It currently contains the RLlib compatibility gates plus a deterministic
authoritative episode store and materialized reference replay. It does **not**
yet implement the Ray `ReplayActor`, asynchronous execution loop, hierarchy, or
graph encoders.

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
