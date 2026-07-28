# Technical debt

This file is the single source of truth for target-hardware GPU validation.
The commands below are prepared but were **not executed while implementing
Phase 11**. CPU tests, component profiling, and CI do not substitute for this
evidence.

Target environment:

- two NVIDIA RTX 3090 GPUs;
- host driver 550.144.03;
- repository devcontainer with Python 3.11;
- Ray/RLlib 2.56.1;
- PyTorch 2.7.0 CUDA 11.8 wheel selected by the `cu118` extra.

All commands run inside a freshly rebuilt repository devcontainer unless a
step explicitly says that it runs on the host.

## Shared preparation and evidence directory

On the host, verify that the container runtime can expose both devices:

```bash
nvidia-smi -L
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi -L
```

Inside the devcontainer, start from the exact commit under review and create
one ignored evidence directory:

```bash
set -euo pipefail

test -z "$(git status --porcelain=v1)"
uv sync --locked --extra cu118 --group dev

run_id="gpu-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="artifacts/performance/$run_id"
mkdir -p "$artifact_dir/profiles" "$artifact_dir/ray"

git rev-parse HEAD | tee "$artifact_dir/commit.txt"
uv --version | tee "$artifact_dir/uv.txt"
nvidia-smi -L | tee "$artifact_dir/nvidia-smi-L.txt"
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total \
  --format=csv,noheader \
  | tee "$artifact_dir/nvidia-inventory.csv"

uv run --locked --extra cu118 --group dev python - <<'PY' |
  tee "$artifact_dir/python-environment.json"
import json

import ray
import torch

print(
    json.dumps(
        {
            "ray": ray.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        indent=2,
        sort_keys=True,
    )
)
PY
```

The environment check must report `cuda_available: true`,
`cuda_device_count: 2`, and CUDA version `11.8` before any gate below is
attempted. Keep the same `artifact_dir` shell variable for the complete run.

Capture device utilization during all functional and performance commands:

```bash
set -euo pipefail

nvidia-smi dmon -s pucvmet -d 1 -o DT \
  >"$artifact_dir/nvidia-dmon.log" 2>&1 &
dmon_pid=$!
trap 'kill "$dmon_pid" 2>/dev/null || true; wait "$dmon_pid" 2>/dev/null || true' EXIT
```

After the final command, stop the monitor cleanly:

```bash
kill "$dmon_pid" 2>/dev/null || true
wait "$dmon_pid" 2>/dev/null || true
trap - EXIT
```

## Phase 8 — target two-GPU gate

**Status:** open

As a single-member end-to-end accelerator smoke, run the finite Pendulum
example:

```bash
timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev \
  python examples/async_sac_pendulum.py \
  --stop-timesteps 5000 \
  --runner-count 4 \
  --num-gpus 1 \
  2>&1 | tee "$artifact_dir/single-member-pendulum-gpu.log"
```

First verify one learner actor and its checkpoint state on one assigned device:

```bash
timeout --signal=TERM --kill-after=30s 900s \
  uv run --locked --extra cu118 --group dev \
  pytest -vv -m gpu tests/gpu/test_single_gpu_learner_actor.py \
  2>&1 | tee "$artifact_dir/single-gpu-pytest.log"
```

Then run the unresolved Phase 8 topology gate:

```bash
timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev \
  pytest -vv -m gpu tests/gpu/test_two_gpu_population.py \
  2>&1 | tee "$artifact_dir/two-gpu-population-pytest.log"
```

Run the user-facing two-member topology on the same two devices:

```bash
timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev \
  python examples/population_two_members.py \
  --stop-timesteps 20000 \
  --runner-count 4 \
  --num-gpus-per-member 1 \
  --storage-path "$artifact_dir/ray" \
  2>&1 | tee "$artifact_dir/two-gpu-population-example.log"
```

The Phase 8 debt closes only when the two-GPU pytest command passes and the
user-facing topology example completes while proving:

- two members overlap in time and both perform learner updates;
- Ray assigns exactly one distinct accelerator ID to each learner;
- both learner-local replay views contain episodes from both producer members;
- stopping the launcher cleans up without a replay ownership failure.

Until then, Phase 8 remains unchecked in `docs/IMPLEMENTATION_PLAN.md`.

## Phase 10 — shared ego-GNN learner on CUDA

**Status:** prepared, not executed

The example now accepts an explicit learner device count. Run a functional
shared-GNN update sequence on one GPU:

```bash
timeout --signal=TERM --kill-after=30s 900s \
  uv run --locked --extra cu118 --group dev \
  python examples/shared_gnn_multiagent.py \
  --episodes 200 \
  --num-gpus-per-learner 1 \
  2>&1 | tee "$artifact_dir/shared-gnn-gpu.log"
```

Run PyTorch's combined Python and autograd profiler against the same finite
script:

```bash
timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev --with pip \
  python -m torch.utils.bottleneck \
  examples/shared_gnn_multiagent.py \
  --episodes 200 \
  --num-gpus-per-learner 1 \
  >"$artifact_dir/shared-gnn-bottleneck.log" 2>&1
```

The functional command must report `num_gpus_per_learner: 1`, a positive
learner-update count, the `shared_graph` replay module, and an advanced module
version. The profiler report must be retained even if the workload is
CPU-bound; low GPU utilization is a result, not a reason to discard evidence.

## Phase 11 — target-GPU performance matrix

**Status:** open; harness complete, hardware evidence absent

The benchmark matrix records JSON reports and driver-local `cProfile`
artifacts. End-to-end commands combine the harness `--max-duration-s` deadline
with an external process timeout that also covers Ray initialization and actor
scheduling. A deterministic correctness or boundedness gate failure makes the
command fail. The largest prepared points require at least 18 Ray CPU slots
for one member with 16 runners and 29 for two concurrent members with 12
runners each; the harness checks this before scheduling and records the
complete Ray resource map in every end-to-end JSON document.

### Component profiles

Run flat and variable-size graph ingest plus direct/queued batch construction:

```bash
timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev \
  python -m benchmarks.replay_ingest \
  --payload all \
  --episodes 10000 \
  --episode-length 32 \
  --profile-dir "$artifact_dir/profiles" \
  --output "$artifact_dir/replay-ingest-short.json"

timeout --signal=TERM --kill-after=30s 1800s \
  uv run --locked --extra cu118 --group dev \
  python -m benchmarks.replay_ingest \
  --payload all \
  --episodes 1000 \
  --episode-length 512 \
  --profile-dir "$artifact_dir/profiles" \
  --output "$artifact_dir/replay-ingest-long.json"

for episode_length in 32 512; do
  for batch_size in 128 512; do
    stem="fast-replay-e${episode_length}-b${batch_size}"
    timeout --signal=TERM --kill-after=30s 1800s \
      uv run --locked --extra cu118 --group dev \
      python -m benchmarks.fast_replay_sampling \
      --payload all \
      --mode all \
      --episodes 256 \
      --episode-length "$episode_length" \
      --batch-size "$batch_size" \
      --batches 1000 \
      --queue-capacity 4 \
      --profile-dir "$artifact_dir/profiles" \
      --output "$artifact_dir/$stem.json"
  done
done
```

These component commands are CPU-side even on a GPU workstation. They isolate
replay fingerprint/decode/collation costs so that device under-utilization in
the end-to-end runs can be attributed instead of guessed.

### One-member comparison

Run the complete stock/direct/queued cross-product on one GPU:

```bash
for runners in 1 4 8 16; do
  for episode_length in 32 512; do
    for batch_size in 128 512; do
      for intensity in 1 4; do
        stem="gpu-single-r${runners}-e${episode_length}-b${batch_size}-u${intensity}"
        timeout --signal=TERM --kill-after=30s 2100s \
          uv run --locked --extra cu118 --group dev \
          python -m benchmarks.end_to_end_throughput \
          --members 1 \
          --mode all \
          --runner-count "$runners" \
          --episode-length "$episode_length" \
          --batch-size "$batch_size" \
          --training-intensity "$intensity" \
          --warmup-timesteps 5000 \
          --measure-timesteps 50000 \
          --queue-capacity 4 \
          --num-gpus-per-learner 1 \
          --max-duration-s 1800 \
          --storage-path "$artifact_dir/ray" \
          --profile-dir "$artifact_dir/profiles" \
          --output "$artifact_dir/$stem.json"
      done
    done
  done
done
```

`stock`, `direct`, and `queued` share the same environment, model sizes, batch
size, update-to-data ratio, runner count, warm-up target, measurement target,
seed, and learner GPU request. Direct mode uses no producer thread or queue;
queued mode uses the same sampler with a bounded capacity-four queue.

### Two-member shared-replay matrix

There is no stock SAC equivalent for two independent learners sharing one
authoritative replay, so this matrix compares only direct and queued runtime
modes:

```bash
for runners in 1 4 8 12; do
  for episode_length in 32 512; do
    for batch_size in 128 512; do
      for intensity in 1 4; do
        stem="gpu-population-r${runners}-e${episode_length}-b${batch_size}-u${intensity}"
        timeout --signal=TERM --kill-after=30s 2100s \
          uv run --locked --extra cu118 --group dev \
          python -m benchmarks.end_to_end_throughput \
          --members 2 \
          --mode all \
          --runner-count "$runners" \
          --episode-length "$episode_length" \
          --batch-size "$batch_size" \
          --training-intensity "$intensity" \
          --warmup-timesteps 5000 \
          --measure-timesteps 50000 \
          --queue-capacity 4 \
          --num-gpus-per-learner 1 \
          --max-duration-s 1800 \
          --storage-path "$artifact_dir/ray" \
          --profile-dir "$artifact_dir/profiles" \
          --output "$artifact_dir/$stem.json"
      done
    done
  done
done
```

In addition to each member's resource gates, every two-member report must pass
`shared_replay_visible`, `member_execution_overlapped`, and
`accelerator_assignment_valid`.

### Evidence review and closure

After all runs, verify matrix coverage, per-document provenance, passing gate
sets, and referenced profiles:

```bash
uv run --locked --extra cu118 --group dev \
  python -m benchmarks.validate_evidence "$artifact_dir"
```

The validator accepts documents produced by more than one clean commit so a
completed matrix point does not need to be repeated after a harness-only fix.
Every document still must record a full Git commit, `git_dirty: false`, and the
required CUDA inventory. The final output lists every commit represented in
the bundle.

Phase 11 closes only after the evidence bundle demonstrates:

- every deterministic gate passes for every matrix point;
- no pending RPC, batch queue, authoritative replay, or learner-local replay
  grows beyond its configured bound;
- learner data-wait, queue-empty fraction, batch-build p50/p95, and learner
  update p50/p95 are present rather than inferred;
- one-member stock/direct/queued reports preserve exact parameter parity;
- two-member runs overlap, receive distinct accelerator IDs, and both sample
  episodes from both producer members;
- `nvidia-dmon.log`, component profiles, end-to-end profiles, and timing
  metrics support a written bottleneck conclusion for flat and graph paths;
- every JSON records its exact clean Git commit and environment inventory;
- the complete `artifacts/performance/$run_id` directory is archived outside
  the Git repository.

No fixed throughput improvement is a closure criterion: throughput depends on
the workstation and driver state. The required result is a reproducible
comparison, bounded runtime behavior, and an evidence-backed bottleneck
finding. Until that review is complete, Phase 11 remains unchecked in
`docs/IMPLEMENTATION_PLAN.md`.
