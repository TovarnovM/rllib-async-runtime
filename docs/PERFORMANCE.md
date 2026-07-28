# Performance gate

Phase 11 adds reproducible measurement tooling; it does not turn benchmark
numbers from one machine into a portability claim. Correctness and boundedness
remain hard gates. Throughput, latency, and the winning mode are observations
that must be reported together with the generated environment metadata.

Target-GPU execution is intentionally not part of the ordinary development
loop. The complete hardware matrix, reproduction commands, artifact
requirements, and reviewed closure record are maintained in
[`../debt.md`](../debt.md).

## Modes and scope

| Mode | Batch construction | Purpose |
| --- | --- | --- |
| `stock` | RLlib-owned replay and learner pipeline | Single-member SAC baseline |
| `direct` | `FastReplay.sample()` and collation on the learner call path | Runtime without background batching |
| `queued` | One producer thread and a bounded FIFO batch queue | Runtime with background batching |

`batch_queue_capacity=0` is a production runtime mode, not a benchmark-only
shortcut. It preserves the same replay sampler RNG state and lifecycle but
constructs each requested batch synchronously. Any positive capacity keeps the
existing bounded prefetch thread.

The two-member benchmark supports `direct` and `queued`. Stock SAC has no
equivalent topology with two independent learners sharing one authoritative
episode store, so the harness rejects `--members 2 --mode stock` rather than
presenting a misleading comparison.

## Reviewed target-GPU evidence

The target-workstation matrix was executed and reviewed on 2026-07-28.

| Field | Value |
| --- | --- |
| Run ID | `gpu-aee9c5a-20260728T090201Z` |
| Hardware | Two NVIDIA GeForce RTX 3090 GPUs |
| Software | Python 3.11.13, Ray 2.56.1, PyTorch 2.7.0+cu118, CUDA 11.8 |
| Clean commits | `aee9c5aef4e5febd14020694698108fcac040190`, `1b8904bc9c46e69e731ffaa9d552de0b58373224` |
| Evidence | 70 valid JSON reports and 180 unique, non-empty profiles |
| Archive SHA-256 | `bbc7591eeda0224495762393f84d46a342a457a78417349464e991c93d0a1c0f` |

The validator accepted every required matrix point and every referenced
profile. All deterministic top-level and member gates passed. The two-member
functional run assigned GPU 0 and GPU 1 to different learners, both learners
updated concurrently, and each replay view received approximately half of its
episodes from each producer. The shared-GNN CUDA run completed 200 learner
updates for `shared_graph` and advanced the module version to 200. Device
telemetry peaked at 62 °C with no thermal-violation or PCIe error samples.

### Measured observations

The following ratios are geometric means across the complete matrix. They use
measured environment throughput and are observations of this run, not
work-normalized speedup claims.

| Comparison | Geometric-mean ratio |
| --- | ---: |
| One-member `direct / stock` | 1.451× |
| One-member `queued / stock` | 1.424× |
| One-member `queued / direct` | 0.982× |
| Two-member aggregate `queued / direct` | 0.995× |

Background batching did not improve the flat end-to-end workload. Direct mode
was classified as batch-supply-bound for 21 of 32 one-member results and 42 of
64 population-member results; the remaining classifications were 10/20
learner-update-bound and 1/2 balanced. Queued mode shifted all 32 one-member
and all 64 population-member results to learner-update-bound without improving
throughput.

The component matrix measured 28.6–70.8k flat transitions/s versus
4.4–8.4k graph transitions/s across direct and queued modes. The graph profiles
localized the additional cost in graph normalization/decode, packed
collation, and `index_add_`/`scatter_add_`; optimizer kernels remain part of
the learner cost. Authoritative replay ingest retained substantial headroom
over end-to-end consumption and was not the observed bottleneck. Small models
plus CPU decoding, collation, and orchestration left the GPUs underutilized.

### Interpretation boundary

The environment-throughput ratios above must not be reported as proof that the
runtime is 1.45× faster than stock RLlib:

- with target training intensity 4, median achieved intensity was 1.218
  (`direct`) and 1.177 (`queued`) for one member, and 1.193/1.176 for
  population members, so learner-update debt accumulated;
- the stock reports did not expose a usable RLlib trained-step counter and
  recorded zero trained steps;
- each matrix point was measured once, without repetitions or confidence
  intervals.

The evidence is sufficient for the Phase 11 contract: reproducible execution,
bounded runtime state, complete measurement coverage, and an evidence-backed
bottleneck finding. A public speedup claim requires a separate
work-normalized, repeated benchmark.

## Benchmark inventory

| Command | Boundary measured | Deterministic gates |
| --- | --- | --- |
| `benchmarks/replay_ingest.py` | episode encoding already complete; authoritative commit, fingerprinting, journal, and FIFO eviction timed | transition, byte, and journal capacity; commit/rejection accounting |
| `benchmarks/fast_replay_sampling.py` | learner-local uniform sampling plus flat or graph collation in direct/queued modes | queue bound, mode, failures, consumed batches, data wait, batch-build timing |
| `benchmarks/end_to_end_throughput.py` | rollout, authoritative replay, local replay, batch supply, stock SAC learner, and runtime event pump | pending RPCs, queue, both replay capacities, failures, timing presence, completion |

The synthetic flat and graph episodes are seeded and deterministic. The graph
workload varies between one and four live nodes and exercises the same
`GraphEpisodeCodec` and `GraphBatchCollator` contracts as the Phase 10 example.

## Cheap CPU instrumentation check

Run inside the devcontainer:

```bash
set -euo pipefail

artifact_dir="artifacts/performance/cpu-smoke-$(git rev-parse --short HEAD)"
mkdir -p "$artifact_dir/profiles"

timeout --signal=TERM --kill-after=30s 600s \
  uv run --locked --extra cu118 --group dev \
  python -m benchmarks.replay_ingest \
  --payload all \
  --episodes 1000 \
  --episode-length 32 \
  --profile-dir "$artifact_dir/profiles" \
  --output "$artifact_dir/replay-ingest.json"

timeout --signal=TERM --kill-after=30s 600s \
  uv run --locked --extra cu118 --group dev \
  python -m benchmarks.fast_replay_sampling \
  --payload all \
  --mode all \
  --episodes 256 \
  --episode-length 32 \
  --batch-size 128 \
  --batches 200 \
  --profile-dir "$artifact_dir/profiles" \
  --output "$artifact_dir/fast-replay.json"
```

This check needs no Ray cluster or accelerator. It proves that both payload
paths, both batch modes, reports, profiles, and deterministic gates work. Do
not compare its throughput against target-hardware results.

## Full CPU matrix

The following loop covers the planned runner counts, episode lengths, batch
sizes, and update-to-data ratios for a single member. It runs each point long
enough to warm up and then measures a separate sample window.

```bash
set -euo pipefail

run_id="cpu-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="artifacts/performance/$run_id"
mkdir -p "$artifact_dir/profiles" "$artifact_dir/ray"

for runners in 1 4 8 16; do
  for episode_length in 32 512; do
    for batch_size in 128 512; do
      for intensity in 1 4; do
        stem="single-r${runners}-e${episode_length}-b${batch_size}-u${intensity}"
        timeout --signal=TERM --kill-after=30s 2100s \
          uv run --locked --extra cu118 --group dev \
          python -m benchmarks.end_to_end_throughput \
          --members 1 \
          --mode all \
          --runner-count "$runners" \
          --episode-length "$episode_length" \
          --batch-size "$batch_size" \
          --training-intensity "$intensity" \
          --warmup-timesteps 2000 \
          --measure-timesteps 20000 \
          --max-duration-s 1800 \
          --storage-path "$artifact_dir/ray" \
          --profile-dir "$artifact_dir/profiles" \
          --output "$artifact_dir/$stem.json"
      done
    done
  done
done
```

The two-member CPU topology is a boundedness and contention gate, not a stock
RLlib parity measurement. Its largest point uses 12 runners per member so the
shared topology requires 29 Ray CPU slots and fits the target 32-slot
workstation while retaining four runner-count levels:

```bash
set -euo pipefail

run_id="cpu-population-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_dir="artifacts/performance/$run_id"
mkdir -p "$artifact_dir/profiles" "$artifact_dir/ray"

for runners in 1 4 8 12; do
  for episode_length in 32 512; do
    for batch_size in 128 512; do
      for intensity in 1 4; do
        stem="population-r${runners}-e${episode_length}-b${batch_size}-u${intensity}"
        timeout --signal=TERM --kill-after=30s 2100s \
          uv run --locked --extra cu118 --group dev \
          python -m benchmarks.end_to_end_throughput \
          --members 2 \
          --mode all \
          --runner-count "$runners" \
          --episode-length "$episode_length" \
          --batch-size "$batch_size" \
          --training-intensity "$intensity" \
          --warmup-timesteps 2000 \
          --measure-timesteps 20000 \
          --max-duration-s 1800 \
          --storage-path "$artifact_dir/ray" \
          --profile-dir "$artifact_dir/profiles" \
          --output "$artifact_dir/$stem.json"
      done
    done
  done
done
```

Both loops deliberately use no fixed throughput threshold. CPU CI and shared
development hosts are noisy; only correctness, measurement presence, and
resource bounds are deterministic pass/fail conditions. The external
`timeout` also covers Ray initialization and actor scheduling before the
harness can enforce its own `--max-duration-s` measurement deadline. The
harness also fails before scheduling when Ray reports fewer than
`runner_count + 2` CPU slots for one member or
`2 * (runner_count + 2) + 1` for two members.

## Reports and interpretation

Every JSON document has `schema_version: 1` and records:

- benchmark name and exact parameters;
- Ray cluster resources observed by the end-to-end harness;
- generation time in UTC;
- Python, platform, Ray, PyTorch, CUDA visibility/version, Git commit, and
  dirty-tree state;
- one result per selected payload or runtime mode;
- all deterministic gates;
- paths to optional profiling and Ray artifacts.

End-to-end runtime results additionally contain the complete final runtime
report and these measured signals:

- post-warm-up environment steps, learner updates, duration, and rates; the
  two-member path establishes per-member baselines only after both members
  reach warm-up and ends its shared timer before Tune checkpoint teardown;
- `batching.data_wait_s`, calls, and timeouts;
- `batching.batch_build_ms_p50/p95`;
- `learner.data_wait_fraction`;
- `learner.batch_queue_empty_fraction`;
- `learner.update_time_ms_p50/p95`;
- learner updates and sampled transitions per second;
- rollout backpressure and pending-RPC high-water marks;
- authoritative and learner-local transition/byte occupancy.

`bottleneck_indicator` is a pressure classifier, not a profiler. It points to
batch supply, rollout/replay ingest, learner update, batch build, or an
ambiguous balanced case. The profile and system measurements determine the
root cause.

Inspect a driver-local profile with:

```bash
uv run --locked --extra cu118 --group dev \
  python -m pstats artifacts/performance/<run>/profiles/<profile>.prof
```

Useful interactive commands are `sort cumulative`, `stats 40`, and
`callers <function>`.

The `cProfile` artifact observes only the process and thread in which it is
created. In direct mode this includes sampling and collation. In queued mode
the driver profile mostly exposes queue wait; producer work is represented by
the batch-build percentiles, and Ray actor or accelerator work requires the
target-system evidence described in `debt.md`.

Representative CPU component profiling during Phase 11 localized:

- flat direct sampling primarily in transition decode/deserialization, then
  collation;
- graph direct sampling in graph normalization/decode and packed collation;
- authoritative ingest in payload fingerprint serialization;
- queued consumer time predominantly in bounded queue wait.

Those observations justify the instrumentation boundaries. The reviewed
target-workstation findings above supersede them for Phase 11 closure; neither
set of measurements is a portable throughput claim.

## Required invariant gates

An end-to-end runtime result is valid only when all reported gates pass:

- controller pending RPC high-water mark does not exceed its configured bound;
- batch queue high-water mark does not exceed capacity, including zero in
  direct mode;
- authoritative and learner-local replay remain within transition and
  approximate-byte capacity;
- rollout, commit, and batch producer failures remain zero;
- data-wait, batch-build, and learner-update measurements are present;
- the requested measurement window completes before `--max-duration-s`.

The approximate byte capacity covers retained episode payloads. It does not
make the full process memory bounded: exact full-generation deduplication
metadata remains monotonic, as documented in the architecture and README.

## Known limitations

- `FlatEpisodeCodec` and graph payload fingerprints remain trusted-local
  pickle-based reference paths, not a zero-copy production format.
- `FastReplay` indexes immutable payload references; it does not eliminate
  per-sample transition decoding.
- The queued pipeline has one Python producer thread. It does not implement
  pinned-memory or CUDA-stream prefetch.
- Learner update wall time is measured around the RLlib update call. The hot
  path adds no forced CUDA synchronization, so accelerator kernel attribution
  belongs to target-system profiling.
- End-to-end synthetic throughput covers the flat single-agent runtime.
  Variable-size GNN costs are isolated by the graph component benchmark and
  the shared-GNN example; a generic asynchronous graph controller is not
  claimed.
- One logical environment still belongs to each rollout actor.
- The two-member mode measures the custom shared-replay topology only; there
  is no fabricated stock equivalent.
- The reviewed target-hardware evidence closes the Phase 8 and Phase 11 gates,
  but it does not establish a work-normalized speedup.
- Benchmark and profiler outputs live under ignored `artifacts/` paths and
  must be archived externally when used as release evidence.
