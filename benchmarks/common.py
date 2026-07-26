"""Shared builders, reporting, and invariant gates for performance benchmarks."""

from __future__ import annotations

import cProfile
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import ray
import torch
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns

from rllib_async.examples import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    SHARED_GNN_MODULE_ID,
)
from rllib_async.gnn import GraphEpisodeCodec
from rllib_async.protocols import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
    MultiModuleTransition,
)

BENCHMARK_SCHEMA_VERSION = 1
ResultT = TypeVar("ResultT")


class PerformanceGateError(RuntimeError):
    """A correctness or boundedness invariant failed during a benchmark."""


def make_flat_episode(
    sequence: int,
    episode_length: int,
    codec: FlatEpisodeCodec,
    *,
    member_id: str = "member-0",
) -> EpisodeEnvelope:
    """Build one deterministic flat SAC episode outside the timed ingest path."""

    transitions = [
        {
            Columns.OBS: np.asarray(
                [step / episode_length, float(sequence % 7), 1.0],
                dtype=np.float32,
            ),
            Columns.NEXT_OBS: np.asarray(
                [(step + 1) / episode_length, float(sequence % 7), 1.0],
                dtype=np.float32,
            ),
            Columns.ACTIONS: np.asarray(
                [((step + sequence) % 3 - 1) / 2],
                dtype=np.float32,
            ),
            Columns.REWARDS: np.float32(1.0),
            Columns.TERMINATEDS: step == episode_length - 1,
            Columns.TRUNCATEDS: False,
        }
        for step in range(episode_length)
    ]
    payload = codec.encode(transitions)
    return EpisodeEnvelope(
        episode_id=f"{member_id}/runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id=member_id,
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({DEFAULT_MODULE_ID: 0}),
        env_steps=episode_length,
        agent_steps=episode_length,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def make_graph_episode(
    sequence: int,
    episode_length: int,
    codec: GraphEpisodeCodec,
    *,
    member_id: str = "member-0",
) -> EpisodeEnvelope:
    """Build one deterministic variable-size ego-graph SAC episode."""

    transitions = []
    for step in range(episode_length):
        node_count = 1 + ((sequence + step) % 4)
        next_node_count = 1 + ((sequence + step + 1) % 4)
        transitions.append(
            MultiModuleTransition(
                env_t=step,
                agent_t=step,
                agent_id="agent-0",
                module_id=SHARED_GNN_MODULE_ID,
                data={
                    Columns.OBS: _graph_observation(node_count, step),
                    Columns.NEXT_OBS: _graph_observation(
                        next_node_count,
                        step + 1,
                    ),
                    Columns.ACTIONS: np.int64((sequence + step) % 3),
                    Columns.REWARDS: np.float32(1.0),
                    Columns.TERMINATEDS: step == episode_length - 1,
                    Columns.TRUNCATEDS: False,
                },
            )
        )
    payload = codec.encode(transitions)
    return EpisodeEnvelope(
        episode_id=f"{member_id}/graph-runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id=member_id,
        runner_id="graph-runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({SHARED_GNN_MODULE_ID: 0}),
        env_steps=episode_length,
        agent_steps=episode_length,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def graph_codec() -> GraphEpisodeCodec:
    return GraphEpisodeCodec(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
    )


def profiled_call(
    function: Callable[[], ResultT],
    profile_path: Path | None,
) -> ResultT:
    """Run one driver-local operation and optionally publish a cProfile file."""

    if profile_path is None:
        return function()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    result = profiler.runcall(function)
    profiler.dump_stats(profile_path)
    return result


def benchmark_document(
    benchmark: str,
    parameters: Mapping[str, Any],
    results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": benchmark,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(),
        "parameters": json_ready(parameters),
        "results": json_ready(results),
    }


def publish_document(document: Mapping[str, Any], output: Path | None) -> None:
    """Print a report and atomically replace an optional artifact path."""

    rendered = json.dumps(
        json_ready(document),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    print(rendered)


def runtime_invariant_gates(
    report: Mapping[str, Any],
    *,
    replay_capacity_transitions: int,
    replay_capacity_bytes: int,
) -> dict[str, bool]:
    """Evaluate deterministic bounds without asserting noisy throughput values."""

    controller = _mapping(report, "controller")
    rollout = _mapping(report, "rollout")
    authoritative = _mapping(report, "authoritative_replay")
    fast_replay = _mapping(report, "fast_replay")
    batching = _mapping(report, "batching")
    learner = _mapping(report, "learner")
    gates = {
        "pending_rpcs_bounded": _number(
            controller,
            "pending_rpc_high_watermark",
        )
        <= _number(controller, "pending_rpc_bound"),
        "batch_queue_bounded": _number(batching, "queue_high_watermark")
        <= _number(batching, "queue_capacity"),
        "authoritative_transitions_bounded": _number(
            authoritative,
            "total_transitions",
        )
        <= replay_capacity_transitions,
        "authoritative_bytes_bounded": _number(
            authoritative,
            "total_estimated_bytes",
        )
        <= replay_capacity_bytes,
        "fast_replay_transitions_bounded": _number(
            fast_replay,
            "total_transitions",
        )
        <= replay_capacity_transitions,
        "fast_replay_bytes_bounded": _number(
            fast_replay,
            "total_estimated_bytes",
        )
        <= replay_capacity_bytes,
        "rollout_failures_absent": sum(
            _number(rollout, name) for name in ("sample_failures", "commit_failures")
        )
        == 0,
        "batch_failures_absent": _number(batching, "producer_failures") == 0,
        "data_wait_measured": (
            _number(batching, "data_wait_calls") > 0
            and 0 <= _number(learner, "data_wait_fraction") <= 1
            and 0 <= _number(learner, "batch_queue_empty_fraction") <= 1
        ),
        "batch_build_measured": (
            _number(batching, "batch_builds") > 0
            and _number(batching, "batch_build_ms_p95") >= 0
        ),
        "learner_update_measured": (
            _number(learner, "learner_updates") > 0
            and _number(learner, "update_time_ms_p95") > 0
        ),
    }
    gates["all_passed"] = all(gates.values())
    return gates


def require_gates(gates: Mapping[str, bool]) -> None:
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise PerformanceGateError(
            f"performance invariant gates failed: {', '.join(failed)}"
        )


def bottleneck_indicator(report: Mapping[str, Any]) -> str:
    """Classify measured pressure; profiles remain the source of root cause."""

    rollout = _mapping(report, "rollout")
    batching = _mapping(report, "batching")
    learner = _mapping(report, "learner")
    if (
        _number(learner, "data_wait_fraction") >= 0.05
        or _number(learner, "batch_queue_empty_fraction") >= 0.05
    ):
        return "batch_supply"
    if _number(rollout, "backpressure_fraction") >= 0.05:
        return "rollout_or_replay_ingest"
    update_p95 = _number(learner, "update_time_ms_p95")
    build_p95 = _number(batching, "batch_build_ms_p95")
    if update_p95 > 2 * max(build_p95, 1e-9):
        return "learner_update"
    if build_p95 > 2 * max(update_p95, 1e-9):
        return "batch_build"
    return "balanced_or_profile_required"


def json_ready(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_ready(nested) for key, nested in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ray": ray.__version__,
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }


def _graph_observation(node_count: int, step: int) -> dict[str, Any]:
    nodes = np.zeros((node_count, GRAPH_NODE_FEATURE_DIM), dtype=np.float32)
    nodes[:, 0] = np.linspace(-1.0, 1.0, node_count, dtype=np.float32)
    nodes[:, 1] = np.float32((step % 17) / 17)
    nodes[0, 2] = 1.0
    nodes[:, 3] = np.float32(1.0)
    edge_count = max(2 * (node_count - 1), 0)
    edges = np.zeros((2, edge_count), dtype=np.int64)
    edge_features = np.zeros(
        (edge_count, GRAPH_EDGE_FEATURE_DIM),
        dtype=np.float32,
    )
    offset = 0
    for node in range(1, node_count):
        edges[:, offset] = (0, node)
        edge_features[offset, 0] = nodes[node, 0]
        offset += 1
        edges[:, offset] = (node, 0)
        edge_features[offset, 0] = -nodes[node, 0]
        offset += 1
    return {
        "node_features": nodes,
        "edge_index": edges,
        "edge_features": edge_features,
        "controlled_node": np.int64(0),
    }


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise PerformanceGateError(f"runtime report has no {key!r} mapping")
    return value


def _number(parent: Mapping[str, Any], key: str) -> float:
    value = parent.get(key)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise PerformanceGateError(f"runtime metric {key!r} is not finite")
    return float(value)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None
