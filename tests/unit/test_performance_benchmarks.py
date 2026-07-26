from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest

from benchmarks.common import (
    PerformanceGateError,
    graph_codec,
    make_flat_episode,
    make_graph_episode,
    require_gates,
    runtime_invariant_gates,
)
from benchmarks.end_to_end_throughput import (
    _profile_path,
    _required_cluster_resources,
    _selected_modes,
    _validate_args,
    build_runtime_config,
    build_sac_config,
)
from rllib_async.protocols import FlatEpisodeCodec


def make_report() -> dict[str, object]:
    return {
        "controller": {
            "pending_rpc_high_watermark": 4,
            "pending_rpc_bound": 5,
        },
        "rollout": {
            "sample_failures": 0,
            "commit_failures": 0,
            "backpressure_fraction": 0.0,
        },
        "authoritative_replay": {
            "total_transitions": 64,
            "total_estimated_bytes": 10_000,
        },
        "fast_replay": {
            "total_transitions": 64,
            "total_estimated_bytes": 10_000,
        },
        "batching": {
            "queue_high_watermark": 2,
            "queue_capacity": 2,
            "producer_failures": 0,
            "data_wait_calls": 4,
            "batch_builds": 4,
            "batch_build_ms_p95": 1.5,
        },
        "learner": {
            "learner_updates": 4,
            "data_wait_fraction": 0.1,
            "batch_queue_empty_fraction": 0.25,
            "update_time_ms_p95": 3.0,
        },
    }


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "mode": "all",
        "members": 1,
        "runner_count": 4,
        "episode_length": 32,
        "batch_size": 128,
        "training_intensity": 1.0,
        "warmup_timesteps": 2_000,
        "measure_timesteps": 20_000,
        "queue_capacity": 4,
        "learner_updates_per_tick": 4,
        "replay_capacity_transitions": 100_000,
        "replay_capacity_bytes": 512 * 1024 * 1024,
        "num_gpus_per_learner": 0,
        "report_interval_s": 0.5,
        "max_duration_s": 1_800.0,
        "seed": 20260726,
        "ray_address": None,
        "storage_path": Path("/tmp/phase11-test"),
        "profile_dir": Path("/tmp/phase11-test/profiles"),
        "output": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_benchmark_episode_builders_satisfy_codec_contracts() -> None:
    flat_codec = FlatEpisodeCodec()
    flat = make_flat_episode(3, 8, flat_codec)
    flat_codec.validate(flat)

    nested_codec = graph_codec()
    nested = make_graph_episode(5, 8, nested_codec)
    nested_codec.validate(nested)

    assert flat.env_steps == flat.agent_steps == 8
    assert nested.env_steps == nested.agent_steps == 8
    assert nested_codec.module_ids(nested) == ("shared_graph",)


def test_runtime_invariant_gates_accept_only_bounded_measured_reports() -> None:
    report = make_report()
    gates = runtime_invariant_gates(
        report,
        replay_capacity_transitions=64,
        replay_capacity_bytes=10_000,
    )

    assert gates["all_passed"]
    require_gates(gates)

    unbounded = copy.deepcopy(report)
    unbounded["batching"]["queue_high_watermark"] = 3
    failed = runtime_invariant_gates(
        unbounded,
        replay_capacity_transitions=64,
        replay_capacity_bytes=10_000,
    )
    assert not failed["batch_queue_bounded"]
    assert not failed["all_passed"]
    with pytest.raises(PerformanceGateError, match="batch_queue_bounded"):
        require_gates(failed)


def test_end_to_end_modes_preserve_parity_and_reject_fake_stock_population() -> None:
    args = make_args()
    stock = build_sac_config(args, stock=True, seed=args.seed)
    custom = build_sac_config(args, stock=False, seed=args.seed)

    assert _selected_modes(args) == ("stock", "direct", "queued")
    assert stock.num_env_runners == args.runner_count
    assert custom.num_env_runners == 0
    for name in (
        "env_config",
        "num_gpus_per_learner",
        "train_batch_size_per_learner",
        "training_intensity",
        "num_steps_sampled_before_learning_starts",
        "policy_model_config",
        "q_model_config",
        "target_network_update_freq",
        "twin_q",
        "seed",
    ):
        assert getattr(stock, name) == getattr(custom, name)

    direct = build_runtime_config(args, mode="direct", member_index=0)
    queued = build_runtime_config(args, mode="queued", member_index=0)
    assert direct.batch_queue_capacity == 0
    assert queued.batch_queue_capacity == args.queue_capacity

    population = make_args(members=2)
    assert _selected_modes(population) == ("direct", "queued")
    assert _required_cluster_resources(args) == (6.0, 0.0)
    assert _required_cluster_resources(
        make_args(members=2, runner_count=16, num_gpus_per_learner=1)
    ) == (37.0, 2.0)
    with pytest.raises(ValueError, match="no shared-replay member topology"):
        _validate_args(make_args(members=2, mode="stock"))


def test_end_to_end_profile_path_identifies_the_complete_matrix_point() -> None:
    first = make_args(
        members=1,
        runner_count=4,
        episode_length=32,
        batch_size=128,
        training_intensity=1.0,
    )
    second = make_args(
        members=1,
        runner_count=4,
        episode_length=512,
        batch_size=512,
        training_intensity=4.0,
    )

    assert _profile_path(first, "direct") != _profile_path(second, "direct")
    assert _profile_path(first, "direct") != _profile_path(first, "queued")
    assert _profile_path(make_args(profile_dir=None), "direct") is None
