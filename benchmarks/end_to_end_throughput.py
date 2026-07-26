"""Compare stock RLlib SAC with direct and queued asynchronous runtime modes."""

from __future__ import annotations

import argparse
import math
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import ray
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED_LIFETIME,
)

from benchmarks.common import (
    benchmark_document,
    bottleneck_indicator,
    profiled_call,
    publish_document,
    require_gates,
    runtime_invariant_gates,
)
from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    PopulationLauncher,
    PopulationMemberSpec,
    SingleMemberAsyncSAC,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("stock", "direct", "queued", "all"),
        default="all",
    )
    parser.add_argument("--members", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--runner-count",
        type=int,
        choices=(1, 4, 8, 16),
        default=4,
    )
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-intensity", type=float, default=1.0)
    parser.add_argument("--warmup-timesteps", type=int, default=2_000)
    parser.add_argument("--measure-timesteps", type=int, default=20_000)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--learner-updates-per-tick", type=int, default=4)
    parser.add_argument("--replay-capacity-transitions", type=int, default=100_000)
    parser.add_argument(
        "--replay-capacity-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--num-gpus-per-learner", type=int, choices=(0, 1), default=0)
    parser.add_argument("--report-interval-s", type=float, default=0.5)
    parser.add_argument("--max-duration-s", type=float, default=1_800)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--ray-address")
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path("/tmp/rllib-async-runtime-benchmarks"),
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_sac_config(
    args: argparse.Namespace,
    *,
    stock: bool,
    seed: int,
) -> SACConfig:
    """Build equivalent stock/custom SAC settings around one cheap environment."""

    return (
        SACConfig()
        .environment(
            SyntheticThroughputEnv,
            env_config={"episode_length": args.episode_length},
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=args.runner_count if stock else 0,
            create_local_env_runner=not stock,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            episodes_to_numpy=True,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
        .training(
            num_steps_sampled_before_learning_starts=args.batch_size * 2,
            policy_model_config={"fcnet_hiddens": [64, 64]},
            q_model_config={"fcnet_hiddens": [64, 64]},
            target_network_update_freq=1,
            train_batch_size_per_learner=args.batch_size,
            training_intensity=args.training_intensity,
            twin_q=True,
        )
        .reporting(
            min_time_s_per_iteration=0,
            min_train_timesteps_per_iteration=0,
            min_sample_timesteps_per_iteration=max(
                args.episode_length * args.runner_count,
                1,
            ),
        )
        .debugging(seed=seed)
    )


def build_runtime_config(
    args: argparse.Namespace,
    *,
    mode: str,
    member_index: int,
) -> AsyncSACRuntimeConfig:
    queue_capacity = 0 if mode == "direct" else args.queue_capacity
    high_watermark = max(args.runner_count * 2, 2)
    return AsyncSACRuntimeConfig(
        member_id=f"member-{member_index}",
        runner_count=args.runner_count,
        max_episode_steps=args.episode_length,
        replay_capacity_transitions=args.replay_capacity_transitions,
        replay_capacity_bytes=args.replay_capacity_bytes,
        replay_journal_capacity=4_096,
        pending_commit_high_watermark=high_watermark,
        pending_commit_low_watermark=args.runner_count,
        batch_size=args.batch_size,
        batch_queue_capacity=queue_capacity,
        learner_updates_per_tick=args.learner_updates_per_tick,
        publication_interval_updates=10,
        evaluation_interval_env_steps=0,
        evaluation_num_episodes=0,
        report_interval_s=args.report_interval_s,
        num_gpus_per_learner=args.num_gpus_per_learner,
        seed=args.seed + member_index,
    )


def run_stock(args: argparse.Namespace) -> dict[str, Any]:
    config = build_sac_config(args, stock=True, seed=args.seed)
    algorithm = config.build_algo()
    deadline = time.monotonic() + args.max_duration_s
    try:
        result: Mapping[str, Any] = {}
        sampled_steps = 0
        while sampled_steps < args.warmup_timesteps:
            _require_before_deadline(deadline)
            result = algorithm.train()
            sampled_steps = _stock_metric(
                result,
                NUM_ENV_STEPS_SAMPLED_LIFETIME,
            )
        base_steps = sampled_steps
        base_trained = (
            _stock_metric(
                result,
                NUM_ENV_STEPS_TRAINED_LIFETIME,
                required=False,
            )
            if result
            else 0
        )
        started = time.perf_counter()
        while sampled_steps - base_steps < args.measure_timesteps:
            _require_before_deadline(deadline)
            result = algorithm.train()
            sampled_steps = _stock_metric(
                result,
                NUM_ENV_STEPS_SAMPLED_LIFETIME,
            )
        duration_s = time.perf_counter() - started
        trained_steps = _stock_metric(
            result,
            NUM_ENV_STEPS_TRAINED_LIFETIME,
            required=False,
        )
        measured_steps = sampled_steps - base_steps
        gates = {
            "measurement_completed": measured_steps >= args.measure_timesteps,
            "finite_throughput": (
                duration_s > 0 and math.isfinite(measured_steps / duration_s)
            ),
        }
        gates["all_passed"] = all(gates.values())
        require_gates(gates)
        return {
            "mode": "stock",
            "members": 1,
            "duration_s": duration_s,
            "measured_env_steps": measured_steps,
            "env_steps_per_s": measured_steps / duration_s,
            "measured_trained_steps": max(trained_steps - base_trained, 0),
            "gates": gates,
        }
    finally:
        algorithm.stop()


def run_single_runtime(
    args: argparse.Namespace,
    mode: str,
) -> dict[str, Any]:
    sac_config = build_sac_config(args, stock=False, seed=args.seed)
    runtime_config = build_runtime_config(
        args,
        mode=mode,
        member_index=0,
    )
    runtime = SingleMemberAsyncSAC(sac_config, runtime_config)
    report: Mapping[str, Any] | None = None
    deadline = time.monotonic() + args.max_duration_s
    try:
        runtime.start()
        sampled_steps = 0
        while sampled_steps < args.warmup_timesteps:
            _require_before_deadline(deadline)
            report = runtime.run_for(args.report_interval_s)
            sampled_steps = int(report["rollout"]["env_steps"])
        base_steps = sampled_steps
        base_updates = (
            int(report["learner"]["learner_updates"]) if report is not None else 0
        )
        started = time.perf_counter()
        while sampled_steps - base_steps < args.measure_timesteps:
            _require_before_deadline(deadline)
            report = runtime.run_for(args.report_interval_s)
            sampled_steps = int(report["rollout"]["env_steps"])
        duration_s = time.perf_counter() - started
        assert report is not None
        measured_steps = sampled_steps - base_steps
        measured_updates = int(report["learner"]["learner_updates"]) - base_updates
        gates = runtime_invariant_gates(
            report,
            replay_capacity_transitions=args.replay_capacity_transitions,
            replay_capacity_bytes=args.replay_capacity_bytes,
        )
        gates["measurement_completed"] = measured_steps >= args.measure_timesteps
        gates["all_passed"] = all(
            passed for name, passed in gates.items() if name != "all_passed"
        )
        require_gates(gates)
        return {
            "mode": mode,
            "members": 1,
            "duration_s": duration_s,
            "measured_env_steps": measured_steps,
            "env_steps_per_s": measured_steps / duration_s,
            "measured_learner_updates": measured_updates,
            "learner_updates_per_s": measured_updates / duration_s,
            "bottleneck_indicator": bottleneck_indicator(report),
            "gates": gates,
            "final_report": report,
        }
    finally:
        runtime.stop(timeout_s=30)


def run_population(
    args: argparse.Namespace,
    mode: str,
) -> dict[str, Any]:
    members = tuple(
        PopulationMemberSpec(
            build_sac_config(
                args,
                stock=False,
                seed=args.seed + member_index,
            ),
            build_runtime_config(
                args,
                mode=mode,
                member_index=member_index,
            ),
        )
        for member_index in range(2)
    )
    run_name = f"phase11-{mode}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    with PopulationLauncher(members) as launcher:
        results = launcher.fit(
            run_config=RunConfig(
                name=run_name,
                storage_path=str(args.storage_path.resolve()),
                stop={
                    "timesteps_total": (args.warmup_timesteps + args.measure_timesteps),
                    "time_total_s": args.max_duration_s,
                },
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
                verbose=0,
            )
        )
        duration_s = time.perf_counter() - started
        member_results = []
        for result in results:
            report = result.metrics
            gates = runtime_invariant_gates(
                report,
                replay_capacity_transitions=args.replay_capacity_transitions,
                replay_capacity_bytes=args.replay_capacity_bytes,
            )
            gates["measurement_completed"] = (
                report["rollout"]["env_steps"]
                >= args.warmup_timesteps + args.measure_timesteps
            )
            gates["all_passed"] = all(
                passed for name, passed in gates.items() if name != "all_passed"
            )
            require_gates(gates)
            member_results.append(
                {
                    "member_id": report["controller"]["member_id"],
                    "env_steps": report["rollout"]["env_steps"],
                    "env_steps_per_s": report["rollout"]["env_steps_per_s"],
                    "learner_updates": report["learner"]["learner_updates"],
                    "learner_updates_per_s": report["learner"]["updates_per_s"],
                    "accelerator_ids": report["learner"]["accelerator_ids"],
                    "bottleneck_indicator": bottleneck_indicator(report),
                    "gates": gates,
                    "final_report": report,
                }
            )
    aggregate_steps = sum(int(item["env_steps"]) for item in member_results)
    expected_member_ids = {"member-0", "member-1"}
    accelerator_ids = [tuple(item["accelerator_ids"]) for item in member_results]
    intervals = [
        (
            float(item["final_report"]["controller"]["started_at_monotonic"]),
            float(item["final_report"]["controller"]["reported_at_monotonic"]),
        )
        for item in member_results
    ]
    gates = {
        "two_members_completed": len(member_results) == 2,
        "finite_aggregate_throughput": (
            duration_s > 0 and math.isfinite(aggregate_steps / duration_s)
        ),
        "shared_replay_visible": all(
            set(item["final_report"]["fast_replay"]["active_producer_episode_counts"])
            == expected_member_ids
            for item in member_results
        ),
        "member_execution_overlapped": (
            len(intervals) == 2
            and max(start for start, _ in intervals) < min(end for _, end in intervals)
        ),
        "accelerator_assignment_valid": (
            args.num_gpus_per_learner == 0
            or (
                len(accelerator_ids) == 2
                and all(len(ids) == 1 for ids in accelerator_ids)
                and accelerator_ids[0] != accelerator_ids[1]
            )
        ),
    }
    gates["all_passed"] = all(gates.values())
    require_gates(gates)
    return {
        "mode": mode,
        "members": 2,
        "duration_s": duration_s,
        "aggregate_env_steps": aggregate_steps,
        "aggregate_env_steps_per_s": aggregate_steps / duration_s,
        "member_results": member_results,
        "gates": gates,
        "ray_storage": args.storage_path / run_name,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    modes = _selected_modes(args)
    ray.init(
        address=args.ray_address,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
        cluster_resources = dict(ray.cluster_resources())
        required_cpus, required_gpus = _required_cluster_resources(args)
        available_cpus = float(cluster_resources.get("CPU", 0.0))
        available_gpus = float(cluster_resources.get("GPU", 0.0))
        if available_cpus < required_cpus:
            raise RuntimeError(
                f"benchmark requires {required_cpus:g} Ray CPU slots, "
                f"cluster reports {available_cpus:g}"
            )
        if available_gpus < required_gpus:
            raise RuntimeError(
                f"benchmark requires {required_gpus:g} Ray GPU slots, "
                f"cluster reports {available_gpus:g}"
            )
        results = []
        for mode in modes:
            profile_path = _profile_path(args, mode)
            result = profiled_call(
                _bind_mode(args, mode),
                profile_path,
            )
            result["profile"] = profile_path
            results.append(result)
    finally:
        ray.shutdown()
    publish_document(
        benchmark_document(
            "end_to_end_throughput",
            {
                **vars(args),
                "profile_dir": args.profile_dir,
                "output": args.output,
                "ray_cluster_resources": cluster_resources,
            },
            results,
        ),
        args.output,
    )


def _selected_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.mode != "all":
        return (args.mode,)
    if args.members == 1:
        return ("stock", "direct", "queued")
    return ("direct", "queued")


def _bind_mode(
    args: argparse.Namespace,
    mode: str,
) -> Callable[[], dict[str, Any]]:
    """Bind one selected mode without duplicating profiling/reporting code."""

    def run() -> dict[str, Any]:
        if mode == "stock":
            return run_stock(args)
        if args.members == 1:
            return run_single_runtime(args, mode)
        return run_population(args, mode)

    return run


def _profile_path(args: argparse.Namespace, mode: str) -> Path | None:
    if args.profile_dir is None:
        return None
    intensity_label = str(args.training_intensity).replace(".", "p")
    return args.profile_dir / (
        f"end_to_end_{args.members}m_{args.runner_count}r_"
        f"e{args.episode_length}_b{args.batch_size}_"
        f"u{intensity_label}_{mode}.prof"
    )


def _required_cluster_resources(
    args: argparse.Namespace,
) -> tuple[float, float]:
    if args.members == 1:
        required_cpus = args.runner_count + 2
    else:
        required_cpus = 2 * (args.runner_count + 2) + 1
    required_gpus = args.members * args.num_gpus_per_learner
    return float(required_cpus), float(required_gpus)


def _validate_args(args: argparse.Namespace) -> None:
    if args.members == 2 and args.mode == "stock":
        raise ValueError(
            "stock SAC has no shared-replay member topology; "
            "use --members 1 for the stock baseline"
        )
    for name in (
        "episode_length",
        "batch_size",
        "measure_timesteps",
        "queue_capacity",
        "learner_updates_per_tick",
        "replay_capacity_transitions",
        "replay_capacity_bytes",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_timesteps < 0:
        raise ValueError("--warmup-timesteps must be non-negative")
    if not math.isfinite(args.training_intensity) or args.training_intensity <= 0:
        raise ValueError("--training-intensity must be finite and positive")
    if not math.isfinite(args.report_interval_s) or args.report_interval_s <= 0:
        raise ValueError("--report-interval-s must be finite and positive")
    if not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0:
        raise ValueError("--max-duration-s must be finite and positive")


def _stock_metric(
    result: Mapping[str, Any],
    key: str,
    *,
    required: bool = True,
) -> int:
    candidates = [result]
    for container_name in ("env_runners", "learners"):
        container = result.get(container_name)
        if isinstance(container, Mapping):
            candidates.append(container)
    for candidate in candidates:
        value = candidate.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return int(value)
    if required:
        raise RuntimeError(f"stock RLlib result does not expose {key!r}")
    return 0


def _require_before_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("benchmark exceeded --max-duration-s")


if __name__ == "__main__":
    main()
