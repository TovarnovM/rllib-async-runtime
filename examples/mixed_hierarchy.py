"""DEMO-02/03 through the full shared-replay runtime.

Run inside the repository devcontainer, for example:
    python examples/mixed_hierarchy.py hierarchy --mode joint --vector-worker
    python examples/mixed_hierarchy.py memory --memory full-prefix --observation graph

The first step/update threshold starts completion checks: every trainable module
must have updated, and evaluation must have observed a new publication. In-flight
episodes finish during drain; the JSON report records actual totals and overshoot.
These functional runs do not claim equal-data learning-quality comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import ray

from rllib_async.examples.temporal_hierarchy import (
    build_memory_configs,
    build_temporal_hierarchy_configs,
)
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime import AsyncRuntimeConfig
from rllib_async.runtime.team import SingleMemberAsyncTeam


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    hierarchy = commands.add_parser("hierarchy", help="mixed DQN/DQN/SAC team")
    hierarchy.add_argument("--mode", choices=("frozen", "joint"), default="joint")
    hierarchy.add_argument("--vector-worker", action="store_true")
    hierarchy.add_argument("--fixed-duration", action="store_true")
    hierarchy.add_argument("--manager-period", type=_positive_int, default=3)
    memory = commands.add_parser(
        "memory", help="hidden-cue task and memoryless baseline"
    )
    memory.add_argument("--algorithm", choices=("dqn", "sac"), default="dqn")
    memory.add_argument("--observation", choices=("vector", "graph"), default="vector")
    for command in (hierarchy, memory):
        command.add_argument(
            "--memory",
            choices=("stateless", "full-prefix", "burn-in"),
            default="stateless" if command is hierarchy else "full-prefix",
        )
        command.add_argument(
            "--burn-in", type=int, default=2,
            help="prefix length for the explicitly approximate burn-in mode",
        )
        command.add_argument("--learning-length", type=_positive_int, default=4)
        command.add_argument("--n-step", type=_positive_int, default=1)
        command.add_argument("--batch-size", type=_positive_int, default=8)
        command.add_argument(
            "--steps", type=_positive_int, default=128,
            help="additional completed physical world ticks, including after restore",
        )
        command.add_argument(
            "--updates", type=_positive_int,
            help="alternative update threshold; functional checks still apply",
        )
        command.add_argument("--timeout-s", type=_positive_float, default=120.0)
        command.add_argument("--seed", type=int, default=20261114)
        command.add_argument("--num-gpus", type=int, choices=(0, 1), default=0)
        command.add_argument(
            "--episode-length", type=_positive_int,
            default=8 if command is hierarchy else 5,
        )
        command.add_argument("--validate-startup", action="store_true")
        command.add_argument(
            "--checkpoint", type=Path, help="save into a new directory"
        )
        command.add_argument(
            "--restore", type=Path, help="resume with the same settings"
        )
        command.add_argument(
            "--output", type=Path, help="also write the JSON report here"
        )
    args = parser.parse_args()
    if args.burn_in < 0:
        parser.error("--burn-in must be non-negative")
    return args


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _module_updates(report: dict[str, Any]) -> dict[str, int]:
    return {
        module_id: dict(counters)["optimizer_updates"]
        for module_id, counters in report["learner"]["module_counters"]
    }


def main() -> None:
    args = parse_args()
    recurrent = args.memory != "stateless"
    if args.task == "hierarchy":
        rollout, modules, _ = build_temporal_hierarchy_configs(
            episode_length=args.episode_length,
            manager_period=args.manager_period,
            variable_duration=not args.fixed_duration,
            vector_worker=args.vector_worker,
            recurrent=recurrent,
            seed=args.seed,
        )
        trainable = ("manager",) if args.mode == "frozen" else tuple(modules)
        max_episode_steps = 2 * args.episode_length
    else:
        rollout, modules, _ = build_memory_configs(
            graph=args.observation == "graph",
            algorithm=args.algorithm,
            episode_length=args.episode_length,
            recurrent=recurrent,
            seed=args.seed,
        )
        trainable = tuple(modules)
        max_episode_steps = args.episode_length
    for config in modules.values():
        config.learners(num_learners=0, num_gpus_per_learner=args.num_gpus)
        config.training(
            n_step=args.n_step,
            train_batch_size_per_learner=args.batch_size,
            training_intensity=1.0,
        )
    burn_in = (
        None if args.memory == "full-prefix"
        else args.burn_in if args.memory == "burn-in" else 0
    )
    sequences = {
        module_id: SequenceConfig(learning_length=args.learning_length, burn_in=burn_in)
        for module_id in modules
    }
    settings = AsyncRuntimeConfig(
        runner_count=1,
        max_episode_steps=max_episode_steps,
        batch_size=args.batch_size,
        learner_updates_per_tick=1,
        publication_interval_updates=1,
        pending_commit_high_watermark=1,
        pending_commit_low_watermark=0,
        evaluation_interval_env_steps=4 * max_episode_steps,
        evaluation_num_episodes=2,
        report_interval_s=0.25,
        num_cpus_per_runner=0,
        num_cpus_per_evaluation_runner=0,
        num_gpus_per_learner=args.num_gpus,
        validate_startup=args.validate_startup,
        seed=args.seed,
    )
    ray.init(num_cpus=2, include_dashboard=False)
    runtime = None
    try:
        if ray.cluster_resources().get("GPU", 0) < args.num_gpus:
            raise RuntimeError("the requested GPU is not available to Ray")
        runtime = SingleMemberAsyncTeam(
            rollout,
            modules,
            settings,
            trainable_modules=trainable,
            sequence_configs=sequences,
            checkpoint_dir=args.restore,
        )
        initial = runtime.get_report()
        initial_steps = initial["rollout"]["world_steps"]
        initial_updates = initial["learner"]["learner_updates"]
        initial_module_updates = _module_updates(initial)
        initial_evaluations = initial["evaluation"]["rounds_completed"]
        started = time.monotonic()
        deadline = started + args.timeout_s
        runtime.start()
        reason = "timeout"
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            report = runtime.run_for(min(0.25, remaining))
            threshold = None
            if report["rollout"]["world_steps"] - initial_steps >= args.steps:
                threshold = "world_steps"
            elif args.updates is not None and (
                report["learner"]["learner_updates"] - initial_updates >= args.updates
            ):
                threshold = "learner_updates"
            updates = _module_updates(report)
            all_modules_updated = all(
                updates[module_id] > initial_module_updates[module_id]
                for module_id in trainable
            )
            evaluated = (
                report["evaluation"]["rounds_completed"] > initial_evaluations
                and report["evaluation"]["latest_learner_updates"] > initial_updates
            )
            if threshold is not None and all_modules_updated and evaluated:
                reason = threshold
                break
        runtime.drain()
        if args.checkpoint is not None:
            args.checkpoint.mkdir(parents=True, exist_ok=True)
            runtime.save_checkpoint(args.checkpoint)
        report = runtime.get_report()
        report["demo"] = {
            "arguments": vars(args),
            "stop_reason": reason,
            "elapsed_s": time.monotonic() - started,
            "trainable_modules": trainable,
            "training_intensity": 1.0,
            "approximate_recurrent_state": args.memory == "burn-in",
            "additional_world_steps": report["rollout"]["world_steps"] - initial_steps,
            "additional_learner_updates": (
                report["learner"]["learner_updates"] - initial_updates
            ),
            "module_updates": _module_updates(report),
            "world_step_overshoot": max(
                report["rollout"]["world_steps"] - initial_steps - args.steps, 0
            ),
            "purpose": "functional run; no learning-quality threshold is asserted",
        }
        text = json.dumps(
            _json_value(report), indent=2, sort_keys=True, allow_nan=False
        )
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        if reason == "timeout":
            raise TimeoutError(
                "demo did not complete its budget, all module updates and evaluation"
            )
    finally:
        if runtime is not None:
            runtime.stop(graceful=False)
        ray.shutdown()


if __name__ == "__main__":
    main()
