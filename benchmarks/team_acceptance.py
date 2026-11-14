"""Fixed-data mixed/recurrent measurements and offline hidden-cue comparisons.

This deliberately uses the existing learner host behind one Ray actor. Dataset
collection and evaluation are synchronous, so world-step budgets are exact.
Offline training gives each module an explicit valid-position budget; replaying
that dataset does not count as new environment interactions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig

from benchmarks.common import (
    benchmark_document,
    profiled_call,
    publish_document,
    require_gates,
)
from rllib_async.examples.temporal_hierarchy import (
    build_memory_configs,
    build_temporal_hierarchy_configs,
)
from rllib_async.protocols import ReplayDelta, WeightsDescriptor
from rllib_async.protocols.episodes import RawMultiModuleEpisodeCodec
from rllib_async.replay import ReplayActor
from rllib_async.replay.prioritized import PrioritizedReplayConfig
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.rollout import MultiModuleEpisodeRunner
from rllib_async.runtime.config import AsyncRuntimeConfig
from rllib_async.runtime.mixed_learner_host import MixedLearnerHost
from rllib_async.runtime.team import prepare_team_config

TRAINING_SEEDS = (20261114, 20261115, 20261116, 20261117, 20261118)
EVALUATION_SEEDS = (910001, 910002, 910003, 910004, 910005)


def process_measurements() -> dict[str, Any]:
    """Read actual Linux counters outside the production training event loop."""

    memory = {}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                memory[key] = int(value.split()[0]) * 1024
    interfaces = {}
    network = Path("/proc/net/dev")
    if network.exists():
        for line in network.read_text().splitlines()[2:]:
            name, values = line.split(":", 1)
            columns = values.split()
            interfaces[name.strip()] = {
                "rx_bytes": int(columns[0]), "tx_bytes": int(columns[8])
            }
    return {
        "pid": os.getpid(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "rss_bytes": memory.get("VmRSS"),
        "process_peak_rss_bytes": memory.get("VmHWM"),
        "interfaces": interfaces,
        "network_scope": "network namespace counters; includes unrelated traffic",
    }


def interface_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        interface: {
            key: after[interface][key] - before[interface][key]
            for key in ("rx_bytes", "tx_bytes")
        }
        for interface in before.keys() & after.keys()
    }


def dataset_digest(episodes) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.episode_id.encode())
        for module_id, rows in episode.payload.encoded_module_transitions:
            digest.update(module_id.encode())
            for row in rows:
                digest.update(row.encoded_data)
    return digest.hexdigest()


class _NoReplayRequests:
    def __getattr__(self, name):
        raise AssertionError(f"local training attempted a replay RPC: {name}")


@ray.remote(max_concurrency=1)
class AcceptanceLearner:
    def __init__(self, modules, spaces, source, sequences, per_configs, args, seed):
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.before_load = process_measurements()
        self.host = MixedLearnerHost(
            modules,
            spaces,
            source,
            RawMultiModuleEpisodeCodec(),
            member_id="acceptance",
            publication_interval_updates=16,
            batch_size=args.batch_size,
            batch_queue_capacity=0,
            batch_seed=seed,
            replay_sync_max_bytes=16 * 1024 * 1024,
            sequence_configs=sequences,
            per_configs=per_configs,
            validate_startup=True,
        )
        self.after_load = process_measurements()
        self.host._fast_replay.wait_for_idle(timeout=30)
        self.host.start()

    def train(self, positions: int, timeout_s: float, profile_path: str | None):
        host = self.host
        replay = host._fast_replay
        snapshot = replay.get_snapshot()
        before_digest = dataset_digest(snapshot.episodes)
        references = {episode.episode_id: episode for episode in snapshot.episodes}
        shared = all(
            episode is references[episode_id]
            for module_id in host._module_ids
            for episode_id, episode in (
                replay.get_sequence_view(module_id).episodes.items()
            )
        )
        before_empty = replay.get_stats()
        empty = ReplayDelta(replay.cursor, replay.cursor, (), False, False)
        for _ in range(100):
            replay.apply_delta(empty)
        after_empty = replay.get_stats()
        devices = {
            module_id: sorted(
                {
                    str(parameter.device)
                    for parameter in (
                        adapter._learner_group._learner.module[module_id].parameters()
                    )
                }
            )
            for module_id, adapter in host._adapters.items()
        }
        gpu = bool(ray.get_runtime_context().get_accelerator_ids().get("GPU"))
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        before = process_measurements()
        before_updates = host._learner_updates
        before_build = host._batch_build_s
        for counter in host._module_counters.values():
            # This is an explicitly offline replay budget, not new sampling credit.
            counter["budget"] = float(positions)
        source = host._replay_actor
        host._replay_actor = _NoReplayRequests()
        started = time.perf_counter()
        deadline = time.monotonic() + timeout_s

        def consume():
            while any(float(c["budget"]) >= 1 for c in host._module_counters.values()):
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixed-data learner budget did not finish")
                updates, _ = host._consume_ready(1)
                if not updates:
                    raise RuntimeError("a module has no eligible data for its budget")

        try:
            profiled_call(consume, Path(profile_path) if profile_path else None)
        finally:
            host._replay_actor = source
        if gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        after = process_measurements()
        publish_started = time.perf_counter()
        host._maybe_publish(force=True)
        publication_s = time.perf_counter() - publish_started
        stats = host.get_stats()
        counters = {mid: dict(values) for mid, values in stats.module_counters}
        gates = {
            "exact_valid_position_budget": all(
                counter["learned_positions"] == positions
                for counter in counters.values()
            ),
            "one_shared_payload_view": shared,
            "payload_unchanged": before_digest
            == dataset_digest(replay.get_snapshot().episodes),
            "no_replay_rpc_during_local_training": True,
            "empty_delta_no_rebuild": (
                before_empty.completed_rebuilds == after_empty.completed_rebuilds
                and before_empty.active_cursor == after_empty.active_cursor
            ),
            "actual_parameter_devices": all(
                values
                and all(value.startswith("cuda:" if gpu else "cpu") for value in values)
                for values in devices.values()
            ),
        }
        gates["all_passed"] = all(gates.values())
        require_gates(gates)
        return {
            "elapsed_s": elapsed,
            "optimizer_updates": host._learner_updates - before_updates,
            "valid_learned_positions": host._learned_positions,
            "valid_learned_positions_per_s": host._learned_positions / elapsed,
            "sample_decode_collate_s": host._batch_build_s - before_build,
            "final_publication_s": publication_s,
            "module_counters": counters,
            "parameter_devices": devices,
            "ray_accelerator_ids": ray.get_runtime_context().get_accelerator_ids(),
            "before_load": self.before_load,
            "after_load": self.after_load,
            "before_training": before,
            "after_training": after,
            "training_interface_byte_deltas": interface_delta(
                before["interfaces"], after["interfaces"]
            ),
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if gpu else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if gpu else None
            ),
            "retained_payload_estimate_bytes": stats.fast_replay.total_estimated_bytes,
            "profile": profile_path,
            "gates": gates,
        }

    def weights(self):
        return self.host.get_published_weights()

    def close(self):
        self.host.stop(timeout_s=30)


def build_case(args, seed: int, memory: str, *, behavior: bool = False):
    if args.task == "memory":
        rollout, modules, _ = build_memory_configs(
            graph=args.graph,
            algorithm=args.algorithm,
            recurrent=memory != "stateless",
            episode_length=args.episode_length,
            seed=seed,
        )
    else:
        rollout, modules, _ = build_temporal_hierarchy_configs(
            recurrent=memory != "stateless",
            vector_worker=True,
            episode_length=args.episode_length,
            seed=seed,
        )
    for config in modules.values():
        config.learners(num_gpus_per_learner=0 if behavior else args.num_gpus)
        config.training(
            train_batch_size_per_learner=args.batch_size,
            n_step=args.n_step,
            target_network_update_freq=0,
        )
        if isinstance(config, DQNConfig) and not isinstance(config, SACConfig):
            config.training(
                num_atoms=(
                    51 if args.dqn_profile in {"c51", "rainbow"} and not behavior else 1
                ),
                v_min=-2.0,
                v_max=2.0,
                noisy=args.dqn_profile == "rainbow" and not behavior,
            )
            if behavior:
                config.training(epsilon=1.0)
            elif args.dqn_profile == "rainbow":
                config.training(epsilon=0.0)
    burn_in = (
        None
        if memory == "full-prefix"
        else args.burn_in if memory == "burn-in" else 0
    )
    sequences = {
        module_id: SequenceConfig(
            learning_length=args.learning_length,
            burn_in=burn_in,
        )
        for module_id in modules
    }
    runtime = AsyncRuntimeConfig(num_gpus_per_learner=0 if behavior else args.num_gpus)
    rollout, modules, spaces, _, sequences = prepare_team_config(
        rollout, modules, runtime, sequence_configs=sequences
    )
    return rollout, modules, spaces, sequences


def initial_weights(rollout, spaces, seed):
    torch.manual_seed(seed)
    spec = copy.deepcopy(
        rollout.get_multi_rl_module_spec(spaces=spaces, inference_only=True)
    )
    spec.inference_only = True
    for module_spec in spec.rl_module_specs.values():
        module_spec.inference_only = True
    module = spec.build()
    return WeightsDescriptor(
        "acceptance",
        dict.fromkeys(spaces, 0),
        0,
        time.monotonic(),
        {mid: module[mid].get_state(inference_only=True) for mid in spaces},
    )


def collect(
    rollout, spaces, weights, *, episodes: int, runner_id: str, episode_length: int
):
    codec = RawMultiModuleEpisodeCodec()
    runner = MultiModuleEpisodeRunner(
        rollout,
        codec,
        member_id="acceptance",
        runner_id=runner_id,
        runner_generation=0,
        max_episode_steps=2 * episode_length,
        module_ids=tuple(spaces),
        initial_weights=weights,
        worker_index=0,
    )
    try:
        return [runner.collect_episode(explore=True) for _ in range(episodes)]
    finally:
        runner.close()


def evaluate(rollout, spaces, weights, args):
    returns = []
    by_seed = {}
    for seed in EVALUATION_SEEDS:
        config = rollout.copy(copy_frozen=False).debugging(seed=seed)
        runner = MultiModuleEpisodeRunner(
            config,
            RawMultiModuleEpisodeCodec(),
            member_id="acceptance",
            runner_id=f"evaluation-{seed}",
            runner_generation=0,
            max_episode_steps=2 * args.episode_length,
            module_ids=tuple(spaces),
            initial_weights=weights,
            worker_index=0,
        )
        try:
            values = [
                runner.collect_episode(explore=False).metrics.episode_return
                for _ in range(args.evaluation_episodes)
            ]
            by_seed[str(seed)] = values
            returns.extend(values)
        finally:
            runner.close()
    return {"mean_return": float(np.mean(returns)), "returns_by_seed": by_seed}


def run_seed(args, seed):
    behavior, _, behavior_spaces, _ = build_case(args, seed, "stateless", behavior=True)
    data = collect(
        behavior,
        behavior_spaces,
        initial_weights(behavior, behavior_spaces, seed),
        episodes=args.world_steps // args.episode_length,
        runner_id=f"dataset-{seed}",
        episode_length=args.episode_length,
    )
    actual_world_steps = sum(result.metrics.world_steps for result in data)
    if actual_world_steps != args.world_steps:
        raise RuntimeError("collection did not satisfy its exact world-step budget")
    episodes = tuple(result.episode for result in data)
    capacity = sum(episode.agent_steps for episode in episodes)
    source = ReplayActor.options(num_cpus=0).remote(
        RawMultiModuleEpisodeCodec(),
        capacity_transitions=capacity,
        capacity_bytes=max(sum(episode.estimated_bytes for episode in episodes), 1),
        journal_capacity=len(episodes),
    )
    results = []
    try:
        before_ingest = process_measurements()
        for episode in episodes:
            acknowledgement = ray.get(source.commit_episode.remote(episode))
            if not acknowledgement.committed:
                raise RuntimeError("benchmark dataset episode was not committed")
        after_ingest = process_measurements()
        authoritative = ray.get(source.get_stats.remote())
        for memory in args.memories:
            rollout, modules, spaces, sequences = build_case(args, seed, memory)
            per_configs = (
                {mid: PrioritizedReplayConfig() for mid in modules}
                if args.sampling == "per"
                else {}
            )
            actor = AcceptanceLearner.options(
                num_cpus=1, num_gpus=args.num_gpus
            ).remote(modules, spaces, source, sequences, per_configs, args, seed)
            try:
                profile_path = (
                    str(args.profile_dir / f"{args.task}-{memory}-{seed}.prof")
                    if args.profile_dir
                    else None
                )
                result = ray.get(
                    actor.train.remote(
                        args.learned_positions, args.timeout_s, profile_path
                    ),
                    timeout=args.timeout_s + 60,
                )
                result.update(
                    {
                        "seed": seed,
                        "memory": memory,
                        "sampling": args.sampling,
                        "dataset_sha256": dataset_digest(episodes),
                        "collected_world_steps": actual_world_steps,
                        "authoritative": asdict(authoritative),
                        "ingest_driver_before": before_ingest,
                        "ingest_driver_after": after_ingest,
                        "ingest_interface_byte_deltas": interface_delta(
                            before_ingest["interfaces"], after_ingest["interfaces"]
                        ),
                    }
                )
                if args.command == "quality":
                    result["evaluation"] = evaluate(
                        rollout, spaces, ray.get(actor.weights.remote()), args
                    )
                results.append(result)
            finally:
                try:
                    ray.get(actor.close.remote(), timeout=30)
                finally:
                    ray.kill(actor, no_restart=True)
    finally:
        ray.kill(source, no_restart=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("components", "quality"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--task", choices=("memory", "hierarchy"),
            default="memory" if name == "quality" else "hierarchy",
        )
        command.add_argument("--algorithm", choices=("dqn", "sac"), default="dqn")
        command.add_argument("--graph", action="store_true")
        command.add_argument(
            "--memories", nargs="+", choices=("stateless", "full-prefix", "burn-in"),
            default=["stateless", "full-prefix"] if name == "quality" else ["full-prefix"],
        )
        command.add_argument(
            "--seeds", nargs="+", type=int,
            default=list(TRAINING_SEEDS) if name == "quality" else [TRAINING_SEEDS[0]],
        )
        command.add_argument(
            "--episode-length", type=int, default=5 if name == "quality" else 8
        )
        command.add_argument(
            "--world-steps", type=int, default=320 if name == "quality" else 32
        )
        command.add_argument(
            "--learned-positions", type=int, default=8192 if name == "quality" else 128
        )
        command.add_argument(
            "--batch-size", type=int, default=32 if name == "quality" else 8
        )
        command.add_argument("--learning-length", type=int, default=5)
        command.add_argument("--burn-in", type=int, default=2)
        command.add_argument("--n-step", type=int, default=3)
        command.add_argument(
            "--sampling", choices=("uniform", "per"), default="uniform"
        )
        command.add_argument(
            "--dqn-profile", choices=("baseline", "c51", "rainbow"), default="baseline"
        )
        command.add_argument("--num-gpus", type=int, choices=(0, 1), default=0)
        command.add_argument("--evaluation-episodes", type=int, default=20)
        command.add_argument("--timeout-s", type=float, default=600)
        command.add_argument("--profile-dir", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "quality":
            command.add_argument("--require-quality", action="store_true")
    args = parser.parse_args()
    for name in (
        "episode_length", "world_steps", "learned_positions", "batch_size",
        "learning_length", "n_step", "evaluation_episodes",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.world_steps % args.episode_length:
        parser.error("--world-steps must be divisible by --episode-length")
    if args.burn_in < 0 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("burn-in must be non-negative and timeout finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) & set(EVALUATION_SEEDS):
        parser.error("training seeds must be unique and disjoint from evaluation")
    if len(set(args.memories)) != len(args.memories):
        parser.error("memory modes must be unique")
    if args.command == "quality" and args.task != "memory":
        parser.error("offline quality comparison is defined only for the memory task")
    if args.dqn_profile == "rainbow" and (args.sampling != "per" or args.n_step < 2):
        parser.error("Rainbow requires --sampling per and --n-step at least 2")
    if args.task == "hierarchy" and (args.graph or args.algorithm != "dqn"):
        parser.error("hierarchy already has fixed mixed graph/vector DQN/SAC roles")
    if args.task == "memory" and args.algorithm == "sac" and args.dqn_profile != "baseline":
        parser.error("DQN profiles do not apply to the SAC memory task")
    return args


def quality_summary(results):
    by_memory = {}
    for memory in sorted({result["memory"] for result in results}):
        returns = {
            result["seed"]: result["evaluation"]["mean_return"]
            for result in results if result["memory"] == memory
        }
        by_memory[memory] = {
            "seed_returns": returns,
            "mean_return": float(np.mean(list(returns.values()))),
            "std_across_training_seeds": float(np.std(list(returns.values()), ddof=1))
            if len(returns) > 1 else None,
        }
    comparison = None
    if "stateless" in by_memory and "full-prefix" in by_memory:
        baseline = by_memory["stateless"]["seed_returns"]
        recurrent = by_memory["full-prefix"]["seed_returns"]
        paired = {
            seed: recurrent[seed] - baseline[seed]
            for seed in baseline.keys() & recurrent.keys()
        }
        gates = {
            "five_preregistered_seeds": set(paired) == set(TRAINING_SEEDS),
            "recurrent_return_at_least_0_6": by_memory["full-prefix"]["mean_return"] >= 0.6,
            "memoryless_return_within_0_3_of_zero": abs(by_memory["stateless"]["mean_return"]) <= 0.3,
            "paired_mean_gain_at_least_0_4": bool(paired) and float(np.mean(list(paired.values()))) >= 0.4,
            "at_least_four_positive_pairs": sum(value > 0 for value in paired.values()) >= 4,
        }
        gates["all_passed"] = all(gates.values())
        comparison = {
            "paired_return_gains": paired,
            "gates": gates,
            "interpretation": "registered criterion met" if gates["all_passed"]
            else "registered criterion not met; quality evidence is inconclusive",
        }
    return {"by_memory": by_memory, "memory_comparison": comparison}


def main():
    args = parse_args()
    torch.set_num_threads(1)
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    document = None
    try:
        if ray.cluster_resources().get("GPU", 0) < args.num_gpus:
            raise RuntimeError("the requested learner GPU is not available to Ray")
        results = []
        quality_plan = {
            "hypothesis": "hidden-cue full-prefix return exceeds memoryless return",
            "memoryless_expected_return": 0.0,
            "full_prefix_minimum_mean_return": 0.6,
            "maximum_absolute_memoryless_mean_return": 0.3,
            "minimum_paired_mean_gain": 0.4,
            "minimum_positive_seed_pairs": 4,
            "required_training_seeds": TRAINING_SEEDS,
            "evaluation_seed_streams": EVALUATION_SEEDS,
            "evaluation_episodes_per_stream": args.evaluation_episodes,
            "raw_world_steps_per_training_seed": args.world_steps,
            "valid_positions_per_module_and_variant": args.learned_positions,
            "target_schedule": "every optimizer update: target_network_update_freq=0",
        }
        document = benchmark_document("team_acceptance", vars(args), results)
        document["status"] = "running"
        document["quality_plan"] = quality_plan if args.command == "quality" else None
        publish_document(document, args.output)
        for seed in args.seeds:
            results.extend(run_seed(args, seed))
            document = benchmark_document("team_acceptance", vars(args), results)
            document["scope"] = "offline fixed-data learning; not online runtime throughput"
            document["evaluation_seeds"] = EVALUATION_SEEDS
            document["lock_sha256"] = hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
            document["quality_plan"] = quality_plan if args.command == "quality" else None
            document["status"] = "running"
            if args.command == "quality":
                document["quality"] = quality_summary(results)
            publish_document(document, args.output)
        document["status"] = "complete"
        publish_document(document, args.output)
        if args.command == "quality" and args.require_quality:
            comparison = document["quality"]["memory_comparison"]
            if comparison is None:
                raise ValueError("quality acceptance requires stateless and full-prefix runs")
            require_gates(comparison["gates"])
    except Exception as error:
        if document is not None:
            document["status"] = "failed"
            document["failure"] = str(error)
            publish_document(document, args.output)
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
