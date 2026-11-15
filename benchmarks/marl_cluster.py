"""Prepare or run one monitored MPE2 GNN population on an existing Ray cluster.

Use inside the repository container. This launcher never starts/stops a cluster
or changes node configuration. Every PBT individual owns a complete policy team.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rllib_async.examples.marl_spread import build_spread_configs
from rllib_async.runtime import (
    AsyncRuntimeConfig,
    FloatMutation,
    PopulationMemberSpec,
    PopulationTrainable,
    SimplePBTConfig,
)


@ray.remote(num_cpus=0.05, max_restarts=0)
class NodeMonitor:
    """Own an external sampler process; collect compressed evidence after training."""

    def __init__(self) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="marl-monitor-"))
        self._path = self._directory / "node.jsonl.gz"
        self._log = (self._directory / "stderr.txt").open("w")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.cluster_monitor",
                "--output",
                str(self._path),
                "--parent-pid",
                str(os.getpid()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )

    def describe(self) -> dict:
        sample = {}
        deadline = time.monotonic() + 10
        while self._process.poll() is None and time.monotonic() < deadline:
            try:
                with gzip.open(self._path, "rt") as stream:
                    sample = json.loads(stream.readline())
                break
            except (OSError, EOFError, ValueError):
                time.sleep(0.05)
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "sampler_pid": self._process.pid,
            "sampler_exit_code": self._process.poll(),
            "sample_written": bool(sample),
            "ray_processes_visible": len(sample.get("processes", {})),
            "gpu_devices_visible": len(sample.get("gpu", {}).get("devices", [])),
            "network_interfaces": {
                name: {key: values.get(key) for key in ("physical", "speed_mbps")}
                for name, values in sample.get("network", {}).items()
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("ray", "torch", "mpe2")
            },
        }

    def stop(self) -> dict:
        self._process.terminate()
        try:
            self._process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3)
        self._log.close()
        result = {
            "exit_code": self._process.returncode,
            "stderr": (self._directory / "stderr.txt").read_text(),
            "data": self._path.read_bytes() if self._path.exists() else b"",
        }
        shutil.rmtree(self._directory)
        return result


def build_members(args: argparse.Namespace) -> tuple[PopulationMemberSpec, ...]:
    members = []
    for index in range(args.members):
        seed = args.seed + index
        rollout, modules = build_spread_configs(
            agents=args.agents,
            discrete_agents=(args.agents // 2 if args.mode == "mixed" else 0),
            max_cycles=args.episode_length,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            training_intensity=args.training_intensity,
            num_gpus=args.num_gpus_per_member,
            seed=seed,
        )
        scale = 0.8 + 0.4 * index / max(1, args.members - 1)
        modules["sac"].training(actor_lr=3e-4 * scale, critic_lr=3e-4 * scale)
        if "dqn" in modules:
            modules["dqn"].training(lr=3e-4 * scale)
        runtime = AsyncRuntimeConfig(
            member_id=f"team-{index:02d}",
            runner_count=args.workers,
            max_episode_steps=args.episode_length,
            batch_size=args.batch_size,
            pending_commit_high_watermark=2 * args.workers,
            pending_commit_low_watermark=args.workers,
            replay_capacity_transitions=args.replay_transitions,
            replay_capacity_bytes=args.replay_bytes,
            learner_updates_per_tick=8,
            publication_interval_updates=8,
            replay_sync_interval_s=0.05,
            evaluation_interval_env_steps=args.evaluation_interval,
            evaluation_num_episodes=args.evaluation_episodes,
            report_interval_s=args.report_interval_s,
            shutdown_timeout_s=120,
            num_cpus_per_runner=args.runner_cpus,
            num_cpus_per_learner=args.learner_cpus,
            num_cpus_per_replay=args.replay_cpus,
            num_cpus_per_evaluation_runner=1,
            num_gpus_per_learner=args.num_gpus_per_member,
            colocate_replay_and_learners=True,
            validate_startup=args.validate_startup,
            seed=seed,
        )
        members.append(PopulationMemberSpec.for_team(rollout, modules, runtime))
    return tuple(members)


def preflight(param_space: dict, nodes: list[dict], available: dict) -> dict:
    request = PopulationTrainable.default_resource_request(param_space)
    required = {}
    for bundle in request.bundles:
        for resource, amount in bundle.items():
            required[resource] = required.get(resource, 0) + amount
    required["CPU"] += 0.05 * len(nodes)
    core = request.bundles[1]
    deficits = {
        resource: amount - available.get(resource, 0)
        for resource, amount in required.items()
        if amount > available.get(resource, 0) + 1e-6
    }
    core_fits = any(
        all(node["Resources"].get(key, 0) >= value for key, value in core.items())
        for node in nodes
    )
    return {
        "required": required,
        "available": available,
        "core_bundle": core,
        "core_fits_one_node_total_resources": core_fits,
        "resource_deficits": deficits,
        "ready": not deficits and core_fits,
        "nodes": [
            {
                key: node.get(key)
                for key in ("NodeID", "NodeManagerAddress", "Resources")
            }
            for node in nodes
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("continuous", "mixed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--members", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--num-gpus-per-member", type=float, default=0.5)
    parser.add_argument("--runner-cpus", type=float, default=1)
    parser.add_argument("--learner-cpus", type=float, default=2)
    parser.add_argument("--replay-cpus", type=float, default=1)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=2048)
    parser.add_argument("--training-intensity", type=float, default=1.0)
    parser.add_argument("--replay-transitions", type=int, default=500_000)
    parser.add_argument("--replay-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--evaluation-interval", type=int, default=20_000)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--reports", type=int, default=120)
    parser.add_argument("--report-interval-s", type=float, default=10)
    parser.add_argument("--pbt-interval-reports", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-startup", action="store_true")
    args = parser.parse_args()
    for name in ("runner_cpus", "learner_cpus", "replay_cpus", "training_intensity"):
        if not 0 < getattr(args, name) < float("inf"):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.reports < 1:
        parser.error("--reports must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    thread_env = {
        name: "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    thread_env.update({"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"})
    ray.init(address=args.address, runtime_env={"env_vars": thread_env})
    monitors = {}
    monitor_errors = {}
    try:
        members = build_members(args)
        mutations = {
            f"sac/{name}": FloatMutation(1e-5, 1e-3)
            for name in ("actor_lr", "critic_lr", "alpha_lr")
        }
        if args.mode == "mixed":
            mutations["dqn/lr"] = FloatMutation(1e-5, 1e-3)
        param_space = {
            "members": members,
            "run_id": args.run_id,
            "report_interval_s": args.report_interval_s,
            "pbt": SimplePBTConfig(
                perturbation_interval_reports=args.pbt_interval_reports,
                metric_key="evaluation/latest_return_mean",
                reward_window_episodes=100,
                min_episodes_after_restart=4,
                mutations=mutations,
                seed=args.seed,
            ),
            "run_mode": "resume" if args.resume else "new",
        }
        if args.resume:
            param_space["checkpoint_path"] = str(args.resume.resolve())
        nodes = [node for node in ray.nodes() if node["Alive"]]
        inventory = preflight(param_space, nodes, ray.available_resources())
        (args.output / "preflight.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        if not inventory["ready"]:
            raise RuntimeError("Resource preflight failed; inspect preflight.json")
        if args.preflight_only:
            print(json.dumps(inventory, indent=2))
            return
        for node in nodes:
            node_id = node["NodeID"]
            monitors[node_id] = NodeMonitor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
            ).remote()
        descriptions = ray.get(
            [actor.describe.remote() for actor in monitors.values()], timeout=120
        )
        (args.output / "monitors.json").write_text(
            json.dumps(descriptions, indent=2) + "\n"
        )
        gpu_nodes = {
            node["NodeID"] for node in nodes if node["Resources"].get("GPU", 0) > 0
        }
        if any(
            item["sampler_exit_code"] is not None
            or not item["sample_written"]
            or not item["ray_processes_visible"]
            or not item["network_interfaces"]
            or (item["node_id"] in gpu_nodes and not item["gpu_devices_visible"])
            for item in descriptions
        ):
            raise RuntimeError("Monitoring preflight failed; inspect monitors.json")
        results = tune.Tuner(
            PopulationTrainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                time_budget_s=args.reports * args.report_interval_s + 600,
            ),
            run_config=RunConfig(
                name="marl",
                storage_path=str(args.output),
                stop={"training_iteration": args.reports},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1, checkpoint_at_end=True
                ),
            ),
        ).fit()
        if len(results) != 1 or results[0].error is not None:
            raise RuntimeError(f"MARL population failed: {results.errors}")
        result = results[0]
        (args.output / "result.json").write_text(
            json.dumps(result.metrics, indent=2, default=str) + "\n"
        )
        if result.metrics["training_iteration"] < args.reports:
            raise RuntimeError("Campaign stopped before its requested report budget")
        if not all(
            all(
                dict(counters)["optimizer_updates"] > 0
                for _, counters in member["learner"]["module_counters"]
            )
            for member in result.metrics["members"].values()
        ):
            raise RuntimeError("Not every policy made learner progress")
        print(f"Results and TensorBoard: {result.path}")
        print(f"Checkpoint: {result.checkpoint.path if result.checkpoint else None}")
    finally:
        for node_id, actor in monitors.items():
            try:
                evidence = ray.get(actor.stop.remote(), timeout=20)
                (args.output / f"node-{node_id}.jsonl.gz").write_bytes(
                    evidence.pop("data")
                )
                if evidence["exit_code"] != 0:
                    monitor_errors[node_id] = evidence
            except Exception as error:
                monitor_errors[node_id] = repr(error)
            finally:
                ray.kill(actor, no_restart=True)
        (args.output / "monitor-errors.json").write_text(
            json.dumps(monitor_errors, indent=2) + "\n"
        )
        ray.shutdown()
    if monitor_errors:
        raise RuntimeError("Monitoring was incomplete; inspect monitor-errors.json")


if __name__ == "__main__":
    main()
