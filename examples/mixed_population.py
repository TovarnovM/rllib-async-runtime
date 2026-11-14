"""DEMO-04: two active DQN/SAC teams, shared replay, PBT and exact resume.

Run inside the devcontainer:
    python examples/mixed_population.py --checkpoint /tmp/mixed-population

--num-gpus 1 reserves one GPU per complete team and requires two GPUs.
Replacement uses observed team returns; this functional smoke is not a quality
comparison. Equal scores postpone replacement instead of fabricating a ranking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ray

from rllib_async.examples.temporal_hierarchy import build_temporal_hierarchy_configs
from rllib_async.replay.prioritized import PrioritizedReplayConfig
from rllib_async.runtime import (
    AsyncRuntimeConfig,
    FloatMutation,
    PopulationAsyncSAC,
    PopulationMemberSpec,
    SimplePBTConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-gpus", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--timeout-s", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.timeout_s < float("inf"):
        parser.error("--timeout-s must be finite and positive")
    args.checkpoint.mkdir(parents=True, exist_ok=False)
    members = []
    for index in range(2):
        seed = args.seed + index
        rollout, modules, _ = build_temporal_hierarchy_configs(
            episode_length=8, manager_period=3,
            variable_duration=True, vector_worker=True, seed=seed,
        )
        for config in modules.values():
            config.learners(num_gpus_per_learner=args.num_gpus)
            config.training(
                n_step=2, train_batch_size_per_learner=4,
                num_steps_sampled_before_learning_starts=0,
                training_intensity=1.0,
            )
        modules["manager"].training(
            noisy=True, num_atoms=5, v_min=-10, v_max=10,
            dueling=True, double_q=True, epsilon=0.0,
        )
        settings = AsyncRuntimeConfig(
            member_id=f"team-{index}",
            runner_count=1,
            max_episode_steps=16,
            batch_size=4,
            learner_updates_per_tick=3,
            publication_interval_updates=1,
            pending_commit_high_watermark=2,
            pending_commit_low_watermark=0,
            evaluation_interval_env_steps=16,
            evaluation_num_episodes=1,
            num_cpus_per_replay=0,
            num_cpus_per_runner=0,
            num_cpus_per_evaluation_runner=0,
            num_cpus_per_learner=0,
            num_gpus_per_learner=args.num_gpus,
            seed=seed,
        )
        members.append(PopulationMemberSpec.for_team(
            rollout, modules, settings,
            per_configs={mid: PrioritizedReplayConfig() for mid in modules},
        ))
    pbt = SimplePBTConfig(
        perturbation_interval_reports=2,
        reward_window_episodes=8,
        min_episodes_after_restart=2,
        seed=args.seed,
        mutations={"manager/lr": FloatMutation(1e-5, 1e-3)},
    )
    ray.init(num_cpus=2, include_dashboard=False)
    population = None
    restored = None
    try:
        if ray.available_resources().get("GPU", 0) < 2 * args.num_gpus:
            raise RuntimeError("Two complete teams require two available GPUs")
        population = PopulationAsyncSAC(members, pbt_config=pbt)
        population.start()
        deadline = time.monotonic() + args.timeout_s
        replacement = None
        while time.monotonic() < deadline:
            report = population.run_for_report_interval(0.1)
            if report["pbt"]["event_happened"]:
                replacement = report["pbt"]
            ready = all(
                all(dict(counters)["optimizer_updates"] > 0
                    for _, counters in member["learner"]["module_counters"])
                for member in report["members"].values()
            )
            if replacement is not None and ready:
                break
        else:
            raise TimeoutError("Teams did not both learn and produce a PBT ranking")
        population.save_checkpoint(args.checkpoint)
        ids = population.runtime_member_ids
        population.stop(graceful=False)
        restored = PopulationAsyncSAC.from_checkpoint(
            members, args.checkpoint, pbt_config=pbt,
        )
        restored.start()
        assert restored.runtime_member_ids == ids
        replay_id = restored.replay_actor._actor_id
        assert all(member._replay_actor._actor_id == replay_id
                   for member in restored.members.values())
        result = {
            "replacement": replacement,
            "restored": restored.run_for_report_interval(0.2),
            "checkpoint": str(args.checkpoint.resolve()),
            "num_gpus_per_team": args.num_gpus,
        }
        encoded = json.dumps(result, indent=2, default=str)
        if args.output:
            args.output.write_text(encoded + "\n")
        print(encoded)
    finally:
        for runtime in (population, restored):
            if runtime is not None:
                runtime.stop(graceful=False)
        ray.shutdown()


if __name__ == "__main__":
    main()
