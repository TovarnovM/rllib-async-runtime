"""Run one Tune trial containing an always-active SAC population."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    FloatMutation,
    PopulationMemberSpec,
    PopulationTrainable,
    SimplePBTConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--reports", type=int, default=10)
    parser.add_argument("--report-interval-s", type=float, default=1.0)
    parser.add_argument("--pbt-interval-reports", type=int, default=5)
    parser.add_argument("--min-episodes-after-restart", type=int, default=4)
    parser.add_argument("--runner-count", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--num-gpus-per-member", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--storage-path", type=Path)
    args = parser.parse_args()
    if args.population_size < 2:
        parser.error("--population-size must be at least 2")
    if args.reports < 1:
        parser.error("--reports must be positive")
    if args.pbt_interval_reports < 1:
        parser.error("--pbt-interval-reports must be positive")
    if args.min_episodes_after_restart < 1:
        parser.error("--min-episodes-after-restart must be positive")
    return args


def build_members(args: argparse.Namespace) -> tuple[PopulationMemberSpec, ...]:
    members: list[PopulationMemberSpec] = []
    for index in range(args.population_size):
        seed = args.seed + index
        learning_rate_scale = 0.8 + 0.4 * index / max(args.population_size - 1, 1)
        sac_config = (
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
                num_env_runners=0,
                create_local_env_runner=True,
                num_envs_per_env_runner=1,
                batch_mode="complete_episodes",
                episodes_to_numpy=True,
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=args.num_gpus_per_member,
            )
            .training(
                actor_lr=3e-4 * learning_rate_scale,
                critic_lr=3e-4 * learning_rate_scale,
                alpha_lr=1e-4 * learning_rate_scale,
                num_steps_sampled_before_learning_starts=256,
                policy_model_config={"fcnet_hiddens": [64, 64]},
                q_model_config={"fcnet_hiddens": [64, 64]},
                train_batch_size_per_learner=128,
                twin_q=True,
            )
            .debugging(seed=seed)
        )
        runtime_config = AsyncSACRuntimeConfig(
            member_id=f"member-{index:02d}",
            runner_count=args.runner_count,
            max_episode_steps=args.episode_length,
            pending_commit_high_watermark=args.runner_count * 2,
            pending_commit_low_watermark=args.runner_count,
            learner_updates_per_tick=4,
            publication_interval_updates=10,
            evaluation_interval_env_steps=0,
            evaluation_num_episodes=0,
            report_interval_s=args.report_interval_s,
            num_gpus_per_learner=args.num_gpus_per_member,
            seed=seed,
        )
        members.append(PopulationMemberSpec(sac_config, runtime_config))
    return tuple(members)


def main() -> None:
    args = parse_args()
    ray.init()
    try:
        results = tune.Tuner(
            PopulationTrainable,
            param_space={
                "members": build_members(args),
                "report_interval_s": args.report_interval_s,
                "pbt": SimplePBTConfig(
                    perturbation_interval_reports=args.pbt_interval_reports,
                    min_episodes_after_restart=(args.min_episodes_after_restart),
                    seed=args.seed,
                    mutations={
                        "actor_lr": FloatMutation(1e-5, 1e-3),
                        "critic_lr": FloatMutation(1e-5, 1e-3),
                        "alpha_lr": FloatMutation(1e-5, 1e-3),
                    },
                ),
            },
            run_config=RunConfig(
                name="async-sac-single-trial-population",
                storage_path=(
                    str(args.storage_path.resolve())
                    if args.storage_path is not None
                    else None
                ),
                stop={"training_iteration": args.reports},
                checkpoint_config=CheckpointConfig(checkpoint_at_end=False),
            ),
        ).fit()
        if len(results) != 1:
            raise RuntimeError("single-trial population produced multiple Tune trials")
        result = results[0]
        if result.error is not None:
            raise result.error
        population = result.metrics["population"]
        print(
            f"reports={population['report_index']}, "
            f"members={population['size']}, "
            f"exploits={population['exploit_count']}, "
            f"replay_transitions={result.metrics['replay']['transitions']}"
        )
        print(f"TensorBoard log directory: {result.path}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
