"""Measure bounded end-to-end throughput on a cheap synthetic environment."""

from __future__ import annotations

import argparse

import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import AsyncSACTrainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-timesteps", type=int, default=20_000)
    parser.add_argument("--runner-count", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(
            num_steps_sampled_before_learning_starts=256,
            policy_model_config={"fcnet_hiddens": [64, 64]},
            q_model_config={"fcnet_hiddens": [64, 64]},
            train_batch_size_per_learner=128,
            twin_q=True,
        )
        .debugging(seed=args.seed)
    )

    ray.init()
    try:
        result = tune.Tuner(
            AsyncSACTrainable,
            param_space={
                "sac_config": sac_config,
                "runtime": {
                    "runner_count": args.runner_count,
                    "max_episode_steps": args.episode_length,
                    "pending_commit_high_watermark": args.runner_count * 2,
                    "pending_commit_low_watermark": args.runner_count,
                    "learner_updates_per_tick": 4,
                    "publication_interval_updates": 10,
                    "evaluation_interval_env_steps": 0,
                    "evaluation_num_episodes": 0,
                    "seed": args.seed,
                },
            },
            run_config=RunConfig(
                name="async-sac-throughput",
                stop={"timesteps_total": args.stop_timesteps},
            ),
        ).fit()[0]
        if result.error is not None:
            raise result.error
        metrics = result.metrics
        print(
            "rollout_env_steps_per_s="
            f"{metrics['rollout']['env_steps_per_s']:.1f}, "
            "learner_updates_per_s="
            f"{metrics['learner']['updates_per_s']:.1f}, "
            "pending_rpc_high_watermark="
            f"{metrics['controller']['pending_rpc_high_watermark']}"
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
