"""Train the end-to-end asynchronous SAC runtime on Pendulum-v1."""

from __future__ import annotations

import argparse
import math

import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.runtime import AsyncSACTrainable


def build_sac_config(*, seed: int, num_gpus: int) -> SACConfig:
    return (
        SACConfig()
        .environment("Pendulum-v1")
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
            num_gpus_per_learner=num_gpus,
        )
        .training(
            actor_lr=3e-4,
            alpha_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            n_step=1,
            num_steps_sampled_before_learning_starts=1_000,
            policy_model_config={"fcnet_hiddens": [256, 256]},
            q_model_config={"fcnet_hiddens": [256, 256]},
            target_network_update_freq=1,
            tau=0.005,
            train_batch_size_per_learner=256,
            twin_q=True,
        )
        .debugging(seed=seed)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-timesteps", type=int, default=50_000)
    parser.add_argument("--runner-count", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--evaluation-interval", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--min-improvement", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init()
    try:
        tuner = tune.Tuner(
            AsyncSACTrainable,
            param_space={
                "sac_config": build_sac_config(
                    seed=args.seed,
                    num_gpus=args.num_gpus,
                ),
                "runtime": {
                    "runner_count": args.runner_count,
                    "max_episode_steps": 200,
                    "pending_commit_high_watermark": args.runner_count * 2,
                    "pending_commit_low_watermark": args.runner_count,
                    "batch_queue_capacity": 8,
                    "learner_updates_per_tick": 4,
                    "publication_interval_updates": 10,
                    "evaluation_interval_env_steps": args.evaluation_interval,
                    "evaluation_num_episodes": args.evaluation_episodes,
                    "num_gpus_per_learner": args.num_gpus,
                    "seed": args.seed,
                },
            },
            run_config=RunConfig(
                name="async-sac-pendulum",
                stop={"timesteps_total": args.stop_timesteps},
            ),
        )
        result = tuner.fit()[0]
        if result.error is not None:
            raise result.error
        column = "evaluation/latest_return_mean"
        history = result.metrics_dataframe
        values = (
            [
                float(value)
                for value in history[column].tolist()
                if value is not None and math.isfinite(float(value))
            ]
            if column in history
            else []
        )
        if values:
            print(
                "Pendulum evaluation return: "
                f"initial={values[0]:.2f}, best={max(values):.2f}, "
                f"final={values[-1]:.2f}"
            )
        if args.require_improvement:
            if len(values) < 2:
                raise RuntimeError("no completed evaluation history was reported")
            improvement = max(values[1:]) - values[0]
            if improvement < args.min_improvement:
                raise RuntimeError(
                    f"evaluation improvement {improvement:.2f} is below "
                    f"{args.min_improvement:.2f}"
                )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
