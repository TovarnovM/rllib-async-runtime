"""Run two fixed SAC members on one authoritative replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    PopulationLauncher,
    PopulationMemberSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-timesteps", type=int, default=20_000)
    parser.add_argument("--runner-count", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--num-gpus-per-member", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--storage-path", type=Path)
    return parser.parse_args()


def build_members(args: argparse.Namespace) -> tuple[PopulationMemberSpec, ...]:
    members: list[PopulationMemberSpec] = []
    for index in range(2):
        seed = args.seed + index
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
                num_steps_sampled_before_learning_starts=256,
                policy_model_config={"fcnet_hiddens": [64, 64]},
                q_model_config={"fcnet_hiddens": [64, 64]},
                train_batch_size_per_learner=128,
                twin_q=True,
            )
            .debugging(seed=seed)
        )
        runtime_config = AsyncSACRuntimeConfig(
            member_id=f"member-{index}",
            runner_count=args.runner_count,
            max_episode_steps=args.episode_length,
            pending_commit_high_watermark=args.runner_count * 2,
            pending_commit_low_watermark=args.runner_count,
            learner_updates_per_tick=4,
            publication_interval_updates=10,
            evaluation_interval_env_steps=0,
            evaluation_num_episodes=0,
            num_gpus_per_learner=args.num_gpus_per_member,
            seed=seed,
        )
        members.append(PopulationMemberSpec(sac_config, runtime_config))
    return tuple(members)


def main() -> None:
    args = parse_args()
    ray.init()
    try:
        with PopulationLauncher(build_members(args)) as launcher:
            results = launcher.fit(
                run_config=RunConfig(
                    name="async-sac-two-member-population",
                    storage_path=(
                        str(args.storage_path.resolve())
                        if args.storage_path is not None
                        else None
                    ),
                    stop={"timesteps_total": args.stop_timesteps},
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=1,
                        checkpoint_at_end=True,
                    ),
                )
            )
            for result in results:
                metrics = result.metrics
                print(
                    f"{metrics['controller']['member_id']}: "
                    f"updates={metrics['learner']['learner_updates']}, "
                    f"gpu={metrics['learner']['accelerator_ids']}, "
                    "producers="
                    f"{metrics['fast_replay']['active_producer_episode_counts']}"
                )
            if args.checkpoint_dir is not None:
                checkpoint_dir = args.checkpoint_dir.resolve()
                checkpoint_dir.mkdir(parents=True, exist_ok=False)
                checkpoint = launcher.save_checkpoint(
                    checkpoint_dir,
                    results=results,
                )
                print(f"population checkpoint: {checkpoint.directory}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
