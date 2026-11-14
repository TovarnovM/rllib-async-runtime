"""DEMO-01: vector DQN through replay, publication, evaluation and resume.

Run in the repository devcontainer:
    python examples/dqn_single_member.py --checkpoint-dir /tmp/dqn-demo

Epsilon uses the member's completed environment-step count, refreshed at each
episode boundary. The local RLlib clock advances within the episode; checkpoint
restore and actor recreation resume from the saved member count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.dqn import DQNConfig

from rllib_async.runtime import AsyncRuntimeConfig, SingleMemberAsyncDQN


def build_dqn_config(*, seed: int = 7) -> DQNConfig:
    return (
        DQNConfig()
        .environment("CartPole-v1")
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
            n_step=1,
            num_atoms=1,
            noisy=False,
            double_q=True,
            dueling=True,
            num_steps_sampled_before_learning_starts=32,
            train_batch_size_per_learner=32,
            target_network_update_freq=64,
        )
        .rl_module(model_config={"fcnet_hiddens": [32], "head_fcnet_hiddens": [32]})
        .debugging(seed=seed)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stop-env-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.stop_env_steps < 128:
        parser.error("--stop-env-steps must be at least 128")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = build_dqn_config(seed=args.seed)
    settings = AsyncRuntimeConfig(
        runner_count=1,
        max_episode_steps=32,
        batch_size=32,
        learner_updates_per_tick=2,
        publication_interval_updates=2,
        pending_commit_high_watermark=2,
        pending_commit_low_watermark=0,
        evaluation_interval_env_steps=64,
        evaluation_num_episodes=1,
        report_interval_s=0.2,
        seed=args.seed,
        num_cpus_per_runner=0,
        num_cpus_per_evaluation_runner=0,
        num_cpus_per_learner=0,
    )
    ray.init(num_cpus=2, include_dashboard=False)
    runtime = None
    try:
        runtime = SingleMemberAsyncDQN(config, settings)
        runtime.start()
        while runtime.run_for()["timesteps_total"] < args.stop_env_steps // 2:
            pass
        runtime.save_checkpoint(args.checkpoint_dir)
        runtime.stop()
        runtime = SingleMemberAsyncDQN.from_checkpoint(
            config, settings, args.checkpoint_dir
        )
        runtime.start()
        while True:
            report = runtime.run_for()
            if (
                report["timesteps_total"] >= args.stop_env_steps
                and report["learner"]["learner_updates"] > 0
                and report["evaluation"]["rounds_completed"] > 0
            ):
                break
        print(
            f"DQN: steps={report['timesteps_total']}, "
            f"updates={report['learner']['learner_updates']}, "
            f"version={report['learner']['latest_module_version']}, "
            f"restores={report['controller']['restore_count']}"
        )
    finally:
        if runtime is not None:
            runtime.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
