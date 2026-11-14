from __future__ import annotations

import time

import pytest
import ray

from rllib_async.runtime import (
    AsyncRuntimeConfig,
    RuntimeState,
    SingleMemberAsyncDQN,
    read_runtime_checkpoint,
)
from tests.helpers import make_dqn_config


@pytest.mark.integration
def test_dqn_runtime_publishes_evaluates_and_resumes(ray_runtime, tmp_path) -> None:
    config = make_dqn_config().training(
        num_steps_sampled_before_learning_starts=8,
        target_network_update_freq=1,
    )
    settings = AsyncRuntimeConfig(
        runner_count=1,
        max_episode_steps=4,
        replay_capacity_transitions=256,
        replay_capacity_bytes=1_000_000,
        pending_commit_high_watermark=2,
        pending_commit_low_watermark=0,
        batch_size=8,
        batch_queue_capacity=0,
        learner_updates_per_tick=2,
        publication_interval_updates=1,
        evaluation_interval_env_steps=8,
        evaluation_num_episodes=1,
        report_interval_s=0.1,
        shutdown_timeout_s=15,
        num_cpus_per_replay=1,
        num_cpus_per_runner=0,
        num_cpus_per_evaluation_runner=0,
        num_cpus_per_learner=0,
    )

    def train_until(runtime, *, updates, env_steps=0):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            report = runtime.run_for(0.1)
            if (
                report["learner"]["learner_updates"] >= updates
                and report["evaluation"]["rounds_completed"] >= 1
                and report["rollout"]["env_steps"] >= env_steps
            ):
                return report
        pytest.fail("DQN runtime did not complete learning and evaluation")

    runtime = SingleMemberAsyncDQN(config, settings)
    restored = None
    try:
        runtime.start()
        before = train_until(runtime, updates=2)
        assert before["controller"]["algorithm"] == "dqn"
        assert before["learner"]["latest_module_version"] >= 1
        assert before["authoritative_replay"]["total_transitions"] >= 8
        checkpoint_dir = tmp_path / "dqn"
        checkpoint_dir.mkdir()
        runtime.save_checkpoint(checkpoint_dir, timeout_s=10)
        saved = read_runtime_checkpoint(checkpoint_dir)
        assert saved.controller["algorithm"] == "dqn"
        runtime.stop(graceful=False)

        restored = SingleMemberAsyncDQN.from_checkpoint(
            config, settings, checkpoint_dir
        )
        learner = ray.get(restored._learner_actor.get_stats.remote())
        assert learner.learner_updates == saved.controller["learner_updates_completed"]
        assert (
            restored._rollout_group.get_stats().env_steps == saved.rollout["env_steps"]
        )
        restored.start()
        after = train_until(
            restored,
            updates=learner.learner_updates + 1,
            env_steps=saved.rollout["env_steps"] + 1,
        )
        assert after["controller"]["restore_count"] == 1
        assert after["rollout"]["env_steps"] > saved.rollout["env_steps"]
        assert after["batching"]["queue_capacity"] == 0
    finally:
        if runtime.state is not RuntimeState.STOPPED:
            runtime.stop(graceful=False)
        if restored is not None:
            restored.stop(timeout_s=15)
