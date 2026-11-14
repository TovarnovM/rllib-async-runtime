from __future__ import annotations

import json
import time

import pytest
import ray
import torch
from ray.exceptions import GetTimeoutError, RayActorError

from rllib_async.runtime import (
    AsyncRuntimeConfig,
    RuntimeState,
    SingleMemberAsyncDQN,
    read_runtime_checkpoint,
)
from tests.helpers import assert_learner_parameter_devices, make_dqn_config


@pytest.mark.integration
@pytest.mark.parametrize(
    "num_gpus_per_learner",
    [pytest.param(0, id="cpu"), pytest.param(1, marks=pytest.mark.gpu, id="gpu")],
)
def test_dqn_runtime_publishes_evaluates_and_resumes(
    ray_runtime, tmp_path, num_gpus_per_learner, request
) -> None:
    if num_gpus_per_learner and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this container")
    config = make_dqn_config().training(
        num_steps_sampled_before_learning_starts=8,
        target_network_update_freq=1,
        epsilon=[[0, 1.0], [10_000, 0.05]],
    )
    config.learners(num_gpus_per_learner=num_gpus_per_learner)
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
        num_gpus_per_learner=num_gpus_per_learner,
    )

    device_type = "cuda" if num_gpus_per_learner else "cpu"
    available_gpus = ray.available_resources().get("GPU", 0)
    assert available_gpus >= num_gpus_per_learner
    request.node.user_properties.append(
        ("ray_gpu_resources", str(ray.cluster_resources().get("GPU", 0)))
    )
    actor_probes = []

    def inspect_learner(runtime, label):
        devices, env_steps, agent_steps = ray.get(
            runtime._learner_actor.__ray_call__.remote(
                lambda actor: (
                    assert_learner_parameter_devices(
                        actor._host._adapter._learner_group, device_type
                    ),
                    actor._host._adapter.sampled_env_steps,
                    actor._host._adapter.sampled_agent_steps,
                )
            ),
            timeout=10,
        )
        request.node.user_properties.append(
            (f"{label}_parameter_devices", json.dumps(devices))
        )
        return env_steps, agent_steps

    def exploration_state(runtime):
        def inspect_runner(actor):
            env_runner = actor._runner._env_runner
            return (
                env_runner.num_env_steps_sampled_lifetime,
                float(env_runner.module.epsilon_schedule.get_current_value()),
            )

        (runner,) = runtime._rollout_group._runners.values()
        return ray.get(runner.__ray_call__.remote(inspect_runner), timeout=10)

    def remember_actors(runtime):
        actor_probes.extend(
            [
                runtime._replay_actor,
                runtime._learner_actor,
                *runtime._rollout_group._runners.values(),
                *runtime._evaluation_group._actors.values(),
            ]
        )

    def assert_actors_stopped():
        # Same finite actor-death polling as the SAC runtime lifecycle gate.
        for actor in actor_probes:
            deadline = time.monotonic() + 5
            while True:
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=1)
                except RayActorError:
                    break
                except GetTimeoutError:
                    pass
                if time.monotonic() >= deadline:
                    pytest.fail(f"Actor {actor} remained alive after stop")
                time.sleep(0.05)
        if num_gpus_per_learner:
            deadline = time.monotonic() + 5
            while ray.available_resources().get("GPU", 0) < available_gpus:
                if time.monotonic() >= deadline:
                    pytest.fail("Learner GPU resources did not return after stop")
                time.sleep(0.05)

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
        remember_actors(runtime)
        runtime.start()
        before = train_until(runtime, updates=2)
        assert before["controller"]["algorithm"] == "dqn"
        assert before["learner"]["latest_module_version"] >= 1
        assert before["authoritative_replay"]["total_transitions"] >= 8
        assert before["rollout"]["episodes_committed"] >= 1
        assert before["learner"]["weight_publications"] >= 1
        checkpoint_dir = tmp_path / "dqn"
        checkpoint_dir.mkdir()
        # Keep the checkpoint cut quiescent for owner-side clock inspection.
        runtime.pause(timeout_s=10)
        checkpoint = runtime.save_checkpoint(checkpoint_dir, timeout_s=10)
        saved = read_runtime_checkpoint(checkpoint_dir)
        # CUDA checkpoint bytes remain opaque to the controller/test process.
        source_steps = inspect_learner(runtime, "source")
        source_learner = ray.get(runtime._learner_actor.get_stats.remote(), timeout=10)
        assert saved.controller["algorithm"] == "dqn"
        assert checkpoint.replay_cursor == saved.replay_cursor
        assert source_learner.fast_replay.cursor == saved.replay_cursor
        assert source_learner.fast_replay.active_cursor == saved.replay_cursor
        assert len(source_learner.accelerator_ids) == num_gpus_per_learner
        assert source_learner.learner_updates == saved.controller[
            "learner_updates_completed"
        ]
        replay = ray.get(runtime._replay_actor.get_stats.remote(), timeout=10)
        assert replay.cursor == saved.replay_cursor
        source_clock, source_epsilon = exploration_state(runtime)
        assert source_clock == saved.rollout["env_steps"]
        assert 0.05 <= source_epsilon < 1.0
        runtime.stop(graceful=False)
        assert_actors_stopped()

        restored = SingleMemberAsyncDQN.from_checkpoint(
            config, settings, checkpoint_dir
        )
        remember_actors(restored)
        assert inspect_learner(restored, "restored") == source_steps
        learner = ray.get(restored._learner_actor.get_stats.remote(), timeout=10)
        assert learner.learner_updates == saved.controller["learner_updates_completed"]
        assert learner.learner_updates == source_learner.learner_updates
        assert learner.latest_module_version == source_learner.latest_module_version
        assert learner.weight_publications == source_learner.weight_publications
        assert len(learner.accelerator_ids) == num_gpus_per_learner
        replay = ray.get(restored._replay_actor.get_stats.remote(), timeout=10)
        assert replay.cursor == saved.replay_cursor
        assert learner.fast_replay.cursor == saved.replay_cursor
        assert learner.fast_replay.active_cursor == saved.replay_cursor
        rollout = restored._rollout_group.get_stats()
        assert rollout.agent_steps == saved.rollout["agent_steps"]
        assert dict(rollout.runner_generations) == {
            runner_id: generation + 1
            for runner_id, generation in saved.rollout["runner_generations"].items()
        }
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
        assert after["rollout"]["agent_steps"] > saved.rollout["agent_steps"]
        assert after["learner"]["latest_module_version"] > learner.latest_module_version
        assert after["learner"]["weight_publications"] > learner.weight_publications
        restored.drain(timeout_s=10)
        env_steps, agent_steps = inspect_learner(restored, "continued")
        assert env_steps > source_steps[0]
        assert agent_steps > source_steps[1]
        resumed_clock, resumed_epsilon = exploration_state(restored)
        assert resumed_clock == restored._rollout_group.get_stats().env_steps
        assert resumed_clock > source_clock
        assert 0.05 <= resumed_epsilon <= source_epsilon
        request.node.user_properties.append(
            (
                "exploration_resume",
                json.dumps(
                    {
                        "source": [source_clock, source_epsilon],
                        "restored": [resumed_clock, resumed_epsilon],
                    }
                ),
            )
        )
    finally:
        try:
            if runtime.state is not RuntimeState.STOPPED:
                runtime.stop(graceful=False)
        finally:
            try:
                if restored is not None:
                    restored.stop(timeout_s=15)
            finally:
                assert_actors_stopped()
