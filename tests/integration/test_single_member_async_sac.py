from __future__ import annotations

import time
from dataclasses import asdict

import pytest
import ray
from ray.exceptions import RayActorError
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    AsyncSACTrainable,
    RuntimeState,
    SingleMemberAsyncSAC,
    read_runtime_checkpoint,
)
from rllib_async.runtime.learner_host import decode_learner_host_checkpoint


def make_configs() -> tuple[SACConfig, AsyncSACRuntimeConfig]:
    sac_config = (
        SACConfig()
        .environment(
            SyntheticThroughputEnv,
            env_config={"episode_length": 4},
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
            num_steps_sampled_before_learning_starts=8,
            policy_model_config={"fcnet_hiddens": [16]},
            q_model_config={"fcnet_hiddens": [16]},
            target_network_update_freq=1,
            train_batch_size_per_learner=8,
            twin_q=True,
        )
        .debugging(seed=20260725)
    )
    runtime_config = AsyncSACRuntimeConfig(
        runner_count=4,
        max_episode_steps=4,
        replay_capacity_transitions=1_024,
        replay_capacity_bytes=10_000_000,
        replay_journal_capacity=256,
        replay_sync_max_bytes=1_000_000,
        pending_commit_high_watermark=8,
        pending_commit_low_watermark=3,
        batch_size=8,
        batch_queue_capacity=2,
        learner_updates_per_tick=2,
        publication_interval_updates=1,
        evaluation_interval_env_steps=8,
        evaluation_num_episodes=2,
        report_interval_s=0.2,
        event_poll_timeout_s=0.01,
        shutdown_timeout_s=15,
        seed=20260725,
        num_cpus_per_replay=1,
        num_cpus_per_runner=0,
        num_cpus_per_evaluation_runner=0,
        num_cpus_per_learner=0,
        num_gpus_per_learner=0,
    )
    return sac_config, runtime_config


def make_runtime() -> SingleMemberAsyncSAC:
    return SingleMemberAsyncSAC(*make_configs())


@pytest.mark.integration
def test_end_to_end_runtime_reports_all_layers_and_stops_actors(
    ray_runtime: None,
) -> None:
    runtime = make_runtime()
    assert runtime._replay_actor is not None
    assert runtime._learner_actor is not None
    assert runtime._rollout_group is not None
    assert runtime._evaluation_group is not None
    actor_probes = [
        (runtime._replay_actor, "get_stats"),
        (runtime._learner_actor, "get_stats"),
        *((actor, "close") for actor in runtime._rollout_group._runners.values()),
        *((actor, "close") for actor in runtime._evaluation_group._actors.values()),
    ]
    report = None
    try:
        runtime.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            report = runtime.run_for(0.2)
            if (
                report["learner"]["learner_updates"] >= 2
                and report["evaluation"]["rounds_completed"] >= 1
            ):
                break
        assert report is not None
        assert report["rollout"]["episodes_committed"] >= 1
        assert report["authoritative_replay"]["total_transitions"] >= 1
        assert report["fast_replay"]["total_transitions"] >= 1
        assert report["batching"]["queue_high_watermark"] <= 2
        assert report["learner"]["learner_updates"] >= 2
        assert report["evaluation"]["rounds_completed"] >= 1
        assert report["evaluation"]["pending_high_watermark"] <= 2
        assert (
            report["controller"]["pending_rpc_high_watermark"]
            <= report["controller"]["pending_rpc_bound"]
        )

        runtime.pause(timeout_s=10)
        assert runtime.state is RuntimeState.PAUSED
        runtime.resume()
        assert runtime.state is RuntimeState.RUNNING
        runtime.run_for(0.1)
    finally:
        runtime.stop(timeout_s=15)
    assert runtime.state is RuntimeState.STOPPED

    for actor, method_name in actor_probes:
        deadline = time.monotonic() + 5
        while True:
            try:
                ray.get(getattr(actor, method_name).remote(), timeout=1)
            except RayActorError:
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
            if time.monotonic() >= deadline:
                pytest.fail(f"{method_name} actor remained alive after stop")
            time.sleep(0.05)


@pytest.mark.integration
def test_controlled_crash_restores_every_runtime_layer(
    ray_runtime: None,
    tmp_path,
) -> None:
    sac_config, runtime_config = make_configs()
    runtime = SingleMemberAsyncSAC(sac_config, runtime_config)
    restored = None
    report = None
    try:
        runtime.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            report = runtime.run_for(0.2)
            if (
                report["learner"]["learner_updates"] >= 2
                and report["evaluation"]["rounds_completed"] >= 1
            ):
                break
        assert report is not None
        checkpoint_dir = tmp_path / "runtime-checkpoint"
        checkpoint_dir.mkdir()
        checkpoint = runtime.save_checkpoint(checkpoint_dir, timeout_s=10)
        assert runtime.state is RuntimeState.RUNNING
        with pytest.raises(FileExistsError, match="already contains"):
            runtime.save_checkpoint(checkpoint_dir, timeout_s=10)
        assert runtime.state is RuntimeState.RUNNING

        saved = read_runtime_checkpoint(checkpoint_dir)
        saved_learner = decode_learner_host_checkpoint(saved.learner)
        assert checkpoint.replay_cursor == saved.replay_cursor
        assert "fast_replay" not in saved_learner
        assert "batch_producer" not in saved_learner
        assert saved.controller["learner_updates_completed"] >= 2
        saved_generations = saved.rollout["runner_generations"]
        replay_snapshot = ray.get(runtime._replay_actor.get_snapshot.remote())
        assert replay_snapshot.episodes

        runtime.stop(graceful=False)
        restored = SingleMemberAsyncSAC.from_checkpoint(
            sac_config,
            runtime_config,
            checkpoint_dir,
        )
        assert restored.state is RuntimeState.CREATED
        restored_replay = ray.get(restored._replay_actor.get_stats.remote())
        restored_learner = ray.get(restored._learner_actor.get_stats.remote())
        restored_rollout = restored._rollout_group.get_stats()
        assert restored_replay.cursor == saved.replay_cursor
        assert restored_learner.fast_replay.cursor == saved.replay_cursor
        assert restored_learner.fast_replay.active_cursor == saved.replay_cursor
        assert (
            restored_learner.learner_updates
            == saved.controller["learner_updates_completed"]
        )
        assert dict(restored_rollout.runner_generations) == {
            runner_id: generation + 1
            for runner_id, generation in saved_generations.items()
        }

        duplicate = ray.get(
            restored._replay_actor.commit_episode.remote(replay_snapshot.episodes[0])
        )
        assert duplicate.duplicate
        assert not duplicate.committed

        restored.start()
        previous_updates = restored_learner.learner_updates
        deadline = time.monotonic() + 20
        continued = None
        while time.monotonic() < deadline:
            continued = restored.run_for(0.2)
            if (
                continued["learner"]["learner_updates"] > previous_updates
                and continued["rollout"]["env_steps"] > saved.rollout["env_steps"]
            ):
                break
        assert continued is not None
        assert continued["learner"]["learner_updates"] > previous_updates
        assert continued["controller"]["restore_count"] == 1
        assert continued["controller"]["checkpoint_sequence"] == 1
        assert continued["rollout"]["env_steps"] > saved.rollout["env_steps"]
    finally:
        if runtime.state is not RuntimeState.STOPPED:
            runtime.stop(graceful=False)
        if restored is not None:
            restored.stop(timeout_s=15)


@pytest.mark.integration
def test_tune_trainable_continues_from_directory_checkpoint(
    ray_runtime: None,
    tmp_path,
) -> None:
    sac_config, runtime_config = make_configs()
    trainable = AsyncSACTrainable(
        config={
            "sac_config": sac_config,
            "runtime": asdict(runtime_config),
        }
    )
    try:
        before = trainable.step()
        checkpoint_dir = tmp_path / "tune-checkpoint"
        checkpoint_dir.mkdir()
        assert trainable.save_checkpoint(str(checkpoint_dir)) is None

        trainable.load_checkpoint(str(checkpoint_dir))
        assert trainable._runtime.state is RuntimeState.RUNNING
        after = trainable.step()
        assert (
            after["learner"]["learner_updates"] >= before["learner"]["learner_updates"]
        )
        assert after["controller"]["restore_count"] == 1
    finally:
        trainable.cleanup()
