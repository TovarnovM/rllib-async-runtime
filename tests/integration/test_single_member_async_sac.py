from __future__ import annotations

import time

import pytest
import ray
from ray.exceptions import RayActorError
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    RuntimeState,
    SingleMemberAsyncSAC,
)


def make_runtime() -> SingleMemberAsyncSAC:
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
    return SingleMemberAsyncSAC(sac_config, runtime_config)


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
