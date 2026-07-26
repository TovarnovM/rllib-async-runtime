from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
import ray
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune import Stopper

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    PopulationLauncher,
    PopulationMemberSpec,
    SingleMemberAsyncSAC,
    read_population_checkpoint_bundle,
)


def make_population_specs(
    *,
    num_gpus_per_learner: int = 0,
) -> tuple[PopulationMemberSpec, PopulationMemberSpec]:
    members: list[PopulationMemberSpec] = []
    for index in range(2):
        seed = 20260726 + index
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
            .learners(
                num_learners=0,
                num_gpus_per_learner=num_gpus_per_learner,
            )
            .training(
                num_steps_sampled_before_learning_starts=8,
                policy_model_config={"fcnet_hiddens": [16]},
                q_model_config={"fcnet_hiddens": [16]},
                target_network_update_freq=1,
                train_batch_size_per_learner=8,
                twin_q=True,
            )
            .debugging(seed=seed)
        )
        runtime_config = AsyncSACRuntimeConfig(
            member_id=f"member-{index}",
            runner_count=4,
            max_episode_steps=4,
            replay_capacity_transitions=2_048,
            replay_capacity_bytes=20_000_000,
            replay_journal_capacity=512,
            replay_sync_max_bytes=1_000_000,
            pending_commit_high_watermark=8,
            pending_commit_low_watermark=3,
            batch_size=8,
            batch_queue_capacity=2,
            learner_updates_per_tick=2,
            publication_interval_updates=1,
            evaluation_interval_env_steps=0,
            evaluation_num_episodes=0,
            report_interval_s=0.1,
            event_poll_timeout_s=0.005,
            shutdown_timeout_s=20,
            seed=seed,
            num_cpus_per_replay=0,
            num_cpus_per_runner=0,
            num_cpus_per_evaluation_runner=0,
            num_cpus_per_learner=0,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        members.append(PopulationMemberSpec(sac_config, runtime_config))
    return members[0], members[1]


def population_actor_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class PopulationReadyStopper(Stopper):
    """Wait for both population data and learner progress, with a hard bound."""

    def __init__(self, *, max_iterations: int = 100) -> None:
        self._max_iterations = max_iterations

    def __call__(self, trial_id: str, result: dict) -> bool:
        del trial_id
        producers = set(
            result.get("fast_replay", {}).get(
                "active_producer_episode_counts",
                {},
            )
        )
        learner_updates = result.get("learner", {}).get("learner_updates", 0)
        ready = learner_updates >= 1 and {"member-0", "member-1"}.issubset(producers)
        return ready or result.get("training_iteration", 0) >= self._max_iterations

    def stop_all(self) -> bool:
        return False


def test_population_ready_stopper_requires_data_and_learner_progress() -> None:
    stopper = PopulationReadyStopper(max_iterations=3)
    one_producer = {
        "training_iteration": 1,
        "learner": {"learner_updates": 1},
        "fast_replay": {
            "active_producer_episode_counts": {"member-0": 1},
        },
    }
    no_updates = {
        "training_iteration": 2,
        "learner": {"learner_updates": 0},
        "fast_replay": {
            "active_producer_episode_counts": {
                "member-0": 1,
                "member-1": 1,
            },
        },
    }
    ready = {
        **no_updates,
        "learner": {"learner_updates": 1},
    }

    assert not stopper("member-0", one_producer)
    assert not stopper("member-0", no_updates)
    assert stopper("member-0", ready)
    assert stopper("member-0", {"training_iteration": 3})
    assert not stopper.stop_all()


@pytest.mark.integration
def test_stopping_one_member_preserves_other_member_and_shared_replay(
    ray_runtime: None,
) -> None:
    specs = make_population_specs()
    launcher = PopulationLauncher(
        specs,
        replay_actor_name=population_actor_name("lifecycle-replay"),
    )
    member_zero = None
    member_one = None
    try:
        launcher.start()
        named_replay = ray.get_actor(
            launcher.descriptor.actor_name,
            namespace=launcher.descriptor.namespace,
        )
        assert named_replay._actor_id == launcher.replay_actor._actor_id

        member_zero = SingleMemberAsyncSAC(
            specs[0].sac_config,
            specs[0].runtime_config,
            replay_actor=launcher.replay_actor,
        )
        member_one = SingleMemberAsyncSAC(
            specs[1].sac_config,
            specs[1].runtime_config,
            replay_actor=launcher.replay_actor,
        )
        assert (
            member_zero._learner_actor._actor_id != member_one._learner_actor._actor_id
        )
        member_zero.start()
        member_one.start()

        reports = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            reports = (
                member_zero.run_for(0.1),
                member_one.run_for(0.1),
            )
            if all(
                report["learner"]["learner_updates"] >= 1
                and set(report["fast_replay"]["active_producer_episode_counts"])
                == {"member-0", "member-1"}
                for report in reports
            ):
                break
        assert reports is not None
        assert all(report["learner"]["learner_updates"] >= 1 for report in reports)
        assert all(
            set(report["fast_replay"]["active_producer_episode_counts"])
            == {"member-0", "member-1"}
            for report in reports
        )
        weights = ray.get(
            [
                member_zero._learner_actor.get_published_weights.remote(),
                member_one._learner_actor.get_published_weights.remote(),
            ]
        )
        assert [weights[0].member_id, weights[1].member_id] == [
            "member-0",
            "member-1",
        ]

        member_zero.stop(timeout_s=20)
        before = member_one.get_report()
        replay_before = launcher.get_replay_stats()
        deadline = time.monotonic() + 20
        after = before
        while time.monotonic() < deadline:
            after = member_one.run_for(0.1)
            if after["rollout"]["env_steps"] > before["rollout"]["env_steps"]:
                break
        assert after["rollout"]["env_steps"] > before["rollout"]["env_steps"]
        replay_after = launcher.get_replay_stats()
        assert replay_after.cursor.mutation_seq > replay_before.cursor.mutation_seq
    finally:
        if member_zero is not None and member_zero.state.value != "stopped":
            member_zero.stop(graceful=False)
        if member_one is not None and member_one.state.value != "stopped":
            member_one.stop(graceful=False)
        launcher.close()


@pytest.mark.integration
def test_tune_population_checkpoint_restores_shared_replay_once(
    ray_runtime: None,
    tmp_path,
) -> None:
    specs = make_population_specs()
    checkpoint_dir = tmp_path / "population-checkpoint"
    checkpoint_dir.mkdir()
    launcher = PopulationLauncher(
        specs,
        replay_actor_name=population_actor_name("tune-replay"),
    )
    try:
        results = launcher.fit(
            run_config=RunConfig(
                name="phase-8-population",
                storage_path=str(tmp_path / "ray-results"),
                stop=PopulationReadyStopper(),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
                verbose=0,
            )
        )
        result_list = list(results)
        assert all(
            result.metrics["learner"]["learner_updates"] >= 1 for result in result_list
        )
        assert all(
            set(result.metrics["fast_replay"]["active_producer_episode_counts"])
            == {"member-0", "member-1"}
            for result in result_list
        )
        assert all(
            result.metrics["controller"]["owns_replay_actor"] is False
            for result in result_list
        )
        for result in result_list:
            assert result.checkpoint is not None
            with result.checkpoint.as_directory() as checkpoint_directory:
                assert not (Path(checkpoint_directory) / "replay.snapshot").exists()
        active_intervals = [
            (
                result.metrics["controller"]["started_at_monotonic"],
                result.metrics["controller"]["reported_at_monotonic"],
            )
            for result in result_list
        ]
        assert max(start for start, _ in active_intervals) < min(
            end for _, end in active_intervals
        )

        checkpoint = launcher.save_checkpoint(checkpoint_dir, results=results)
        manifest, replay_state, member_states = read_population_checkpoint_bundle(
            checkpoint_dir
        )
        assert checkpoint.member_ids == ("member-0", "member-1")
        assert manifest.replay_cursor.mutation_seq > 0
        assert replay_state.mutation_seq == manifest.replay_cursor.mutation_seq
        assert set(member_states) == {"member-0", "member-1"}
    finally:
        launcher.close()

    restored = PopulationLauncher.from_checkpoint(
        specs,
        checkpoint_dir,
        replay_actor_name=population_actor_name("restored-replay"),
    )
    try:
        assert restored.get_replay_stats().cursor == manifest.replay_cursor
        resumed = restored.fit(
            run_config=RunConfig(
                name="phase-8-population-restored",
                storage_path=str(tmp_path / "ray-results-restored"),
                stop={"training_iteration": 1},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
                verbose=0,
            )
        )
        assert all(
            result.metrics["controller"]["restore_count"] == 1 for result in resumed
        )
    finally:
        restored.close()
