from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    AsyncSACTrainable,
    SharedReplayDescriptor,
)
from tests.helpers import make_sac_config


def test_runtime_config_uses_sac_batch_and_gpu_defaults() -> None:
    sac_config = make_sac_config()
    sac_config.training(train_batch_size_per_learner=64)
    runtime = AsyncSACRuntimeConfig.from_mapping(
        {
            "evaluation_interval_env_steps": 0,
            "evaluation_num_episodes": 0,
        },
        sac_config=sac_config,
    )

    assert runtime.batch_size == 64
    assert runtime.num_gpus_per_learner == 0


def test_runtime_config_rejects_unbounded_or_ambiguous_settings() -> None:
    sac_config = make_sac_config()

    with pytest.raises(ValueError, match="unknown runtime"):
        AsyncSACRuntimeConfig.from_mapping(
            {"mystery_queue": 10},
            sac_config=sac_config,
        )
    with pytest.raises(ValueError, match="both be zero or positive"):
        AsyncSACRuntimeConfig(
            evaluation_interval_env_steps=0,
            evaluation_num_episodes=1,
        )
    with pytest.raises(ValueError, match="between 1 and 16"):
        AsyncSACRuntimeConfig(runner_count=17)
    with pytest.raises(ValueError, match="between 1 and 16"):
        AsyncSACRuntimeConfig(runner_count=0)
    for member_id in (".", "..", "parent/child", r"parent\child"):
        with pytest.raises(ValueError, match="path segment"):
            AsyncSACRuntimeConfig(member_id=member_id)


def test_runtime_config_accepts_phase_11_direct_single_runner_baseline() -> None:
    config = AsyncSACRuntimeConfig(
        runner_count=1,
        batch_queue_capacity=0,
    )

    assert config.runner_count == 1
    assert config.batch_queue_capacity == 0


def test_trainable_resource_request_covers_every_nonzero_child_actor() -> None:
    sac_config = make_sac_config()
    resources = AsyncSACTrainable.default_resource_request(
        {
            "sac_config": sac_config,
            "runtime": {
                "runner_count": 4,
                "evaluation_num_episodes": 2,
                "evaluation_interval_env_steps": 100,
                "num_cpus_per_replay": 1,
                "num_cpus_per_learner": 1,
                "num_cpus_per_runner": 0.5,
                "num_cpus_per_evaluation_runner": 0.25,
            },
        }
    )

    assert resources.bundles == [
        {"CPU": 1.0},
        {"CPU": 1.0},
        {"CPU": 1.0},
        *[{"CPU": 0.5}] * 4,
        *[{"CPU": 0.25}] * 2,
    ]


def test_population_trainable_does_not_reserve_shared_replay_per_trial() -> None:
    sac_config = make_sac_config()
    resources = AsyncSACTrainable.default_resource_request(
        {
            "member": {
                "sac_config": sac_config,
                "runtime": {
                    "member_id": "member-1",
                    "runner_count": 4,
                    "evaluation_num_episodes": 0,
                    "evaluation_interval_env_steps": 0,
                    "num_cpus_per_replay": 1,
                    "num_cpus_per_learner": 1,
                    "num_cpus_per_runner": 0,
                },
            },
            "shared_replay": {
                "actor_name": "population-replay",
                "namespace": "population-test",
            },
        }
    )

    assert resources.bundles == [
        {"CPU": 1.0},
        {"CPU": 1.0},
    ]
    assert SharedReplayDescriptor.from_mapping(
        {
            "actor_name": "population-replay",
            "namespace": "population-test",
        }
    ) == SharedReplayDescriptor(
        actor_name="population-replay",
        namespace="population-test",
    )


def test_network_defaults_and_legacy_checkpoint_settings() -> None:
    runtime = AsyncSACRuntimeConfig()
    assert runtime.batch_queue_capacity == 0
    assert runtime.learner_updates_per_tick == runtime.publication_interval_updates == 8
    assert runtime.replay_sync_interval_s == 0.05
    assert not runtime.validate_startup
    legacy_runtime = replace(
        runtime,
        batch_queue_capacity=4,
        learner_updates_per_tick=1,
        publication_interval_updates=1,
    )
    saved = asdict(legacy_runtime)
    for name in (
        "validate_startup",
        "replay_sync_interval_s",
        "colocate_replay_and_learners",
    ):
        saved.pop(name)
    assert legacy_runtime.matches_checkpoint(saved)
    assert not runtime.matches_checkpoint(saved)
    for invalid in (-1, float("nan"), True):
        with pytest.raises(ValueError, match="replay_sync_interval_s"):
            replace(runtime, replay_sync_interval_s=invalid)


def test_colocation_reserves_one_core_bundle_without_taking_rollout_cpus() -> None:
    resources = AsyncSACTrainable.default_resource_request(
        {
            "sac_config": make_sac_config(),
            "runtime": {
                "colocate_replay_and_learners": True,
                "runner_count": 2,
                "evaluation_num_episodes": 1,
                "num_cpus_per_runner": 0.5,
                "num_cpus_per_evaluation_runner": 0.25,
            },
        }
    )
    assert resources.bundles == [
        {"CPU": 1.0},
        {"CPU": 2.0},
        {"CPU": 0.5},
        {"CPU": 0.5},
        {"CPU": 0.25},
    ]
