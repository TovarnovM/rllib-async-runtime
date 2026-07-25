from __future__ import annotations

import pytest

from rllib_async.runtime import AsyncSACRuntimeConfig, AsyncSACTrainable
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
    with pytest.raises(ValueError, match="between 4 and 16"):
        AsyncSACRuntimeConfig(runner_count=17)


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
