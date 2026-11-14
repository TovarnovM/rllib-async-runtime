from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.env import INPUT_ENV_SPACES

from rllib_async.runtime import (
    AsyncDQNTrainable,
    AsyncRuntimeConfig,
    AsyncSACRuntimeConfig,
    SingleMemberAsyncDQN,
    SingleMemberAsyncSAC,
)
from tests.helpers import make_dqn_config


def test_dqn_runtime_preserves_vector_environment_space_descriptor(monkeypatch):
    spaces = {
        INPUT_ENV_SPACES: (
            gym.spaces.Box(-1.0, 1.0, shape=(1, 4)),
            gym.spaces.MultiDiscrete([2]),
        ),
        DEFAULT_MODULE_ID: (
            gym.spaces.Box(-1.0, 1.0, shape=(4,)),
            gym.spaces.Discrete(2),
        ),
    }
    monkeypatch.setattr(
        SingleMemberAsyncSAC, "_resolve_spaces", staticmethod(lambda config: spaces)
    )
    assert SingleMemberAsyncDQN._resolve_spaces(make_dqn_config()) is spaces

    spaces[DEFAULT_MODULE_ID] = (
        spaces[DEFAULT_MODULE_ID][0],
        gym.spaces.MultiDiscrete([2]),
    )
    with pytest.raises(ValueError, match="Discrete action space"):
        SingleMemberAsyncDQN._resolve_spaces(make_dqn_config())


def test_dqn_reuses_runtime_settings_and_tune_resources() -> None:
    assert AsyncRuntimeConfig is AsyncSACRuntimeConfig
    config = make_dqn_config()
    parsed, runtime, shared, checkpoint = AsyncDQNTrainable._parse_config(
        {"dqn_config": config, "runtime": {"runner_count": 1}}
    )
    assert parsed is config
    assert runtime.batch_size == config.train_batch_size_per_learner
    assert shared is checkpoint is None
    resources = AsyncDQNTrainable.default_resource_request(
        {"dqn_config": config, "runtime": {"runner_count": 1}}
    )
    assert resources.bundles


def test_dqn_rejects_population_and_unsupported_modes_before_actor_creation() -> None:
    with pytest.raises(ValueError, match="population"):
        AsyncDQNTrainable._parse_config({"dqn_config": make_dqn_config(), "member": {}})
    with pytest.raises(ValueError, match="noisy"):
        SingleMemberAsyncDQN(
            make_dqn_config().training(noisy=True), AsyncRuntimeConfig()
        )
    with pytest.raises(ValueError, match="n_step=1"):
        SingleMemberAsyncDQN(make_dqn_config().training(n_step=3), AsyncRuntimeConfig())


@pytest.mark.parametrize(
    ("runtime_type", "saved_algorithm"),
    [(SingleMemberAsyncDQN, None), (SingleMemberAsyncSAC, "dqn")],
)
def test_checkpoint_cannot_cross_algorithm_boundary(runtime_type, saved_algorithm):
    runtime = object.__new__(runtime_type)
    state = {"state_version": 1}
    if saved_algorithm is not None:
        state["algorithm"] = saved_algorithm
    with pytest.raises(ValueError, match="algorithm"):
        runtime._restore_controller_checkpoint_state(SimpleNamespace(controller=state))


def test_legacy_sac_controller_checkpoint_remains_accepted() -> None:
    runtime = object.__new__(SingleMemberAsyncSAC)
    runtime._learner_update_budget_for_sampled = lambda sampled: 0
    state = dict.fromkeys(
        (
            "pump_iterations",
            "reports",
            "pending_rpc_high_watermark",
            "learner_updates_completed",
            "last_report_env_steps",
            "last_report_episodes",
            "next_evaluation_env_steps",
            "checkpoint_sequence",
            "restore_count",
        ),
        0,
    )
    state["state_version"] = 1
    runtime._restore_controller_checkpoint_state(SimpleNamespace(controller=state))
    assert runtime._restore_count == 1
