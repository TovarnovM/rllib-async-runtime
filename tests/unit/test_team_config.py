from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib_async.examples.temporal_hierarchy import build_temporal_hierarchy_configs
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime import AsyncRuntimeConfig
from rllib_async.runtime.team import prepare_team_config


def test_team_preparation_keeps_algorithms_and_owns_independent_config_copies() -> None:
    rollout, modules, expected_spaces = build_temporal_hierarchy_configs(
        vector_worker=True, recurrent=True
    )
    sequence = SequenceConfig(learning_length=3, burn_in=2)
    prepared, learners, spaces, trainable, sequences = prepare_team_config(
        rollout,
        modules,
        AsyncRuntimeConfig(),
        trainable_modules=("worker_1", "manager"),
        sequence_configs={"manager": sequence},
    )

    assert prepared is not rollout
    assert set(spaces) == set(expected_spaces) == set(modules)
    assert trainable == ("manager", "worker_1")
    assert sequences["manager"] == sequence
    assert sequences["worker_0"] == sequences["worker_1"] == SequenceConfig()
    assert isinstance(learners["manager"], DQNConfig)
    assert isinstance(learners["worker_0"], DQNConfig)
    assert isinstance(learners["worker_1"], SACConfig)
    assert isinstance(spaces["worker_0"][0], gym.spaces.Box)
    assert isinstance(spaces["manager"][0], gym.spaces.Dict)
    assert isinstance(prepared.rl_module_spec, MultiRLModuleSpec)
    assert set(prepared.rl_module_spec.rl_module_specs) == set(modules)
    for module_id, config in learners.items():
        assert config is not modules[module_id]
        assert set(config.policies) == {module_id}
        assert set(config.rl_module_spec.rl_module_specs) == {module_id}
        assert config.policy_mapping_fn("new-agent-instance") == module_id
        assert config.count_steps_by == "agent_steps"
        assert config.num_learners == 0
        assert isinstance(modules[module_id].rl_module_spec, RLModuleSpec)
    learners["manager"].training(gamma=0.5)
    assert modules["manager"].gamma != 0.5


@pytest.mark.parametrize("module_change", ["missing", "extra", "empty"])
def test_team_rejects_incomplete_or_unknown_policy_ownership(module_change) -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs()
    if module_change == "missing":
        modules.pop("manager")
    elif module_change == "extra":
        modules["unknown"] = modules["manager"].copy(copy_frozen=False)
    else:
        modules.clear()
    with pytest.raises(ValueError, match="module_configs|same IDs"):
        prepare_team_config(rollout, modules, AsyncRuntimeConfig())


@pytest.mark.parametrize(
    "trainable", [(), ("unknown",), ("manager", "manager")]
)
def test_team_rejects_ambiguous_trainable_modules(trainable) -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs()
    with pytest.raises(ValueError, match="trainable_modules"):
        prepare_team_config(
            rollout, modules, AsyncRuntimeConfig(), trainable_modules=trainable
        )


def test_team_rejects_unknown_sequence_consumers_and_untyped_settings() -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs()
    with pytest.raises(ValueError, match="unknown module"):
        prepare_team_config(
            rollout,
            modules,
            AsyncRuntimeConfig(),
            sequence_configs={"unknown": SequenceConfig()},
        )
    with pytest.raises(TypeError, match="SequenceConfig"):
        prepare_team_config(
            rollout,
            modules,
            AsyncRuntimeConfig(),
            sequence_configs={"manager": {"learning_length": 2}},
        )


@pytest.mark.parametrize("invalid", ["remote_learner", "gpu_mismatch", "prefetch"])
def test_team_rejects_unsupported_resource_and_queue_ownership(invalid) -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs()
    settings = AsyncRuntimeConfig()
    if invalid == "remote_learner":
        modules["manager"].learners(num_learners=1)
        message = "local learners"
    elif invalid == "gpu_mismatch":
        modules["manager"].learners(num_gpus_per_learner=1)
        message = "num_gpus_per_learner"
    else:
        settings = replace(settings, batch_queue_capacity=1)
        message = "batch_queue_capacity"
    with pytest.raises(ValueError, match=message):
        prepare_team_config(rollout, modules, settings)


def test_team_requires_declared_spaces_and_discrete_dqn_actions() -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs()
    rollout.policies["manager"].observation_space = None
    with pytest.raises(ValueError, match="declare both spaces"):
        prepare_team_config(rollout, modules, AsyncRuntimeConfig())

    rollout, modules, _ = build_temporal_hierarchy_configs()
    rollout.policies["worker_0"].action_space = gym.spaces.Box(
        -1.0, 1.0, (1,), dtype=np.float32
    )
    with pytest.raises(ValueError, match="Discrete"):
        prepare_team_config(rollout, modules, AsyncRuntimeConfig())
