from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib_async.examples.temporal_hierarchy import build_temporal_hierarchy_configs
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime import AsyncRuntimeConfig, SingleMemberAsyncSAC
from rllib_async.runtime.team import (
    SingleMemberAsyncTeam,
    _rollout_contract,
    prepare_team_config,
)


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


@pytest.mark.parametrize("trainable", [(), ("unknown",), ("manager", "manager")])
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


def test_team_preparation_resolves_overrides_before_sampling_and_budgeting() -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs()
    modules["manager"].multi_agent(
        algorithm_config_overrides_per_module={
            "manager": {
                "gamma": 0.5,
                "n_step": 2,
                "train_batch_size_per_learner": 3,
                "training_intensity": 2.0,
                "num_steps_sampled_before_learning_starts": 7,
            }
        }
    )
    _, learners, _, _, _ = prepare_team_config(
        rollout, modules, AsyncRuntimeConfig()
    )
    manager = learners["manager"]
    assert manager.gamma == 0.5
    assert manager.n_step == 2
    assert manager.train_batch_size_per_learner == 3
    assert manager.training_intensity == 2.0
    assert manager.num_steps_sampled_before_learning_starts == 7
    assert manager.get_config_for_module("manager").gamma == manager.gamma
    assert modules["manager"].gamma != manager.gamma


def test_rollout_contract_preserves_copies_and_distinguishes_data_semantics() -> None:
    rollout, _, _ = build_temporal_hierarchy_configs()
    original = _rollout_contract(rollout)
    assert _rollout_contract(rollout.copy(copy_frozen=False)) == original
    changed = rollout.copy(copy_frozen=False)
    changed.environment(normalize_actions=not rollout.normalize_actions)
    assert _rollout_contract(changed) != original
    changed = rollout.copy(copy_frozen=False)
    changed.environment(clip_actions=not rollout.clip_actions)
    assert _rollout_contract(changed) != original
    changed = rollout.copy(copy_frozen=False)
    changed.env_config["manager_period"] += 1
    assert _rollout_contract(changed) != original

    def routing(worker):
        def map_agent(agent_id, *args, **kwargs):
            return "manager" if agent_id == "manager" else worker

        return map_agent

    changed.policy_mapping_fn = routing("worker_0")
    worker_zero = _rollout_contract(changed)
    assert _rollout_contract(changed.copy(copy_frozen=False)) == worker_zero
    changed.policy_mapping_fn = routing("worker_1")
    assert _rollout_contract(changed) != worker_zero


def test_team_restore_rejects_changed_semantics_before_applying_controller_state(
    monkeypatch,
) -> None:
    runtime = SingleMemberAsyncTeam.__new__(SingleMemberAsyncTeam)
    runtime._team_rollout_contract = "current-semantics"
    applied = []
    monkeypatch.setattr(
        SingleMemberAsyncSAC,
        "_restore_controller_checkpoint_state",
        lambda self, state: applied.append(state),
    )
    checkpoint = SimpleNamespace(controller={"rollout_contract": "old-semantics"})
    with pytest.raises(ValueError, match="policy routing or action transforms"):
        runtime._restore_controller_checkpoint_state(checkpoint)
    assert applied == []
    checkpoint.controller["rollout_contract"] = "current-semantics"
    runtime._restore_controller_checkpoint_state(checkpoint)
    assert applied == [checkpoint]
