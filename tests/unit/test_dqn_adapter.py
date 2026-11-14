from __future__ import annotations

import copy
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
    DefaultDQNTorchRLModule,
)
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from rllib_async.learner.action_masks import ActionMaskedDQNTorchRLModule
from rllib_async.learner.dqn_adapter import (
    DQN_TARGET_UPDATE_STATE,
    DQNBatchError,
    DQNLearnerAdapter,
    build_rllib_dqn_batch,
    prepare_dqn_config,
    validate_dqn_config,
)
from tests.helpers import assert_finite_losses, assert_tree_close, make_dqn_config


def fixed_batch(size: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    return {
        Columns.OBS: rng.normal(size=(size, 4)).astype(np.float32),
        Columns.NEXT_OBS: rng.normal(size=(size, 4)).astype(np.float32),
        Columns.ACTIONS: rng.integers(0, 2, size=size, dtype=np.int64),
        Columns.REWARDS: rng.normal(size=size).astype(np.float32),
        Columns.TERMINATEDS: np.arange(size) == 0,
        Columns.TRUNCATEDS: np.arange(size) == 1,
    }


def spaces() -> dict[str, tuple[gym.Space, gym.Space]]:
    return {
        DEFAULT_MODULE_ID: (
            gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
            gym.spaces.Discrete(2),
        )
    }


def payload(adapter: DQNLearnerAdapter) -> dict:
    learner = adapter.get_state()["learner_group"]["learner"]
    return {
        key: learner[key]
        for key in (COMPONENT_RL_MODULE, COMPONENT_OPTIMIZER, DQN_TARGET_UPDATE_STATE)
    }


def test_discrete_batch_contract() -> None:
    result = build_rllib_dqn_batch(fixed_batch()).policy_batches[DEFAULT_MODULE_ID]
    assert result.count == 8
    assert result[Columns.ACTIONS].dtype == np.int64
    np.testing.assert_array_equal(result["n_step"], np.ones(8, dtype=np.int64))
    np.testing.assert_array_equal(result["weights"], np.ones(8, dtype=np.float32))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (Columns.ACTIONS, np.zeros(8, dtype=np.float32), "integer array"),
        (Columns.ACTIONS, np.zeros((8, 1), dtype=np.int64), "one-dimensional"),
        (Columns.ACTIONS, -np.ones(8, dtype=np.int64), "Discrete action"),
        ("n_step", np.full(8, 2, dtype=np.int64), "one-step"),
    ],
)
def test_rejects_invalid_dqn_batches(column, value, message) -> None:
    batch = fixed_batch()
    batch[column] = value
    with pytest.raises(DQNBatchError, match=message):
        build_rllib_dqn_batch(batch)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_step": 3}, "n_step=1"),
        ({"n_step": (1, 3)}, "n_step=1"),
        ({"noisy": True}, "noisy"),
        ({"burn_in_len": 2}, "recurrent"),
        ({"num_atoms": 0}, "num_atoms"),
        ({"num_atoms": 5, "v_min": 1.0, "v_max": 1.0}, "v_min"),
    ],
)
def test_rejects_unsupported_config_before_learner_build(changes, message) -> None:
    config = make_dqn_config().training(**changes)
    with pytest.raises(ValueError, match=message):
        validate_dqn_config(config)


@pytest.mark.parametrize("option", ["use_lstm", "use_attention"])
def test_recurrent_config_requires_sequence_replay(option) -> None:
    config = make_dqn_config().rl_module(model_config={option: True})
    with pytest.raises(ValueError, match="recurrent"):
        validate_dqn_config(config)


@pytest.mark.parametrize(
    "action_space", [gym.spaces.Discrete(2, start=1), gym.spaces.Box(-1.0, 1.0, (1,))]
)
def test_dqn_requires_zero_based_discrete_actions(action_space) -> None:
    configured_spaces = spaces()
    configured_spaces[DEFAULT_MODULE_ID] = (
        configured_spaces[DEFAULT_MODULE_ID][0],
        action_space,
    )
    with pytest.raises(ValueError, match="zero-based Discrete"):
        validate_dqn_config(make_dqn_config(), spaces=configured_spaces)


def test_default_module_preparation_does_not_mutate_caller() -> None:
    config = make_dqn_config()
    prepared = prepare_dqn_config(config)
    assert config.rl_module_spec.module_class is DefaultDQNTorchRLModule
    assert prepared.rl_module_spec.module_class is ActionMaskedDQNTorchRLModule


@pytest.mark.parametrize(
    ("double_q", "dueling", "num_atoms"),
    [(False, False, 1), (True, True, 1), (True, True, 5)],
)
def test_adapter_matches_stock_dqn_learner(double_q, dueling, num_atoms) -> None:
    config = make_dqn_config().training(
        double_q=double_q, dueling=dueling, num_atoms=num_atoms
    )
    baseline_config = prepare_dqn_config(config)
    baseline_config.learners(learner_class=DQNTorchLearner)
    baseline = baseline_config.build_learner_group(spaces=spaces())
    adapter = None
    try:
        adapter = DQNLearnerAdapter(
            config, spaces=spaces(), member_id="dqn", publication_interval_updates=1
        )
        baseline.set_state(copy.deepcopy(adapter.get_state()["learner_group"]))
        batch = fixed_batch()
        expected = baseline.update(
            batch=build_rllib_dqn_batch(batch),
            timesteps={
                NUM_ENV_STEPS_SAMPLED_LIFETIME: 16,
                NUM_AGENT_STEPS_SAMPLED_LIFETIME: 16,
            },
        )
        actual = adapter.update(batch, sampled_env_steps=16)
        assert actual.performed
        assert actual.published_weights is not None
        assert_finite_losses(expected)
        assert_finite_losses(actual.learner_results)
        expected_state = baseline.get_state()["learner"]
        for key in (COMPONENT_RL_MODULE, COMPONENT_OPTIMIZER):
            assert_tree_close(payload(adapter)[key], expected_state[key])
        assert payload(adapter)[DQN_TARGET_UPDATE_STATE] == {DEFAULT_MODULE_ID: 16}
    finally:
        baseline.shutdown()
        if adapter is not None:
            adapter.close()


def test_full_restore_preserves_optimizer_rng_and_target_update_schedule() -> None:
    config = make_dqn_config().training(num_steps_sampled_before_learning_starts=8)
    source = DQNLearnerAdapter(
        config, spaces=spaces(), member_id="dqn", publication_interval_updates=2
    )
    restored = None
    try:
        batch = fixed_batch()
        assert not source.update(batch, sampled_env_steps=4).performed
        assert source.update(batch, sampled_env_steps=8).published_weights is None
        checkpoint = pickle.loads(pickle.dumps(source.get_state()))
        assert payload(source)[DQN_TARGET_UPDATE_STATE] == {DEFAULT_MODULE_ID: 0}
        assert payload(source)[COMPONENT_OPTIMIZER]
        expected_rng = torch.rand(4)
        restored = DQNLearnerAdapter(
            config, spaces=spaces(), member_id="dqn", publication_interval_updates=2
        )
        restored.set_state(copy.deepcopy(checkpoint))
        torch.testing.assert_close(torch.rand(4), expected_rng)
        assert_tree_close(payload(source), payload(restored))
        for steps, target_steps in ((15, 0), (16, 16)):
            step_checkpoint = source.get_state()
            actual = source.update(batch, sampled_env_steps=steps)
            restored.set_state(copy.deepcopy(step_checkpoint))
            resumed = restored.update(batch, sampled_env_steps=steps)
            assert actual.learner_updates == resumed.learner_updates
            assert bool(actual.published_weights) == bool(resumed.published_weights)
            assert_tree_close(payload(source), payload(restored))
            assert payload(source)[DQN_TARGET_UPDATE_STATE] == {
                DEFAULT_MODULE_ID: target_steps
            }
        incompatible = copy.deepcopy(checkpoint)
        incompatible["config_contract"]["double_q"] = not config.double_q
        with pytest.raises(ValueError, match="configuration"):
            restored.set_state(incompatible)
        different_model = copy.deepcopy(checkpoint)
        different_model["module_config_contract"]["rl_module_spec"].model_config = {
            "fcnet_activation": "tanh"
        }
        with pytest.raises(ValueError, match="module configuration"):
            restored.set_state(different_model)
        invalid_clock = copy.deepcopy(checkpoint)
        invalid_clock["learner_group"]["learner"][DQN_TARGET_UPDATE_STATE][
            DEFAULT_MODULE_ID
        ] = 9
        with pytest.raises(ValueError, match="target update state"):
            restored.set_state(invalid_clock)
    finally:
        source.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("validate_startup", [False, True])
def test_dqn_content_diagnostics_run_only_at_startup(
    monkeypatch, validate_startup
) -> None:
    import rllib_async.learner.dqn_adapter as module

    original = module.build_rllib_dqn_batch
    calls = []

    def build(batch, *, validate=True):
        calls.append(validate)
        return original(batch, validate=validate)

    monkeypatch.setattr(module, "build_rllib_dqn_batch", build)
    with DQNLearnerAdapter(
        make_dqn_config(),
        spaces=spaces(),
        member_id="dqn",
        publication_interval_updates=8,
        validate_startup=validate_startup,
    ) as adapter:
        publication = adapter.get_published_weights()
        original_state = copy.deepcopy(publication.state)
        adapter.update(fixed_batch(), sampled_env_steps=8)
        adapter.update(fixed_batch(), sampled_env_steps=16)
        assert calls == [validate_startup, False]
        assert adapter.get_published_weights() is publication
        assert_tree_close(publication.state, original_state)


def test_discrete_action_bounds_checked_by_startup_diagnostics() -> None:
    with DQNLearnerAdapter(
        make_dqn_config(),
        spaces=spaces(),
        member_id="dqn",
        publication_interval_updates=8,
        validate_startup=True,
    ) as adapter:
        batch = fixed_batch()
        batch[Columns.ACTIONS][:] = 2
        with pytest.raises(DQNBatchError, match="Discrete action"):
            adapter.update(batch, sampled_env_steps=8)
