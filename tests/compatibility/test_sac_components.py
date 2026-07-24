from __future__ import annotations

import copy

import numpy as np
import torch
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core import (
    COMPONENT_OPTIMIZER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO
from ray.rllib.utils.replay_buffers.episode_replay_buffer import (
    EpisodeReplayBuffer,
)

from tests.helpers import (
    SAC_TEMPERATURE_STATE,
    TemperatureCheckpointSACLearner,
    assert_finite_losses,
    assert_tree_close,
    flattened_key_paths,
    learner_checkpoint_payload,
    make_sac_config,
    update_learner,
)


def test_one_env_runner_returns_a_complete_episode_and_applies_weights() -> None:
    config = make_sac_config()
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    learner_group = None

    try:
        episodes = runner.sample(num_episodes=1, random_actions=True)
        assert len(episodes) == 1
        assert len(episodes[0]) > 0
        assert episodes[0].is_done
        assert episodes[0].is_terminated or episodes[0].is_truncated
        assert runner.num_envs == 1
        assert config.num_envs_per_env_runner == 1

        module_spec = config.get_rl_module_spec(
            spaces=runner.get_spaces(),
            inference_only=True,
        )
        module = module_spec.build()
        assert module is not None

        learner_group = config.build_learner_group(spaces=runner.get_spaces())
        learner_module = learner_group.get_state(
            components="learner/rl_module",
            inference_only=True,
        )["learner"][COMPONENT_RL_MODULE]
        runner.set_state(
            {
                COMPONENT_RL_MODULE: learner_module,
                WEIGHTS_SEQ_NO: 1,
            }
        )

        next_episode = runner.sample(num_episodes=1)[0]
        versions = np.asarray(
            next_episode.get_extra_model_outputs(WEIGHTS_SEQ_NO),
            dtype=np.int64,
        )
        assert versions.size == len(next_episode)
        assert np.all(versions == 1)
    finally:
        if learner_group is not None:
            learner_group.shutdown()
        runner.stop()


def test_fixed_replay_sample_updates_sac_and_round_trips_full_state() -> None:
    config = make_sac_config(learner_class=TemperatureCheckpointSACLearner)
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    first = second = None

    try:
        collected = runner.sample(num_episodes=1, random_actions=True)
        replay = EpisodeReplayBuffer(capacity=1_024)
        replay.add(collected)
        replay.rng = np.random.default_rng(20260724)
        training_episodes = replay.sample(
            num_items=32,
            n_step=1,
            gamma=config.gamma,
            sample_episodes=True,
            to_numpy=True,
        )

        first = config.build_learner_group(spaces=runner.get_spaces())
        first_results = update_learner(first, training_episodes)
        assert_finite_losses(first_results)

        first_state = first.get_state()
        first_payload = learner_checkpoint_payload(first_state)
        assert any(
            "target" in path.lower()
            for path in flattened_key_paths(first_payload[COMPONENT_RL_MODULE])
        )
        assert any(
            "alpha" in path.lower()
            for path in flattened_key_paths(first_payload[COMPONENT_OPTIMIZER])
        )
        assert first_payload[SAC_TEMPERATURE_STATE]

        second = config.build_learner_group(spaces=runner.get_spaces())
        second.set_state(copy.deepcopy(first_state))
        assert_tree_close(
            first_payload,
            learner_checkpoint_payload(second.get_state()),
        )

        torch.manual_seed(20260724)
        update_learner(first, training_episodes)
        torch.manual_seed(20260724)
        update_learner(second, training_episodes)
        assert_tree_close(
            learner_checkpoint_payload(first.get_state()),
            learner_checkpoint_payload(second.get_state()),
        )
    finally:
        if first is not None:
            first.shutdown()
        if second is not None:
            second.shutdown()
        runner.stop()


def test_stock_rllib_state_does_not_contain_current_sac_temperature() -> None:
    config = make_sac_config(learner_class=SACTorchLearner)
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    source = restored = None

    try:
        source = config.build_learner_group(spaces=runner.get_spaces())
        source_log_alpha = source._learner.curr_log_alpha[DEFAULT_MODULE_ID]
        with torch.no_grad():
            source_log_alpha.fill_(0.75)

        state = source.get_state()
        assert SAC_TEMPERATURE_STATE not in state["learner"]
        assert COMPONENT_RL_MODULE in state["learner"]
        assert COMPONENT_OPTIMIZER in state["learner"]

        restored = config.build_learner_group(spaces=runner.get_spaces())
        restored.set_state(copy.deepcopy(state))
        restored_log_alpha = restored._learner.curr_log_alpha[DEFAULT_MODULE_ID]
        assert not torch.allclose(source_log_alpha, restored_log_alpha)
    finally:
        if source is not None:
            source.shutdown()
        if restored is not None:
            restored.shutdown()
        runner.stop()
