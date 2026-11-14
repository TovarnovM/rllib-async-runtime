from __future__ import annotations

import copy

import numpy as np
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

from rllib_async.learner.dqn_adapter import (
    CheckpointableDQNTorchLearner,
    prepare_dqn_config,
)
from tests.helpers import assert_finite_losses, make_dqn_config, update_learner


def test_dqn_env_runner_replay_learner_and_weight_publication() -> None:
    config = prepare_dqn_config(make_dqn_config())
    config.learners(learner_class=CheckpointableDQNTorchLearner)
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    learner = None
    try:
        episodes = runner.sample(num_episodes=1, random_actions=True)
        assert len(episodes) == 1 and episodes[0].is_done
        assert np.asarray(episodes[0].get_actions()).dtype.kind in "iu"
        replay = EpisodeReplayBuffer(capacity=128)
        replay.add(episodes)
        replay.rng = np.random.default_rng(19)
        training_episodes = replay.sample(
            num_items=8,
            n_step=1,
            gamma=config.gamma,
            sample_episodes=True,
            to_numpy=True,
        )
        learner = config.build_learner_group(spaces=runner.get_spaces())
        assert_finite_losses(update_learner(learner, training_episodes))
        weights = learner.get_state(
            components="learner/rl_module", inference_only=True
        )["learner"][COMPONENT_RL_MODULE]
        runner.set_state({COMPONENT_RL_MODULE: copy.deepcopy(weights), WEIGHTS_SEQ_NO: 1})
        next_episode = runner.sample(num_episodes=1)[0]
        versions = np.asarray(next_episode.get_extra_model_outputs(WEIGHTS_SEQ_NO))
        assert versions.size == len(next_episode)
        assert np.all(versions == 1)
    finally:
        if learner is not None:
            learner.shutdown()
        runner.stop()
