from __future__ import annotations

import copy

import numpy as np
import pytest
from ray.rllib.core import COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

from rllib_async.protocols import FlatEpisodeCodec, WeightsDescriptor
from rllib_async.rollout import (
    EpisodeRunner,
    WeightVersionError,
    make_episode_id,
)
from tests.helpers import make_sac_config


def make_weights(version: int = 0) -> tuple[object, WeightsDescriptor]:
    config = make_sac_config()
    probe = SingleAgentEnvRunner(config=config, worker_index=0)
    try:
        module_state = probe.get_state(components=COMPONENT_RL_MODULE)[
            COMPONENT_RL_MODULE
        ]
    finally:
        probe.stop()
    return config, WeightsDescriptor(
        member_id="member-0",
        module_versions={DEFAULT_MODULE_ID: version},
        learner_updates=version,
        published_at_monotonic=float(version),
        state={DEFAULT_MODULE_ID: copy.deepcopy(module_state)},
    )


def test_episode_runner_collects_flat_time_limited_episode() -> None:
    config, initial = make_weights()
    codec = FlatEpisodeCodec()

    with EpisodeRunner(
        config,
        codec,
        member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        max_episode_steps=5,
        initial_weights=initial,
        worker_index=1,
    ) as runner:
        result = runner.collect_episode()

    episode = result.episode
    assert episode.episode_id == "member-0/runner-0/0/0"
    assert episode.env_steps == 5
    assert not episode.terminated and episode.truncated
    assert dict(episode.behavior_versions) == {DEFAULT_MODULE_ID: 0}
    assert result.metrics.env_steps == 5
    transitions = [
        codec.get_transition(episode, index) for index in range(episode.env_steps)
    ]
    assert set(transitions[0]) == {
        Columns.OBS,
        Columns.NEXT_OBS,
        Columns.ACTIONS,
        Columns.REWARDS,
        Columns.TERMINATEDS,
        Columns.TRUNCATEDS,
    }
    assert all(not transition[Columns.TERMINATEDS] for transition in transitions)
    assert all(not transition[Columns.TRUNCATEDS] for transition in transitions[:-1])
    assert transitions[-1][Columns.TRUNCATEDS]


def test_weights_change_only_between_episodes_and_ignore_stale_publication() -> None:
    config, initial = make_weights()
    newer = WeightsDescriptor(
        member_id=initial.member_id,
        module_versions={DEFAULT_MODULE_ID: 1},
        learner_updates=1,
        published_at_monotonic=1.0,
        state=copy.deepcopy(initial.state),
    )
    codec = FlatEpisodeCodec()

    with EpisodeRunner(
        config,
        codec,
        member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        max_episode_steps=3,
        initial_weights=initial,
        worker_index=1,
    ) as runner:
        first = runner.collect_episode()
        second = runner.collect_episode(newer)
        assert not runner.install_weights(initial)
        third = runner.collect_episode()

    assert first.episode.local_episode_seq == 0
    assert second.episode.local_episode_seq == 1
    assert third.episode.local_episode_seq == 2
    assert dict(first.episode.behavior_versions) == {DEFAULT_MODULE_ID: 0}
    assert dict(second.episode.behavior_versions) == {DEFAULT_MODULE_ID: 1}
    assert dict(third.episode.behavior_versions) == {DEFAULT_MODULE_ID: 1}


def test_equal_version_cannot_identify_a_different_publication() -> None:
    config, initial = make_weights()
    conflicting = WeightsDescriptor(
        member_id=initial.member_id,
        module_versions=initial.module_versions,
        learner_updates=initial.learner_updates + 1,
        published_at_monotonic=1.0,
        state=copy.deepcopy(initial.state),
    )

    with (
        EpisodeRunner(
            config,
            FlatEpisodeCodec(),
            member_id="member-0",
            runner_id="runner-0",
            runner_generation=0,
            max_episode_steps=2,
            initial_weights=initial,
            worker_index=1,
        ) as runner,
        pytest.raises(WeightVersionError, match="multiple publications"),
    ):
        runner.install_weights(conflicting)


def test_generation_changes_identity_while_sequence_restarts() -> None:
    assert make_episode_id("member-0", "runner-0", 0, 0) != make_episode_id(
        "member-0",
        "runner-0",
        1,
        0,
    )
    with pytest.raises(ValueError, match="path segment"):
        make_episode_id("member/0", "runner-0", 0, 0)


def test_rollout_metrics_reject_non_finite_return() -> None:
    from rllib_async.rollout import EpisodeRolloutMetrics

    with pytest.raises(ValueError, match="episode_return"):
        EpisodeRolloutMetrics(
            episode_time_s=0.0,
            episode_return=np.nan,
            env_steps=1,
            agent_steps=1,
        )
