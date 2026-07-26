from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
import ray
from ray.rllib.core import COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

from rllib_async.protocols import FlatEpisodeCodec, WeightsDescriptor
from rllib_async.replay.actor import ReplayActor
from rllib_async.replay.reference import EpisodeStore
from rllib_async.rollout import AsyncRolloutGroup
from tests.helpers import make_sac_config


class VariableDelayOneStepEnv(gym.Env):
    metadata: ClassVar[dict[str, object]] = {}

    def __init__(self, env_context: Mapping[str, object]) -> None:
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        worker_index = int(getattr(env_context, "worker_index", 0))
        self._delay_s = 0.0 if worker_index == 1 else 0.35

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        time.sleep(self._delay_s)
        reward = float(self.np_random.random())
        return np.ones(3, dtype=np.float32), reward, True, False, {}


@ray.remote(num_cpus=0, max_concurrency=1)
class DelayedReplayActor:
    def __init__(self, codec: FlatEpisodeCodec, delay_s: float) -> None:
        self._delay_s = delay_s
        self._store = EpisodeStore(
            codec,
            capacity_transitions=1_000,
            capacity_bytes=10_000_000,
            journal_capacity=1_000,
            store_generation="delayed-rollout-test",
        )

    def commit_episode(self, episode):
        time.sleep(self._delay_s)
        return self._store.commit_episode(episode)

    def get_stats(self):
        return self._store.get_stats()


def make_rollout_config_and_weights() -> tuple[object, WeightsDescriptor]:
    config = (
        make_sac_config().environment(VariableDelayOneStepEnv).debugging(seed=20260725)
    )
    probe = SingleAgentEnvRunner(config=config, worker_index=0)
    try:
        module_state = probe.get_state(components=COMPONENT_RL_MODULE)[
            COMPONENT_RL_MODULE
        ]
    finally:
        probe.stop()
    weights = WeightsDescriptor(
        member_id="member-0",
        module_versions={DEFAULT_MODULE_ID: 0},
        learner_updates=0,
        published_at_monotonic=0.0,
        state={DEFAULT_MODULE_ID: copy.deepcopy(module_state)},
    )
    return config, weights


def wait_for_completions(
    group: AsyncRolloutGroup,
    *,
    count: int,
    timeout_s: float = 15.0,
) -> list:
    completions = []
    deadline = time.monotonic() + timeout_s
    while len(completions) < count and time.monotonic() < deadline:
        completions.extend(group.poll(timeout_s=0.5, max_events=1))
    assert len(completions) >= count
    return completions


@pytest.mark.integration
def test_async_rollout_has_no_episode_barrier_and_tracks_weight_lag(
    ray_runtime: None,
) -> None:
    config, initial = make_rollout_config_and_weights()
    newer = WeightsDescriptor(
        member_id=initial.member_id,
        module_versions={DEFAULT_MODULE_ID: 1},
        learner_updates=1,
        published_at_monotonic=1.0,
        state=copy.deepcopy(initial.state),
    )
    codec = FlatEpisodeCodec()
    replay = ReplayActor.options(num_cpus=0).remote(
        codec,
        capacity_transitions=1_000,
        capacity_bytes=10_000_000,
        journal_capacity=1_000,
        store_generation="rollout-async-test",
    )
    group = AsyncRolloutGroup(
        config,
        codec,
        replay,
        member_id="member-0",
        initial_weights=initial,
        runner_count=4,
        max_episode_steps=2,
        pending_commit_high_watermark=8,
        pending_commit_low_watermark=3,
        num_cpus_per_runner=0,
    )

    try:
        group.start()
        assert group.update_weights(newer)
        assert not group.update_weights(initial)

        first = wait_for_completions(group, count=1)[0]
        first_stats = group.get_stats()
        assert first.policy_version_lag == 1
        assert dict(first.episode.behavior_versions) == {DEFAULT_MODULE_ID: 0}
        assert (
            first_stats.pending_sample_calls > 0 or first_stats.episodes_collected < 4
        )

        completions = [first]
        deadline = time.monotonic() + 15.0
        while (
            not any(
                completion.episode.behavior_versions[DEFAULT_MODULE_ID] == 1
                for completion in completions
            )
            and time.monotonic() < deadline
        ):
            completions.extend(group.poll(timeout_s=0.5, max_events=1))
        assert any(
            completion.episode.behavior_versions[DEFAULT_MODULE_ID] == 1
            for completion in completions
        )

        duplicate = ray.get(replay.commit_episode.remote(first.episode))
        assert not duplicate.committed and duplicate.duplicate
        replay_stats = ray.get(replay.get_stats.remote())
        assert replay_stats.duplicate_commits == 1

        restarted_runner = first.episode.runner_id
        assert group.restart_runner(restarted_runner) == 1
        deadline = time.monotonic() + 15.0
        restarted = None
        while restarted is None and time.monotonic() < deadline:
            for completion in group.poll(timeout_s=0.5, max_events=1):
                if (
                    completion.episode.runner_id == restarted_runner
                    and completion.episode.runner_generation == 1
                ):
                    restarted = completion
                    break
        assert restarted is not None
        assert restarted.episode.local_episode_seq == 0
        assert restarted.episode.episode_id != first.episode.episode_id
        assert restarted.metrics.episode_return != first.metrics.episode_return

        stats = group.get_stats()
        assert stats.runner_restarts == 1
        assert stats.policy_version_lag_p95 >= 0.0
        assert stats.outstanding_high_watermark <= 8
    finally:
        group.stop()
        ray.kill(replay)


@pytest.mark.integration
def test_pending_commit_watermarks_reserve_slots_before_sampling(
    ray_runtime: None,
) -> None:
    config, initial = make_rollout_config_and_weights()
    codec = FlatEpisodeCodec()
    replay = DelayedReplayActor.remote(codec, 0.25)
    group = AsyncRolloutGroup(
        config,
        codec,
        replay,
        member_id="member-0",
        initial_weights=initial,
        runner_count=4,
        max_episode_steps=2,
        pending_commit_high_watermark=4,
        pending_commit_low_watermark=1,
        num_cpus_per_runner=0,
    )

    try:
        group.start()
        deadline = time.monotonic() + 15.0
        observed_backpressure = False
        resumed = False
        while time.monotonic() < deadline and not resumed:
            group.poll(timeout_s=0.5, max_events=4)
            stats = group.get_stats()
            outstanding = stats.pending_sample_calls + stats.pending_episode_commits
            assert outstanding <= 4
            assert stats.pending_episode_commits <= 4
            observed_backpressure |= stats.backpressure_events > 0
            resumed = observed_backpressure and stats.sample_calls_started > 4

        assert observed_backpressure
        assert resumed
        stats = group.get_stats()
        assert stats.outstanding_high_watermark == 4
        assert stats.backpressure_fraction > 0.0
    finally:
        group.stop()
        ray.kill(replay)
