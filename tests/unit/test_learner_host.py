from __future__ import annotations

import time

import numpy as np
import pytest
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

import rllib_async.runtime.learner_host as learner_host_module
from rllib_async.protocols import EpisodeEnvelope, FlatEpisodeCodec, FrozenVersions
from rllib_async.replay.reference import EpisodeStore
from rllib_async.runtime import LearnerHost, LearnerHostState
from tests.helpers import make_sac_config


class ImmediateRemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class ImmediateReplayActor:
    def __init__(self, store: EpisodeStore) -> None:
        self.get_snapshot = ImmediateRemoteMethod(store.get_snapshot)
        self.get_delta = ImmediateRemoteMethod(store.get_delta)


def test_learner_host_rejects_unconstructed_n_step_targets() -> None:
    config = make_sac_config()
    config.training(n_step=3)

    with pytest.raises(ValueError, match="n_step=1"):
        LearnerHost(
            config,
            {},
            object(),
            FlatEpisodeCodec(),
            member_id="member-0",
            publication_interval_updates=1,
            batch_size=8,
            batch_queue_capacity=1,
            batch_seed=7,
            replay_sync_max_bytes=1_000_000,
        )


def make_episode(codec: FlatEpisodeCodec, size: int = 64) -> EpisodeEnvelope:
    transitions = [
        {
            Columns.OBS: np.asarray(
                [index / size, 0.0, 0.0],
                dtype=np.float32,
            ),
            Columns.NEXT_OBS: np.asarray(
                [(index + 1) / size, 0.0, 0.0],
                dtype=np.float32,
            ),
            Columns.ACTIONS: np.asarray([0.0], dtype=np.float32),
            Columns.REWARDS: 1.0,
            Columns.TERMINATEDS: index == size - 1,
            Columns.TRUNCATEDS: False,
        }
        for index in range(size)
    ]
    payload = codec.encode(transitions)
    return EpisodeEnvelope(
        episode_id="member-0/runner-0/0/0",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=0,
        behavior_versions=FrozenVersions({DEFAULT_MODULE_ID: 0}),
        env_steps=size,
        agent_steps=size,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def test_one_sync_can_feed_multiple_local_sac_updates(monkeypatch) -> None:
    monkeypatch.setattr(learner_host_module.ray, "get", lambda value: value)
    config = make_sac_config()
    config.training(
        train_batch_size_per_learner=8,
        num_steps_sampled_before_learning_starts=0,
    )
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1_000,
        capacity_bytes=10_000_000,
    )
    host = LearnerHost(
        config,
        runner.get_spaces(),
        ImmediateReplayActor(store),
        codec,
        member_id="member-0",
        publication_interval_updates=2,
        batch_size=8,
        batch_queue_capacity=4,
        batch_seed=7,
        replay_sync_max_bytes=1_000_000,
    )
    try:
        host.start()
        store.commit_episode(make_episode(codec))
        first = host.tick(
            sampled_env_steps=64,
            sampled_agent_steps=64,
            max_updates=1,
        )
        assert first.synced_transactions == 1

        deadline = time.monotonic() + 5
        while (
            host.get_stats().batch_producer.queue_size < 3
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        before = host.get_stats()
        tick = host.tick(
            sampled_env_steps=64,
            sampled_agent_steps=64,
            max_updates=3,
        )
        after = host.get_stats()

        assert tick.updates_performed == 3
        assert after.sync_requests == before.sync_requests + 1
        assert after.learner_updates == before.learner_updates + 3
        assert tick.published_weights is not None
        assert tick.published_weights.module_versions[DEFAULT_MODULE_ID] >= 1
        assert after.fast_replay.total_transitions == 64

        host.pause(timeout_s=1)
        assert host.get_stats().state is LearnerHostState.PAUSED
        host.resume()
        assert host.get_stats().state is LearnerHostState.RUNNING
        host.drain(
            sampled_env_steps=64,
            sampled_agent_steps=64,
            timeout_s=2,
        )
        drained = host.get_stats()
        assert drained.state is LearnerHostState.PAUSED
        assert drained.batch_producer.batches_dropped == 0
    finally:
        host.stop(timeout_s=2)
        runner.stop()
