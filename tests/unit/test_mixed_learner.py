from __future__ import annotations

import copy
from dataclasses import replace

import gymnasium as gym
import numpy as np
from ray.rllib.policy.policy import PolicySpec

import rllib_async.runtime.learner_host as host_module
from rllib_async.protocols import EpisodeEnvelope
from rllib_async.protocols.episodes import (
    MultiModuleTransition,
    RawMultiModuleEpisodeCodec,
)
from rllib_async.replay.reference import EpisodeStore
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime.config import AsyncRuntimeConfig
from rllib_async.runtime.mixed_learner_host import MixedLearnerHost
from rllib_async.runtime.team import prepare_team_config
from tests.helpers import assert_tree_close, make_dqn_config, make_sac_config
from tests.unit.test_learner_host import ImmediateReplayActor


def team_configuration(*, manager_warmup: int = 0):
    spaces = {
        "manager": (
            gym.spaces.Box(-10, 10, (4,), dtype=np.float32),
            gym.spaces.Discrete(2),
        ),
        "worker": (
            gym.spaces.Box(-10, 10, (3,), dtype=np.float32),
            gym.spaces.Box(-2, 2, (1,), dtype=np.float32),
        ),
    }
    rollout = make_sac_config().multi_agent(
        policies={
            mid: PolicySpec(observation_space=obs, action_space=action)
            for mid, (obs, action) in spaces.items()
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    configs = {
        "manager": make_dqn_config().training(
            train_batch_size_per_learner=2,
            num_steps_sampled_before_learning_starts=manager_warmup,
            training_intensity=1.0,
            n_step=2,
        ),
        "worker": make_sac_config().training(
            train_batch_size_per_learner=2,
            num_steps_sampled_before_learning_starts=0,
            training_intensity=1.0,
            n_step=2,
        ),
    }
    _, configs, spaces, _, _ = prepare_team_config(
        rollout, configs, AsyncRuntimeConfig(batch_queue_capacity=0)
    )
    return configs, spaces


def raw_episode(codec, *, number: int, counts: dict[str, int]) -> EpisodeEnvelope:
    rows = []
    for mid, count in counts.items():
        width = 4 if mid == "manager" else 3
        for index in range(count):
            duration = 2 if mid == "manager" else 1
            rows.append(
                MultiModuleTransition(
                    env_t=index,
                    agent_t=index,
                    agent_id=mid,
                    module_id=mid,
                    data={
                        "obs": np.full(width, index / 10, dtype=np.float32),
                        "new_obs": np.full(width, (index + 1) / 10, dtype=np.float32),
                        "actions": index % 2
                        if mid == "manager"
                        else np.asarray([0.0], dtype=np.float32),
                        "rewards": float(duration),
                        "terminateds": index == count - 1,
                        "truncateds": False,
                        "world_t": index * duration,
                        "duration": duration,
                        "raw_rewards": (1.0,) * duration,
                        "state_reset": index == 0,
                        "subtask_terminated": False,
                    },
                )
            )
    payload = codec.encode(rows)
    return EpisodeEnvelope(
        episode_id=f"team/runner/0/{number}",
        schema_version=codec.schema_version,
        producer_member_id="team",
        runner_id="runner",
        runner_generation=0,
        local_episode_seq=number,
        behavior_versions={mid: 0 for mid in counts},
        env_steps=payload.env_steps,
        agent_steps=payload.transition_count,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def make_host(configs, spaces, store, codec, **kwargs):
    return MixedLearnerHost(
        configs,
        spaces,
        ImmediateReplayActor(store),
        codec,
        member_id="team",
        publication_interval_updates=1,
        batch_size=2,
        batch_queue_capacity=0,
        batch_seed=7,
        replay_sync_max_bytes=1_000_000,
        sequence_configs={mid: SequenceConfig(learning_length=2) for mid in configs},
        **kwargs,
    )


def sync(host) -> None:
    host._sync_once()
    host._fast_replay.wait_for_idle(timeout=2)


def tick(host, max_updates=1):
    # These deliberately unrelated rollout counters cannot buy learner budget.
    return host.tick(
        sampled_env_steps=1_000_000,
        sampled_agent_steps=1_000_000,
        sampled_module_steps={"manager": 1_000_000, "worker": 1_000_000},
        max_updates=max_updates,
    )


def test_partial_readiness_fair_budget_duplicates_eviction_and_frozen_worker(
    monkeypatch,
):
    monkeypatch.setattr(host_module.ray, "get", lambda value: value)
    codec = RawMultiModuleEpisodeCodec()
    store = EpisodeStore(codec, capacity_transitions=8, capacity_bytes=10_000_000)
    configs, spaces = team_configuration(manager_warmup=3)
    host = make_host(configs, spaces, store, codec)
    try:
        host.start()
        episode = raw_episode(codec, number=0, counts={"manager": 2, "worker": 6})
        store.commit_episode(episode)
        sync(host)
        manager_before = copy.deepcopy(
            host._adapters["manager"].get_state()["learner_group"]
        )
        assert tick(host).updates_performed == 1
        assert host._module_counters["manager"]["optimizer_updates"] == 0
        assert_tree_close(
            host._adapters["manager"].get_state()["learner_group"], manager_before
        )
        before = copy.deepcopy(host._module_counters)
        store.commit_episode(episode)
        sync(host)
        assert host._module_counters == before
        store.commit_episode(
            raw_episode(codec, number=1, counts={"manager": 2, "worker": 6})
        )
        sync(host)
        assert host._module_counters["manager"]["ingested_decisions"] == 4
        assert host._module_counters["manager"]["budget"] == 1
        tick(host, max_updates=2)
        assert all(
            clock["optimizer_updates"] > 0
            for clock in host._module_counters.values()
        )
        while tick(host, max_updates=10).updates_performed:
            pass
        assert tick(host, max_updates=10).updates_performed == 0
        assert host._module_counters["manager"]["learned_positions"] == 1
        assert host._module_counters["worker"]["learned_positions"] == 12
        assert host._fast_replay.total_transitions == 8
        assert len(set(host.get_published_weights().module_versions.values())) == 1
    finally:
        host.stop()

    frozen = make_host(configs, spaces, store, codec, trainable_modules=("manager",))
    try:
        initial = copy.deepcopy(frozen._adapters["worker"].get_state()["learner_group"])
        frozen.start()
        # Initial snapshot establishes warm-up; it creates no historical debt.
        assert tick(frozen, max_updates=10).updates_performed == 0
        store.commit_episode(
            raw_episode(codec, number=2, counts={"manager": 4, "worker": 4})
        )
        sync(frozen)
        assert tick(frozen, max_updates=2).updates_performed > 0
        assert_tree_close(
            frozen._adapters["worker"].get_state()["learner_group"], initial
        )
        assert frozen._module_counters["worker"]["optimizer_updates"] == 0
    finally:
        frozen.stop()


def test_mixed_checkpoint_replays_next_selection_and_optimizer_step(monkeypatch):
    monkeypatch.setattr(host_module.ray, "get", lambda value: value)
    codec = RawMultiModuleEpisodeCodec()
    store = EpisodeStore(codec, capacity_transitions=100, capacity_bytes=10_000_000)
    configs, spaces = team_configuration()
    source = make_host(configs, spaces, store, codec)
    restored = None
    try:
        source.start()
        store.commit_episode(
            raw_episode(codec, number=0, counts={"manager": 6, "worker": 6})
        )
        sync(source)
        tick(source)
        source.drain(sampled_env_steps=12, sampled_agent_steps=12)
        saved = copy.deepcopy(source.get_checkpoint_state())
        source.resume()
        tick(source, max_updates=2)
        expected_counters = copy.deepcopy(source._module_counters)
        expected_rng = {mid: rng.getstate() for mid, rng in source._samplers.items()}
        expected_learners = {
            mid: adapter.get_state()["learner_group"]
            for mid, adapter in source._adapters.items()
        }
        restored = make_host(configs, spaces, store, codec, checkpoint_state=saved)
        assert restored._module_counters == saved["module_counters"]
        restored.start()
        tick(restored, max_updates=2)
        assert restored._module_counters == expected_counters
        assert {
            mid: rng.getstate() for mid, rng in restored._samplers.items()
        } == expected_rng
        for mid, adapter in restored._adapters.items():
            assert_tree_close(
                adapter.get_state()["learner_group"], expected_learners[mid]
            )
    finally:
        source.stop()
        if restored is not None:
            restored.stop()


def test_ingestion_budget_uses_valid_positions_and_does_not_reset_on_eviction():
    host = object.__new__(MixedLearnerHost)
    host._configs = {
        "worker": type(
            "Config", (), {"num_steps_sampled_before_learning_starts": 3}
        )()
    }
    host._intensities = {"worker": 2.5}
    host._trainable_modules = ("worker",)
    host._module_counters = {
        "worker": {
            "ingested_decisions": 0,
            "learned_positions": 0,
            "optimizer_updates": 0,
            "budget": 0.0,
            "ready": False,
        }
    }
    codec = RawMultiModuleEpisodeCodec()
    episode = raw_episode(codec, number=0, counts={"worker": 5})
    host._record_replay_ingestion((episode,))
    assert host._module_counters["worker"]["budget"] == 5.0
    host._record_replay_ingestion(
        (replace(episode, episode_id="another", local_episode_seq=1),)
    )
    assert host._module_counters["worker"]["ingested_decisions"] == 10
    assert host._module_counters["worker"]["budget"] == 17.5
