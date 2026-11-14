from __future__ import annotations

import copy
import random
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE
from ray.rllib.policy.policy import PolicySpec

import rllib_async.runtime.learner_host as host_module
from rllib_async.learner import SAC_TEMPERATURE_STATE
from rllib_async.protocols import EpisodeEnvelope
from rllib_async.protocols.episodes import (
    MultiModuleTransition,
    RawMultiModuleEpisodeCodec,
)
from rllib_async.replay.prioritized import PrioritizedReplayConfig
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
        member_id=kwargs.pop("member_id", "team"),
        publication_interval_updates=1,
        batch_size=2,
        batch_queue_capacity=0,
        batch_seed=kwargs.pop("batch_seed", 7),
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
            clock["optimizer_updates"] > 0 for clock in host._module_counters.values()
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


@pytest.mark.parametrize("prioritized", [False, True])
def test_mixed_checkpoint_replays_next_selection_and_optimizer_step(
    monkeypatch, prioritized
):
    monkeypatch.setattr(host_module.ray, "get", lambda value: value)
    codec = RawMultiModuleEpisodeCodec()
    store = EpisodeStore(codec, capacity_transitions=100, capacity_bytes=10_000_000)
    configs, spaces = team_configuration()
    per = {mid: PrioritizedReplayConfig() for mid in configs} if prioritized else {}
    source = make_host(configs, spaces, store, codec, per_configs=per)
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
        expected_priorities = {
            mid: sampler.get_state()
            for mid, sampler in source._prioritized_samplers.items()
        }
        restored = make_host(
            configs, spaces, store, codec, checkpoint_state=saved, per_configs=per
        )
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
        assert {
            mid: sampler.get_state()
            for mid, sampler in restored._prioritized_samplers.items()
        } == expected_priorities
    finally:
        source.stop()
        if restored is not None:
            restored.stop()


def test_ingestion_budget_uses_valid_positions_and_does_not_reset_on_eviction():
    host = object.__new__(MixedLearnerHost)
    host._configs = {
        "worker": type("Config", (), {"num_steps_sampled_before_learning_starts": 3})()
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


def test_per_residual_budget_returns_without_sampling_or_busy_loop():
    class BoundedClocks(dict):
        reads = 0

        def __getitem__(self, key):
            self.reads += 1
            assert self.reads <= 4, "residual budgets caused unbounded polling"
            return super().__getitem__(key)

    host = object.__new__(MixedLearnerHost)
    host._trainable_modules = ("manager", "worker")
    host._next_module = 0
    host._module_counters = BoundedClocks(
        {mid: {"ready": True, "budget": 1.0} for mid in host._trainable_modules}
    )
    host._sequence_configs = {
        mid: SequenceConfig(learning_length=2) for mid in host._trainable_modules
    }
    host._prioritized_samplers = {mid: object() for mid in host._trainable_modules}
    host._fast_replay = type(
        "Replay",
        (),
        {"module_transition_counts": (("manager", 10), ("worker", 10))},
    )()
    host._maybe_publish = lambda: None
    assert host._consume_ready(100) == (0, None)
    assert host._module_counters.reads == 2


def test_mixed_pbt_transfers_targets_and_alpha_with_fresh_optimizer_priorities_rng(
    monkeypatch,
):
    monkeypatch.setattr(host_module.ray, "get", lambda value: value)
    codec = RawMultiModuleEpisodeCodec()
    store = EpisodeStore(codec, capacity_transitions=100, capacity_bytes=10_000_000)
    configs, spaces = team_configuration()
    per = {mid: PrioritizedReplayConfig() for mid in configs}
    source = make_host(configs, spaces, store, codec, per_configs=per)
    recipient = None
    try:
        source.start()
        store.commit_episode(
            raw_episode(codec, number=0, counts={"manager": 6, "worker": 6})
        )
        sync(source)
        tick(source, max_updates=2)
        transferred = source.export_pbt_state()
        assert transferred.module_states["manager"].algorithm == "DQN"
        assert transferred.module_states["worker"].algorithm == "SAC"
        before = copy.deepcopy(source.get_published_weights())
        mismatched = replace(
            transferred,
            module_states={
                **transferred.module_states,
                "worker": replace(
                    transferred.module_states["worker"], config_contract={}
                ),
            },
        )
        with pytest.raises(ValueError, match="configuration"):
            source.validate_pbt_state(mismatched)
        assert_tree_close(source.get_published_weights().state, before.state)

        mutated = {
            mid: config.copy(copy_frozen=False) for mid, config in configs.items()
        }
        mutated["manager"].training(lr=0.002)
        mutated["worker"].training(actor_lr=0.002, critic_lr=0.002, alpha_lr=0.002)
        recipient = make_host(
            mutated,
            spaces,
            store,
            codec,
            per_configs=per,
            pbt_state=transferred,
            member_id="recipient",
            batch_seed=19,
        )
        assert recipient.get_published_weights().member_id == "recipient"
        assert recipient.get_published_weights().learner_updates == 0
        for index, mid in enumerate(sorted(configs)):
            donor = source._adapters[mid].get_state()["learner_group"]["learner"]
            received = recipient._adapters[mid].get_state()["learner_group"]["learner"]
            assert_tree_close(received[COMPONENT_RL_MODULE], donor[COMPONENT_RL_MODULE])
            for optimizer in received[COMPONENT_OPTIMIZER].values():
                assert not optimizer["state"]["state"]
            if mid == "worker":
                assert_tree_close(
                    received[SAC_TEMPERATURE_STATE], donor[SAC_TEMPERATURE_STATE]
                )
            clock = recipient._module_counters[mid]
            assert (
                clock["budget"]
                == clock["learned_positions"]
                == clock["optimizer_updates"]
                == 0
            )
            assert (
                recipient._samplers[mid].getstate()
                == random.Random(19 + index).getstate()
            )
            state = recipient._prioritized_samplers[mid].get_state()
            assert state["max_priority"] == 1.0
            assert all(entry is None or entry[2] == 1.0 for entry in state["entries"])
        recipient.start()
        assert tick(recipient, max_updates=100).updates_performed == 0
    finally:
        source.stop()
        if recipient is not None:
            recipient.stop()
