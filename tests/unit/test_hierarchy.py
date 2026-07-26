from __future__ import annotations

import pickle
import random
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from ray.rllib.core.columns import Columns

from rllib_async.examples import (
    HIERARCHY_MODULE_IDS,
    MANAGER_MODULE_ID,
    WORKER_0_MODULE_ID,
    WORKER_1_MODULE_ID,
    HierarchySwitchEnv,
)
from rllib_async.protocols import (
    EpisodeEnvelope,
    EpisodeValidationError,
    FrozenVersions,
    MultiModuleEpisodeCodec,
    MultiModuleTransition,
)
from rllib_async.replay import (
    EpisodeStore,
    FastReplay,
    MultiModuleBatchCollator,
    ReferenceFastReplay,
)


def transition(
    *,
    env_t: int,
    agent_t: int,
    agent_id: str,
    module_id: str,
    action: int | float = 0.0,
) -> MultiModuleTransition:
    action_array = (
        np.asarray(action, dtype=np.int64)
        if module_id == MANAGER_MODULE_ID
        else np.asarray([action], dtype=np.float32)
    )
    obs_size = 5 if module_id == MANAGER_MODULE_ID else 4
    return MultiModuleTransition(
        env_t=env_t,
        agent_t=agent_t,
        agent_id=agent_id,
        module_id=module_id,
        data={
            Columns.OBS: np.full(obs_size, env_t, dtype=np.float32),
            Columns.NEXT_OBS: np.full(obs_size, env_t + 1, dtype=np.float32),
            Columns.ACTIONS: action_array,
            Columns.REWARDS: float(env_t),
            Columns.TERMINATEDS: False,
            Columns.TRUNCATEDS: False,
        },
    )


def make_episode(
    codec: MultiModuleEpisodeCodec,
    sequence: int,
    transitions: list[MultiModuleTransition],
    *,
    env_steps: int,
) -> EpisodeEnvelope:
    payload = codec.encode(transitions)
    return EpisodeEnvelope(
        episode_id=f"member-0/runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions(
            {module_id: 3 for module_id in payload.module_ids}
        ),
        env_steps=env_steps,
        agent_steps=len(transitions),
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def test_hierarchy_env_exposes_only_manager_and_active_worker_turns() -> None:
    env = HierarchySwitchEnv(
        {
            "episode_length": 4,
            "manager_period": 2,
        }
    )

    observations, _ = env.reset(seed=20260726)
    assert set(observations) == {MANAGER_MODULE_ID, WORKER_0_MODULE_ID}

    observations, _, terminateds, _, _ = env.step(
        {
            MANAGER_MODULE_ID: 1,
            WORKER_0_MODULE_ID: np.asarray([0.25], dtype=np.float32),
        }
    )
    assert set(observations) == {WORKER_1_MODULE_ID}
    assert not terminateds["__all__"]

    observations, _, _, _, _ = env.step(
        {WORKER_1_MODULE_ID: np.asarray([-0.25], dtype=np.float32)}
    )
    assert set(observations) == {MANAGER_MODULE_ID, WORKER_1_MODULE_ID}

    observations, _, _, _, _ = env.step(
        {
            MANAGER_MODULE_ID: 0,
            WORKER_1_MODULE_ID: np.asarray([0.5], dtype=np.float32),
        }
    )
    assert set(observations) == {WORKER_0_MODULE_ID}

    with pytest.raises(ValueError, match="active action turn"):
        env.step(
            {
                WORKER_0_MODULE_ID: np.asarray([0.0], dtype=np.float32),
                WORKER_1_MODULE_ID: np.asarray([0.0], dtype=np.float32),
            }
        )


def test_hierarchy_env_rejects_invalid_manager_action_before_mutation() -> None:
    env = HierarchySwitchEnv({"episode_length": 4, "manager_period": 2})
    expected_env = HierarchySwitchEnv({"episode_length": 4, "manager_period": 2})
    env.reset(seed=20260726)
    expected_env.reset(seed=20260726)
    worker_action = np.asarray([0.25], dtype=np.float32)

    with pytest.raises(ValueError, match="manager action"):
        env.step(
            {
                MANAGER_MODULE_ID: 2,
                WORKER_0_MODULE_ID: worker_action,
            }
        )

    actual = env.step(
        {
            MANAGER_MODULE_ID: 1,
            WORKER_0_MODULE_ID: worker_action,
        }
    )
    expected = expected_env.step(
        {
            MANAGER_MODULE_ID: 1,
            WORKER_0_MODULE_ID: worker_action,
        }
    )
    assert actual[1:] == expected[1:]
    assert actual[0].keys() == expected[0].keys()
    for agent_id in actual[0]:
        np.testing.assert_array_equal(actual[0][agent_id], expected[0][agent_id])


def test_multi_module_codec_preserves_sparse_agent_timeline() -> None:
    codec = MultiModuleEpisodeCodec()
    transitions = [
        transition(
            env_t=0,
            agent_t=0,
            agent_id=MANAGER_MODULE_ID,
            module_id=MANAGER_MODULE_ID,
            action=1,
        ),
        transition(
            env_t=0,
            agent_t=0,
            agent_id=WORKER_0_MODULE_ID,
            module_id=WORKER_0_MODULE_ID,
        ),
        transition(
            env_t=1,
            agent_t=0,
            agent_id=WORKER_1_MODULE_ID,
            module_id=WORKER_1_MODULE_ID,
        ),
        transition(
            env_t=2,
            agent_t=1,
            agent_id=MANAGER_MODULE_ID,
            module_id=MANAGER_MODULE_ID,
            action=0,
        ),
        transition(
            env_t=2,
            agent_t=1,
            agent_id=WORKER_1_MODULE_ID,
            module_id=WORKER_1_MODULE_ID,
        ),
    ]
    episode = make_episode(codec, 0, transitions, env_steps=3)

    codec.validate(episode)

    assert codec.module_ids(episode) == HIERARCHY_MODULE_IDS
    assert codec.module_transition_count(episode, MANAGER_MODULE_ID) == 2
    assert codec.module_transition_count(episode, WORKER_0_MODULE_ID) == 1
    assert codec.module_transition_count(episode, WORKER_1_MODULE_ID) == 2
    restored = codec.get_module_transition(episode, WORKER_1_MODULE_ID, 1)
    assert restored.env_t == 2
    assert restored.agent_t == 1
    assert restored.agent_id == WORKER_1_MODULE_ID
    assert restored.module_id == WORKER_1_MODULE_ID
    with pytest.raises(IndexError):
        codec.get_transition(episode, -1)
    assert pickle.loads(pickle.dumps(episode)) == episode

    with pytest.raises(EpisodeValidationError, match="behavior version"):
        codec.validate(
            replace(
                episode,
                behavior_versions=FrozenVersions(
                    {
                        MANAGER_MODULE_ID: 3,
                        WORKER_0_MODULE_ID: 3,
                    }
                ),
            )
        )
    with pytest.raises(EpisodeValidationError, match="environment timestep"):
        make_episode(
            codec,
            1,
            [
                transition(
                    env_t=0,
                    agent_t=0,
                    agent_id=MANAGER_MODULE_ID,
                    module_id=MANAGER_MODULE_ID,
                ),
                transition(
                    env_t=2,
                    agent_t=1,
                    agent_id=MANAGER_MODULE_ID,
                    module_id=MANAGER_MODULE_ID,
                ),
            ],
            env_steps=3,
        )


def test_fast_replay_builds_uniform_module_specific_views() -> None:
    codec = MultiModuleEpisodeCodec()
    first = make_episode(
        codec,
        0,
        [
            transition(
                env_t=0,
                agent_t=0,
                agent_id=MANAGER_MODULE_ID,
                module_id=MANAGER_MODULE_ID,
            ),
            transition(
                env_t=0,
                agent_t=0,
                agent_id=WORKER_0_MODULE_ID,
                module_id=WORKER_0_MODULE_ID,
            ),
            transition(
                env_t=1,
                agent_t=0,
                agent_id=WORKER_1_MODULE_ID,
                module_id=WORKER_1_MODULE_ID,
            ),
        ],
        env_steps=2,
    )
    second = make_episode(
        codec,
        1,
        [
            transition(
                env_t=0,
                agent_t=0,
                agent_id=MANAGER_MODULE_ID,
                module_id=MANAGER_MODULE_ID,
            ),
            *[
                transition(
                    env_t=env_t,
                    agent_t=env_t,
                    agent_id=WORKER_0_MODULE_ID,
                    module_id=WORKER_0_MODULE_ID,
                )
                for env_t in range(4)
            ],
        ],
        env_steps=4,
    )
    store = EpisodeStore(
        codec,
        capacity_transitions=20,
        capacity_bytes=100_000,
        store_generation="module-views",
    )
    store.commit_episode(first)
    store.commit_episode(second)
    actual = FastReplay(codec)
    reference = ReferenceFastReplay(codec)
    snapshot = store.get_snapshot()
    actual.load_snapshot(snapshot)
    reference.load_snapshot(snapshot)

    assert actual.module_ids == reference.module_ids == HIERARCHY_MODULE_IDS
    assert dict(actual.module_transition_counts) == {
        MANAGER_MODULE_ID: 2,
        WORKER_0_MODULE_ID: 5,
        WORKER_1_MODULE_ID: 1,
    }
    assert actual.module_transition_counts == reference.module_transition_counts
    actual_rng = random.Random(20260726)
    reference_rng = random.Random(20260726)
    coordinates = actual.sample_module_coordinates(
        WORKER_0_MODULE_ID,
        20_000,
        rng=actual_rng,
    )
    assert coordinates == reference.sample_module_coordinates(
        WORKER_0_MODULE_ID,
        20_000,
        rng=reference_rng,
    )
    counts = Counter(coordinates)
    assert len(counts) == 5
    assert all(
        count / len(coordinates) == pytest.approx(0.2, abs=0.015)
        for count in counts.values()
    )
    assert all(
        item.module_id == MANAGER_MODULE_ID
        for item in actual.sample_module(
            MANAGER_MODULE_ID,
            100,
            rng=random.Random(7),
        )
    )


def test_fast_replay_rebuilds_module_views_after_delta_eviction() -> None:
    codec = MultiModuleEpisodeCodec()
    first = make_episode(
        codec,
        0,
        [
            transition(
                env_t=0,
                agent_t=0,
                agent_id=MANAGER_MODULE_ID,
                module_id=MANAGER_MODULE_ID,
            ),
            transition(
                env_t=0,
                agent_t=0,
                agent_id=WORKER_0_MODULE_ID,
                module_id=WORKER_0_MODULE_ID,
            ),
            transition(
                env_t=1,
                agent_t=0,
                agent_id=WORKER_1_MODULE_ID,
                module_id=WORKER_1_MODULE_ID,
            ),
        ],
        env_steps=2,
    )
    second = make_episode(
        codec,
        1,
        [
            transition(
                env_t=0,
                agent_t=0,
                agent_id=MANAGER_MODULE_ID,
                module_id=MANAGER_MODULE_ID,
            ),
            *[
                transition(
                    env_t=env_t,
                    agent_t=env_t,
                    agent_id=WORKER_0_MODULE_ID,
                    module_id=WORKER_0_MODULE_ID,
                )
                for env_t in range(4)
            ],
        ],
        env_steps=4,
    )
    store = EpisodeStore(
        codec,
        capacity_transitions=5,
        capacity_bytes=100_000,
        store_generation="module-delta-eviction",
    )
    store.commit_episode(first)
    snapshot = store.get_snapshot()
    actual = FastReplay(codec)
    reference = ReferenceFastReplay(codec)
    actual.load_snapshot(snapshot)
    reference.load_snapshot(snapshot)

    ack = store.commit_episode(second)
    assert ack.evicted_episode_ids == (first.episode_id,)
    delta = store.get_delta(snapshot.cursor, max_bytes=100_000)
    actual.apply_delta(delta)
    reference.apply_delta(delta)
    actual.wait_for_idle(timeout=5.0)
    assert actual.active_cursor == delta.next_cursor

    assert actual.episode_ids == reference.episode_ids == (second.episode_id,)
    assert actual.module_transition_counts == (
        (MANAGER_MODULE_ID, 1),
        (WORKER_0_MODULE_ID, 4),
    )
    assert actual.module_transition_counts == reference.module_transition_counts
    coordinates = actual.sample_module_coordinates(
        WORKER_0_MODULE_ID,
        100,
        rng=random.Random(20260726),
    )
    assert {episode_id for episode_id, _ in coordinates} == {second.episode_id}
    with pytest.raises(KeyError):
        actual.sample_module(
            WORKER_1_MODULE_ID,
            1,
            rng=random.Random(1),
        )
    actual.close()


def test_multi_module_collator_groups_and_strips_provenance() -> None:
    transitions = [
        transition(
            env_t=0,
            agent_t=0,
            agent_id=WORKER_0_MODULE_ID,
            module_id=WORKER_0_MODULE_ID,
            action=0.25,
        ),
        transition(
            env_t=0,
            agent_t=0,
            agent_id=MANAGER_MODULE_ID,
            module_id=MANAGER_MODULE_ID,
            action=1,
        ),
        transition(
            env_t=1,
            agent_t=0,
            agent_id=WORKER_1_MODULE_ID,
            module_id=WORKER_1_MODULE_ID,
            action=-0.25,
        ),
    ]

    batch = MultiModuleBatchCollator(
        expected_module_ids=HIERARCHY_MODULE_IDS,
    ).collate(transitions)

    assert tuple(batch) == HIERARCHY_MODULE_IDS
    assert batch[MANAGER_MODULE_ID][Columns.OBS].shape == (1, 5)
    assert batch[MANAGER_MODULE_ID][Columns.ACTIONS].shape == (1,)
    assert batch[WORKER_0_MODULE_ID][Columns.OBS].shape == (1, 4)
    assert batch[WORKER_1_MODULE_ID][Columns.ACTIONS].shape == (1, 1)
    assert all(
        set(module_batch)
        == {
            Columns.OBS,
            Columns.NEXT_OBS,
            Columns.ACTIONS,
            Columns.REWARDS,
            Columns.TERMINATEDS,
            Columns.TRUNCATEDS,
        }
        for module_batch in batch.values()
    )
