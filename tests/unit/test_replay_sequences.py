from __future__ import annotations

import pickle
import random
from dataclasses import replace

import numpy as np
import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    EpisodeValidationError,
    MultiModuleEpisodeCodec,
    MultiModuleTransition,
    RawMultiModuleEpisodeCodec,
    SchemaMismatchError,
)
from rllib_async.replay import EpisodeStore, FastReplay, ReferenceFastReplay
from rllib_async.replay.sequences import (
    SequenceBatchCollator,
    SequenceConfig,
    collate_sequences,
    make_sequence_window,
    prepare_sequence,
)


def raw_transition(
    index: int,
    *,
    agent_t: int | None = None,
    agent_id: str = "agent",
    module_id: str = "policy",
    world_t: int | None = None,
    rewards: tuple[float, ...] = (1.0,),
    reset: bool | None = None,
    terminated: bool = False,
    truncated: bool = False,
    subtask: bool = False,
    obs: object | None = None,
    next_obs: object | None = None,
) -> MultiModuleTransition:
    step = index if agent_t is None else agent_t
    return MultiModuleTransition(
        env_t=index,
        agent_t=step,
        agent_id=agent_id,
        module_id=module_id,
        data={
            "obs": np.asarray([step], dtype=np.float32) if obs is None else obs,
            "new_obs": np.asarray([step + 1], dtype=np.float32)
            if next_obs is None
            else next_obs,
            "actions": np.int64(step % 2),
            "rewards": float(sum(rewards)),
            "terminateds": terminated,
            "truncateds": truncated,
            "world_t": step if world_t is None else world_t,
            "duration": len(rewards),
            "raw_rewards": rewards,
            "state_reset": step == 0 if reset is None else reset,
            "subtask_terminated": subtask,
        },
    )


def raw_episode(
    rows: list[MultiModuleTransition], sequence: int = 0
) -> EpisodeEnvelope:
    codec = RawMultiModuleEpisodeCodec()
    payload = codec.encode(rows)
    return EpisodeEnvelope(
        episode_id=f"member/runner/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member",
        runner_id="runner",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions={module: 0 for module in payload.module_ids},
        env_steps=payload.env_steps,
        agent_steps=payload.transition_count,
        terminated=any(row.data["terminateds"] for row in rows),
        truncated=not any(row.data["terminateds"] for row in rows),
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def store_with(*episodes: EpisodeEnvelope, capacity: int = 100) -> EpisodeStore:
    store = EpisodeStore(
        RawMultiModuleEpisodeCodec(),
        capacity_transitions=capacity,
        capacity_bytes=10_000_000,
        journal_capacity=10,
    )
    for episode in episodes:
        store.commit_episode(episode)
    return store


def window_at(
    episode: EpisodeEnvelope,
    *,
    start: int = 0,
    length: int = 1,
    n_step: int = 1,
    burn_in: int | None = None,
    module_id: str = "policy",
    history: int = 0,
):
    codec = RawMultiModuleEpisodeCodec()
    return make_sequence_window(
        codec,
        episode,
        module_id,
        codec.module_histories(episode, module_id)[history],
        start,
        sequence_config=SequenceConfig(length, burn_in),
        n_step=n_step,
    )


@pytest.mark.parametrize("truncated", [False, True])
def test_raw_returns_use_physical_ticks_and_real_final_observation(truncated) -> None:
    episode = raw_episode(
        [
            raw_transition(0, world_t=0, rewards=(1.0, 2.0)),
            raw_transition(1, world_t=2, rewards=(3.0,)),
            raw_transition(
                2,
                world_t=3,
                rewards=(4.0, 5.0, 6.0),
                terminated=not truncated,
                truncated=truncated,
                next_obs=np.asarray([99.0]),
            ),
        ]
    )
    before = pickle.dumps(episode)
    window = window_at(episode, length=3, n_step=2)
    batch = collate_sequences([window], gamma=0.5, n_step=2)
    np.testing.assert_allclose(batch["rewards"], [2.75, 7.0, 8.0])
    np.testing.assert_array_equal(batch["n_step"], [3, 4, 3])
    np.testing.assert_array_equal(batch["decision_horizons"], [2, 2, 1])
    np.testing.assert_array_equal(
        batch["terminateds"], [False, not truncated, not truncated]
    )
    np.testing.assert_array_equal(batch["truncateds"], [False, truncated, truncated])
    np.testing.assert_array_equal(batch["new_obs"].reshape(-1), [2.0, 99.0, 99.0])
    # A second consumer chooses gamma independently without mutating raw replay.
    other = collate_sequences([window], gamma=1.0, n_step=2)
    np.testing.assert_allclose(other["rewards"], [6.0, 18.0, 15.0])
    assert pickle.dumps(episode) == before


def test_full_prefix_bounded_burn_in_and_short_tail_have_only_real_loss_rows() -> None:
    episode = raw_episode([raw_transition(i, terminated=i == 4) for i in range(5)])
    exact = window_at(episode, start=3, length=4, n_step=3)
    approximate = window_at(episode, start=3, length=4, n_step=3, burn_in=1)
    assert [row.agent_t for row in exact.prefix] == [0, 1, 2]
    assert [row.agent_t for row in approximate.prefix] == [2]
    assert not exact.approximate and approximate.approximate
    assert exact.valid_count == approximate.valid_count == 2
    assert exact.handle == approximate.handle == (episode.episode_id, "agent", 3)
    batch = collate_sequences([exact, approximate], gamma=0.9, n_step=3)
    assert len(batch["rewards"]) == 4
    np.testing.assert_array_equal(batch["recurrent"]["seq_lens"], [6, 4])
    np.testing.assert_array_equal(batch["recurrent"]["prefix_lens"], [3, 1])
    np.testing.assert_array_equal(batch["recurrent"]["current_indices"], [3, 4, 7, 8])
    np.testing.assert_array_equal(batch["recurrent"]["next_indices"], [5, 5, 9, 9])
    np.testing.assert_allclose(batch["rewards"], [1.9, 1.0, 1.9, 1.0])
    with pytest.raises(ValueError, match="target horizon"):
        collate_sequences([exact], gamma=0.9, n_step=2)


def test_sequences_never_cross_agents_episode_or_reset_boundaries() -> None:
    episode = raw_episode(
        [
            raw_transition(0, agent_t=0, agent_id="a"),
            raw_transition(1, agent_t=0, agent_id="b"),
            raw_transition(2, agent_t=1, agent_id="a", subtask=True),
            raw_transition(3, agent_t=1, agent_id="b", terminated=True),
            raw_transition(4, agent_t=2, agent_id="a", reset=True),
            raw_transition(5, agent_t=3, agent_id="a", terminated=True),
        ]
    )
    codec = RawMultiModuleEpisodeCodec()
    assert codec.module_histories(episode, "policy") == ((0, 2), (1, 3), (4, 5))
    second = raw_episode([raw_transition(0, terminated=True)], sequence=1)
    store = store_with(episode, second)
    reference = ReferenceFastReplay(codec)
    reference.load_snapshot(store.get_snapshot())
    with FastReplay(codec) as replay:
        replay.load_snapshot(store.get_snapshot())
        windows = replay.sample_sequences(
            "policy",
            100,
            rng=random.Random(74),
            sequence_config=SequenceConfig(3),
            n_step=3,
        )
        oracle = reference.sample_sequences(
            "policy",
            100,
            rng=random.Random(74),
            sequence_config=SequenceConfig(3),
            n_step=3,
        )
        assert [item.handle for item in windows] == [item.handle for item in oracle]
        for window in windows:
            rows = window.prefix + window.learning + window.continuation
            assert {row.agent_id for row in rows} == {window.agent_id}
            assert not any(row.data["state_reset"] for row in rows[1:])
            assert len(rows) <= 2
        assert {item.handle for item in windows} == {
            (episode.episode_id, "a", 0),
            (episode.episode_id, "a", 1),
            (episode.episode_id, "a", 2),
            (episode.episode_id, "a", 3),
            (episode.episode_id, "b", 0),
            (episode.episode_id, "b", 1),
            (second.episode_id, "agent", 0),
        }


def test_window_boundary_keeps_bootstrap_and_old_final_observation() -> None:
    episode = raw_episode(
        [
            raw_transition(0, next_obs=np.asarray([10.0])),
            raw_transition(1, reset=True, obs=np.asarray([20.0]), terminated=True),
        ]
    )
    window = window_at(episode, length=1, n_step=3)
    batch = collate_sequences([window], gamma=0.9, n_step=3)
    assert not batch["terminateds"][0]
    np.testing.assert_array_equal(batch["new_obs"], [[10.0]])
    np.testing.assert_array_equal(batch["n_step"], [1])


def test_subtask_terminal_disables_bootstrap_without_rewriting_world_terminal() -> None:
    episode = raw_episode([raw_transition(0, subtask=True)])
    window = window_at(episode)
    prepared = prepare_sequence(window, gamma=0.9)
    assert not window.learning[0].data["terminateds"]
    assert prepared.terminateds == (True,)
    assert not episode.terminated


def test_eviction_keeps_selected_window_readable_and_empty_delta_does_not_decode(
    monkeypatch,
) -> None:
    codec = RawMultiModuleEpisodeCodec()
    episode = raw_episode([raw_transition(0), raw_transition(1, terminated=True)])
    store = store_with(episode, capacity=2)
    with FastReplay(codec) as replay:
        replay.load_snapshot(store.get_snapshot())
        window = replay.sample_sequences("policy", 1, rng=random.Random(0))[0]
        assert window.episode is episode
        before = replay.get_stats().completed_rebuilds
        monkeypatch.setattr(
            codec,
            "module_histories",
            lambda *args: pytest.fail("empty delta rebuilt history"),
        )
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
        replay.wait_for_idle(timeout=2)
        assert replay.get_stats().completed_rebuilds == before
        monkeypatch.undo()
        store.commit_episode(
            raw_episode([raw_transition(0, terminated=True)], sequence=1)
        )
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
        replay.wait_for_idle(timeout=2)
        assert episode.episode_id not in replay.episode_ids
        assert window.episode is episode
        assert prepare_sequence(window, gamma=0.9).rewards == (1.0,)


def graph(nodes: int, features: int) -> dict[str, object]:
    return {
        "node_features": np.arange(nodes * features, dtype=np.float32).reshape(
            nodes, features
        ),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "controlled_node": 0,
    }


def test_one_payload_supports_two_graph_schemas_and_vector_observations() -> None:
    episode = raw_episode(
        [
            raw_transition(
                0,
                agent_t=0,
                agent_id="a",
                module_id="graph_a",
                obs=graph(2, 2),
                next_obs=graph(3, 2),
                terminated=True,
            ),
            raw_transition(
                1,
                agent_t=0,
                agent_id="b",
                module_id="graph_b",
                obs=graph(1, 4),
                next_obs=graph(2, 4),
                terminated=True,
            ),
            raw_transition(
                2,
                agent_t=0,
                agent_id="c",
                module_id="vector",
                obs=np.ones(5),
                next_obs=np.zeros(5),
                terminated=True,
            ),
        ]
    )
    original = pickle.dumps(episode)
    for module, shape in (("graph_a", (2, 2)), ("graph_b", (1, 4))):
        batch = collate_sequences(
            [window_at(episode, module_id=module)], gamma=0.95, validate=True
        )
        assert batch["obs"]["node_features"].shape == shape
        assert batch["recurrent"]["seq_lens"].tolist() == [2]
    vector = collate_sequences([window_at(episode, module_id="vector")], gamma=0.9)
    assert vector["obs"].shape == (1, 5)
    assert pickle.dumps(episode) == original
    with pytest.raises(ValueError, match="module semantics"):
        collate_sequences(
            [
                window_at(episode, module_id="graph_a"),
                window_at(episode, module_id="vector"),
            ],
            gamma=0.9,
        )


def test_raw_codec_removes_transport_padding_before_commit() -> None:
    padded = {
        "node_features": np.ones((1000, 2), dtype=np.float32),
        "edge_index": np.asarray([[0, 999], [1, 999]], dtype=np.int64),
        "controlled_node": 0,
        "node_count": np.int64(2),
        "edge_count": np.int64(1),
    }
    row = raw_transition(0, obs=padded, next_obs=padded, terminated=True)
    episode = raw_episode([row])
    decoded = RawMultiModuleEpisodeCodec().get_module_transition(episode, "policy", 0)
    assert decoded.data["obs"]["node_features"].shape == (2, 2)
    assert decoded.data["obs"]["edge_index"].shape == (2, 1)
    assert "node_count" not in decoded.data["obs"]
    assert "edge_count" not in decoded.data["obs"]
    assert episode.estimated_bytes < len(pickle.dumps(row.data)) // 2


def test_raw_startup_checks_retry_and_do_not_repeat_after_success(monkeypatch) -> None:
    from rllib_async.gnn import episodes as graph_episodes

    codec = RawMultiModuleEpisodeCodec(validate_startup=True)
    invalid = graph(2, 2)
    invalid["node_features"][0, 0] = np.nan
    with pytest.raises(EpisodeValidationError, match="finite"):
        codec.encode(
            [raw_transition(0, obs=invalid, next_obs=graph(2, 2), terminated=True)]
        )
    valid = raw_transition(0, obs=graph(2, 2), next_obs=graph(3, 2), terminated=True)
    codec.encode([valid])
    monkeypatch.setattr(
        graph_episodes,
        "normalize_graph_observation",
        lambda *args, **kwargs: pytest.fail("startup graph validation repeated"),
    )
    codec.encode([valid])


def test_startup_rejects_disconnected_observation_history() -> None:
    codec = RawMultiModuleEpisodeCodec(validate_startup=True)
    with pytest.raises(EpisodeValidationError, match="following decision"):
        codec.encode(
            [
                raw_transition(0, next_obs=np.asarray([42.0])),
                raw_transition(1, terminated=True),
            ]
        )
    codec.encode([raw_transition(0), raw_transition(1, terminated=True)])


def test_stateless_collation_does_not_build_recurrent_history() -> None:
    episode = raw_episode([raw_transition(0), raw_transition(1, terminated=True)])
    window = window_at(episode, length=1, n_step=2, burn_in=0)
    collator = SequenceBatchCollator(gamma=0.5, n_step=2, include_history=False)
    batch = collator.collate([window])
    assert "recurrent" not in batch
    np.testing.assert_allclose(batch["rewards"], [1.5])


def test_startup_validation_retries_failure_then_stops(monkeypatch) -> None:
    from rllib_async.replay import sequences

    episode = raw_episode([raw_transition(0, terminated=True)])
    window = window_at(episode)
    collator = SequenceBatchCollator(gamma=0.9, validate_startup=True)
    original = sequences._stack_observations
    validations = []

    def fail_once(*args, **kwargs):
        validations.append(kwargs["validate"])
        if len(validations) == 1:
            raise ValueError("startup failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(sequences, "_stack_observations", fail_once)
    with pytest.raises(ValueError, match="startup failure"):
        collator.collate([window])
    collator.collate([window])
    assert validations == [True, True, True, True]
    collator.collate([window])
    assert validations == [True, True, True, True, False, False, False]


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration", 0),
        ("duration", 2),
        ("raw_rewards", (float("nan"),)),
        ("raw_rewards", (1.0, 2.0)),
        ("rewards", 99.0),
        ("state_reset", False),
    ],
)
def test_raw_codec_rejects_incomplete_or_inconsistent_time_contract(
    field, value
) -> None:
    row = raw_transition(0, terminated=True)
    data = dict(row.data)
    data[field] = value
    with pytest.raises(EpisodeValidationError):
        RawMultiModuleEpisodeCodec().encode([replace(row, data=data)])


def test_legacy_sparse_data_cannot_silently_invent_raw_rewards_or_durations() -> None:
    row = raw_transition(0, terminated=True)
    legacy_data = {
        key: value
        for key, value in row.data.items()
        if key
        not in {
            "world_t",
            "duration",
            "raw_rewards",
            "state_reset",
            "subtask_terminated",
        }
    }
    with pytest.raises(EpisodeValidationError, match="cannot be migrated"):
        RawMultiModuleEpisodeCodec().encode([replace(row, data=legacy_data)])
    old = replace(raw_episode([row]), schema_version=1)
    with pytest.raises(SchemaMismatchError):
        RawMultiModuleEpisodeCodec().validate(old)
    with (
        FastReplay(MultiModuleEpisodeCodec()) as replay,
        pytest.raises(TypeError, match="raw physical-time"),
    ):
        replay.sample_sequences("policy", 1, rng=random.Random(0))


def test_raw_store_duplicate_and_restore_preserve_sequence_rng_selection() -> None:
    episode = raw_episode([raw_transition(i, terminated=i == 2) for i in range(3)])
    store = store_with(episode)
    before = store.get_snapshot()
    store.commit_episode(episode)
    assert store.get_snapshot() == before
    restored = EpisodeStore.from_state(
        RawMultiModuleEpisodeCodec(), pickle.loads(pickle.dumps(store.export_state()))
    )
    with (
        FastReplay(RawMultiModuleEpisodeCodec()) as left,
        FastReplay(RawMultiModuleEpisodeCodec()) as right,
    ):
        left.load_snapshot(store.get_snapshot())
        right.load_snapshot(restored.get_snapshot())
        rng = random.Random(9)
        left.sample_sequences("policy", 2, rng=rng)
        saved = rng.getstate()
        expected = left.sample_sequences("policy", 8, rng=rng)
        rng.setstate(saved)
        actual = right.sample_sequences("policy", 8, rng=rng)
        assert [item.handle for item in expected] == [item.handle for item in actual]
