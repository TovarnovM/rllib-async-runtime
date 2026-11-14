from __future__ import annotations

import copy
import gc
import pickle
import random
import weakref
from dataclasses import fields, replace

import numpy as np
import pytest

from rllib_async.protocols.episodes import EpisodeEnvelope, RawMultiModuleEpisodeCodec
from rllib_async.replay import FastReplay
from rllib_async.replay.prioritized import (
    PrioritizedReplayConfig,
    PrioritizedSequenceSampler,
)
from rllib_async.replay.sequences import SequenceConfig
from tests.unit.test_replay_sequences import raw_episode, raw_transition, store_with


class FractionRandom(random.Random):
    def __init__(self, values):
        super().__init__(0)
        self._values = iter(values)

    def random(self):
        return next(self._values)


class TrackableEpisode(EpisodeEnvelope):
    """Allow a weak reference without changing the production payload layout."""


def episode(length: int, sequence: int = 0):
    return raw_episode(
        [raw_transition(i, terminated=i == length - 1) for i in range(length)],
        sequence=sequence,
    )


def test_priority_probabilities_and_weights_follow_window_start_measure() -> None:
    store = store_with(episode(3))
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(
            replay, "policy", sequence_config=SequenceConfig(learning_length=3),
            config=PrioritizedReplayConfig(alpha=1.0, beta=1.0, epsilon=1.0),
        )
        initial = sampler.sample(3, rng=FractionRandom([0.0, 0.4, 0.8]))
        assert sampler.update_priorities(initial.handles, [1.0, 3.0, 7.0]) == 3
        sampled = sampler.sample(3, rng=FractionRandom([1 / 14, 3 / 14, 10 / 14]))
        assert [window.learn_start for window in sampled.windows] == [0, 1, 2]
        np.testing.assert_allclose(sampled.probabilities, [2 / 14, 4 / 14, 8 / 14])
        np.testing.assert_allclose(sampled.window_weights, [1.0, 0.5, 0.25])
        np.testing.assert_allclose(sampled.weights, [1.0, 1.0, 1.0, 0.5, 0.5, 0.25])
        assert [window.valid_count for window in sampled.windows] == [3, 2, 1]


def test_alpha_zero_matches_uniform_oracle_and_rng_frontier_exactly() -> None:
    store = store_with(episode(2), episode(5, sequence=1))
    config = SequenceConfig(learning_length=3, burn_in=1)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(
            replay, "policy", sequence_config=config, n_step=2,
            config=PrioritizedReplayConfig(alpha=0.0),
        )
        first = sampler.sample(20, rng=random.Random(1))
        sampler.update_priorities(first.handles, list(range(20)))
        uniform_rng = random.Random(73)
        priority_rng = random.Random(73)
        expected = replay.sample_sequences(
            "policy", 64, rng=uniform_rng, sequence_config=config, n_step=2
        )
        actual = sampler.sample(64, rng=priority_rng)
        assert [window.handle for window in actual.windows] == [
            window.handle for window in expected
        ]
        assert uniform_rng.getstate() == priority_rng.getstate()
        np.testing.assert_allclose(actual.probabilities, 1 / 7)
        np.testing.assert_array_equal(actual.weights, np.ones(sum(
            window.valid_count for window in expected
        )))


def test_duplicate_feedback_max_abs_and_consumers_have_independent_priorities() -> None:
    source = episode(3)
    store = store_with(source)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        config = PrioritizedReplayConfig(alpha=1.0, epsilon=0.25)
        left = PrioritizedSequenceSampler(replay, "policy", config=config)
        right = PrioritizedSequenceSampler(replay, "policy", config=config)
        sample = left.sample(3, rng=FractionRandom([0.0, 0.4, 0.8]))
        other = right.sample(3, rng=FractionRandom([0.0, 0.4, 0.8]))
        assert all(window.episode is source for window in (*sample.windows, *other.windows))
        view = replay.get_sequence_view("policy")
        assert view is replay.get_sequence_view("policy")
        assert left.update_priorities(
            [sample.handles[0], sample.handles[0], sample.handles[1]],
            [2.0, -8.0, 1.0],
        ) == 2
        assert [entry[2] for entry in left.get_state()["entries"] if entry] == [8.25, 1.25, 1.0]
        assert [entry[2] for entry in right.get_state()["entries"] if entry] == [1.0, 1.0, 1.0]
        assert left.update_priorities([], []) == 0


def test_evicted_handle_cannot_modify_reused_slot_or_drop_sample_lease() -> None:
    source = episode(2)
    store = store_with(source, capacity=2)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy")
        selected = sampler.sample(1, rng=random.Random(0))
        store.commit_episode(episode(1, sequence=1))
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
        # Logical retention rejects this update even if the new view is not yet published.
        assert sampler.update_priorities(selected.handles, [1_000_000.0]) == 0
        replay.wait_for_idle(timeout=2)
        current = sampler.sample(1, rng=random.Random(0))
        assert current.handles[0].episode_id != selected.handles[0].episode_id
        before = sampler.get_state()
        assert sampler.update_priorities(selected.handles, [1_000_000.0]) == 0
        assert sampler.get_state() == before
        assert selected.windows[0].episode is source
        assert selected.windows[0].learning[0].data["raw_rewards"] == (1.0,)


def test_dormant_consumer_releases_evicted_payload_after_last_window_lease() -> None:
    original = episode(2)
    source = TrackableEpisode(
        **{field.name: getattr(original, field.name) for field in fields(original)}
    )
    retained = weakref.ref(source)
    store = store_with(source, capacity=2)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy")
        selected = sampler.sample(1, rng=random.Random(0)).windows[0]
        del source, original
        store.commit_episode(episode(1, sequence=1))
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
        replay.wait_for_idle(timeout=2)
        # The authoritative journal is an intentional independent retention owner.
        del store
        gc.collect()
        assert retained() is selected.episode
        del selected
        gc.collect()
        assert retained() is None
        assert sampler.sample(1, rng=random.Random(0)).windows[0].episode.episode_id.endswith("/1")


def test_generation_and_sequence_contract_reject_stale_priority_feedback() -> None:
    source = episode(2)
    store = store_with(source)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy")
        sample = sampler.sample(1, rng=random.Random(0))
        changed = replace(sample.handles[0], learning_length=2)
        assert sampler.update_priorities([changed], [100.0]) == 0
        replacement = store_with(source)
        replay.load_snapshot(replacement.get_snapshot())
        assert sampler.update_priorities(sample.handles, [100.0]) == 0
        current = sampler.sample(1, rng=random.Random(0))
        assert current.handles[0].store_generation != sample.handles[0].store_generation


def test_per_handles_separate_agent_instances_and_reset_histories() -> None:
    source = raw_episode([
        raw_transition(0, agent_t=0, agent_id="a", subtask=True),
        raw_transition(1, agent_t=0, agent_id="b", terminated=True),
        raw_transition(2, agent_t=1, agent_id="a", reset=True, terminated=True),
    ])
    store = store_with(source)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy", sequence_config=SequenceConfig(4))
        selected = sampler.sample(3, rng=FractionRandom([0.0, 0.4, 0.8]))
        assert {(handle.agent_id, handle.agent_t) for handle in selected.handles} == {
            ("a", 0), ("b", 0), ("a", 1)
        }
        assert all(window.valid_count == 1 for window in selected.windows)


def test_priority_checkpoint_restores_next_sample_and_future_slot_reuse() -> None:
    store = store_with(episode(2), episode(3, sequence=1), capacity=5)
    config = PrioritizedReplayConfig(alpha=0.8, beta=0.7)
    with FastReplay(RawMultiModuleEpisodeCodec()) as left, FastReplay(RawMultiModuleEpisodeCodec()) as right:
        left.load_snapshot(store.get_snapshot())
        first = PrioritizedSequenceSampler(left, "policy", config=config)
        selected = first.sample(5, rng=FractionRandom([0.0, 0.21, 0.41, 0.61, 0.81]))
        first.update_priorities(selected.handles, [1, 2, 3, 4, 5])
        store.commit_episode(episode(1, sequence=2))
        left.apply_delta(store.get_delta(left.cursor, max_bytes=1_000_000))
        left.wait_for_idle(timeout=2)
        state = pickle.loads(pickle.dumps(first.get_state()))
        assert state["entries"][1] is None  # A free slot must not be compacted on restore.
        right.load_snapshot(pickle.loads(pickle.dumps(store.get_snapshot())))
        restored = PrioritizedSequenceSampler(right, "policy", config=config)
        restored.set_state(state)
        assert restored.get_state() == state
        rng = random.Random(71)
        saved_rng = rng.getstate()
        expected = first.sample(50, rng=rng)
        rng.setstate(saved_rng)
        actual = restored.sample(50, rng=rng)
        assert actual.handles == expected.handles
        np.testing.assert_array_equal(actual.probabilities, expected.probabilities)
        np.testing.assert_array_equal(actual.weights, expected.weights)
        store.commit_episode(episode(2, sequence=3))
        for replay in (left, right):
            replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
            replay.wait_for_idle(timeout=2)
        assert first.get_state() == restored.get_state()
        assert first.sample(20, rng=random.Random(8)).handles == restored.sample(20, rng=random.Random(8)).handles


def test_checkpoint_rejects_bad_state_atomically_and_allows_explicit_newer_replay() -> None:
    store = store_with(episode(2), capacity=3)
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy")
        state = sampler.get_state()
        invalid = copy.deepcopy(state)
        invalid["max_priority"] = float("nan")
        with pytest.raises(ValueError, match="storage metadata"):
            sampler.set_state(invalid)
        assert sampler.get_state() == state
        store.commit_episode(episode(2, sequence=1))
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
        replay.wait_for_idle(timeout=2)
        with pytest.raises(ValueError, match="frontier"):
            sampler.set_state(state)
        sampler.set_state(state, allow_replay_ahead=True)
        assert sampler.sample_count == 2
        assert all(handle.episode_id.endswith("/1") for handle in sampler.sample(10, rng=random.Random(2)).handles)


def test_unchanged_and_empty_delta_sampling_do_not_rebuild_priority_metadata(monkeypatch) -> None:
    codec = RawMultiModuleEpisodeCodec()
    store = store_with(episode(3))
    with FastReplay(codec) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy")
        sampler.sample(1, rng=random.Random(0))
        monkeypatch.setattr(codec, "module_decision_ids", lambda *args: pytest.fail("metadata was rebuilt"))
        monkeypatch.setattr(sampler, "_handle", lambda *args: pytest.fail("priority index was rebuilt"))
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=1_000_000))
        replay.wait_for_idle(timeout=2)
        sampler.sample(10, rng=random.Random(0))


def test_nonfinite_priority_batch_is_rejected_without_partial_update() -> None:
    store = store_with(episode(2))
    with FastReplay(RawMultiModuleEpisodeCodec()) as replay:
        replay.load_snapshot(store.get_snapshot())
        sampler = PrioritizedSequenceSampler(replay, "policy")
        selected = sampler.sample(2, rng=FractionRandom([0.0, 0.8]))
        before = sampler.get_state()
        with pytest.raises(ValueError, match="finite"):
            sampler.update_priorities(selected.handles, [10.0, float("nan")])
        assert sampler.get_state() == before


@pytest.mark.parametrize("values", [
    {"alpha": -1.0}, {"alpha": float("nan")}, {"beta": 2.0},
    {"epsilon": 0.0}, {"epsilon": float("inf")}, {"alpha": True},
])
def test_priority_configuration_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError):
        PrioritizedReplayConfig(**values)
