from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.protocols.replay import (
    ReplayCursor,
    ReplayDelta,
    ReplayTransaction,
)
from rllib_async.replay import FastReplay
from rllib_async.replay.reference import (
    CursorMismatchError,
    EpisodeStore,
    FullResyncRequiredError,
    ReferenceFastReplay,
    ReplayError,
)


def make_episode(
    codec: FlatEpisodeCodec,
    sequence: int,
    transitions: list[object],
) -> EpisodeEnvelope:
    payload = codec.encode(transitions)
    transition_count = len(payload.encoded_transitions)
    return EpisodeEnvelope(
        episode_id=f"member-0/runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({"default_policy": 5}),
        env_steps=transition_count,
        agent_steps=transition_count,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def assert_equivalent(
    actual: FastReplay,
    reference: ReferenceFastReplay,
    *,
    seed: int,
) -> None:
    assert actual.cursor == reference.cursor
    assert actual.episode_ids == reference.episode_ids
    assert actual.get_snapshot() == reference.get_snapshot()

    if actual.total_transitions:
        actual_rng = random.Random(seed)
        reference_rng = random.Random(seed)
        assert actual.sample_coordinates(
            100,
            rng=actual_rng,
        ) == reference.sample_coordinates(
            100,
            rng=reference_rng,
        )


def sync_once_or_resync(
    store: EpisodeStore,
    actual: FastReplay,
    reference: ReferenceFastReplay,
    *,
    max_bytes: int,
) -> None:
    assert actual.cursor == reference.cursor
    assert actual.cursor is not None
    delta = store.get_delta(actual.cursor, max_bytes=max_bytes)

    if delta.full_resync_required:
        actual_before = actual.get_snapshot()
        reference_before = reference.get_snapshot()
        with pytest.raises(FullResyncRequiredError):
            actual.apply_delta(delta)
        with pytest.raises(FullResyncRequiredError):
            reference.apply_delta(delta)
        assert actual.get_snapshot() == actual_before
        assert reference.get_snapshot() == reference_before

        snapshot = store.get_snapshot()
        actual.load_snapshot(snapshot)
        reference.load_snapshot(snapshot)
    else:
        actual.apply_delta(delta)
        reference.apply_delta(delta)


def test_snapshot_bootstrap_materializes_index_without_copying_payloads() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="snapshot-bootstrap",
    )
    store.commit_episode(make_episode(codec, 0, [0, 1]))
    store.commit_episode(make_episode(codec, 1, [2, 3, 4]))
    snapshot = store.get_snapshot()
    replay = FastReplay(codec)

    replay.load_snapshot(snapshot)

    materialized = replay.get_snapshot()
    assert materialized == snapshot
    assert replay.total_transitions == 5
    assert replay.total_estimated_bytes == snapshot.total_estimated_bytes
    assert all(
        local is authoritative
        for local, authoritative in zip(
            materialized.episodes,
            snapshot.episodes,
            strict=True,
        )
    )


def test_delta_swap_is_atomic_and_sampler_excludes_evicted_transitions() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=3,
        capacity_bytes=10_000,
        store_generation="atomic-swap",
    )
    first = make_episode(codec, 0, [("first", 0), ("first", 1)])
    second = make_episode(codec, 1, [("second", 0)])
    retained = make_episode(
        codec,
        2,
        [("retained", 0), ("retained", 1), ("retained", 2)],
    )
    store.commit_episode(first)
    store.commit_episode(second)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())

    store.commit_episode(retained)
    assert replay.cursor is not None
    delta = store.get_delta(replay.cursor, max_bytes=10_000)
    replay.apply_delta(delta)

    assert replay.get_snapshot() == store.get_snapshot()
    assert replay.episode_ids == (retained.episode_id,)
    assert set(replay.sample(1_000, rng=random.Random(20260724))) == {
        ("retained", 0),
        ("retained", 1),
        ("retained", 2),
    }


def test_invalid_delta_does_not_partially_replace_payload_or_sampling_index() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="invalid-delta",
    )
    original = make_episode(codec, 0, [("original", 0), ("original", 1)])
    store.commit_episode(original)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    before = replay.get_snapshot()
    before_samples = replay.sample(100, rng=random.Random(17))
    first = make_episode(codec, 1, [("new", 0)])
    second = make_episode(codec, 2, [("new", 1)])
    invalid = ReplayDelta(
        base_cursor=before.cursor,
        next_cursor=ReplayCursor("invalid-delta", before.cursor.mutation_seq + 2),
        transactions=(
            ReplayTransaction(before.cursor.mutation_seq + 1, first, ()),
            ReplayTransaction(
                before.cursor.mutation_seq + 2,
                second,
                ("unknown-episode",),
            ),
        ),
        full_resync_required=False,
        has_more=False,
    )

    with pytest.raises(ReplayError, match="unknown episode_id"):
        replay.apply_delta(invalid)

    assert replay.get_snapshot() == before
    assert replay.sample(100, rng=random.Random(17)) == before_samples


def test_non_fifo_eviction_is_rejected_without_mutation() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="non-fifo-delta",
    )
    first = make_episode(codec, 0, [0])
    second = make_episode(codec, 1, [1])
    added = make_episode(codec, 2, [2])
    store.commit_episode(first)
    store.commit_episode(second)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    before = replay.get_snapshot()
    invalid = ReplayDelta(
        base_cursor=before.cursor,
        next_cursor=ReplayCursor(
            "non-fifo-delta",
            before.cursor.mutation_seq + 1,
        ),
        transactions=(
            ReplayTransaction(
                before.cursor.mutation_seq + 1,
                added,
                (second.episode_id,),
            ),
        ),
        full_resync_required=False,
        has_more=False,
    )

    with pytest.raises(ReplayError, match="does not evict FIFO"):
        replay.apply_delta(invalid)
    assert replay.get_snapshot() == before


def test_invalid_snapshot_does_not_replace_existing_view() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="invalid-snapshot",
    )
    store.commit_episode(make_episode(codec, 0, [0, 1]))
    replay = FastReplay(codec)
    snapshot = store.get_snapshot()
    replay.load_snapshot(snapshot)

    with pytest.raises(ReplayError, match="transition total"):
        replay.load_snapshot(
            replace(snapshot, total_transitions=snapshot.total_transitions + 1)
        )

    assert replay.get_snapshot() == snapshot


def test_stale_cursor_requires_explicit_snapshot_resync() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        journal_capacity=1,
        store_generation="stale-resync",
    )
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    before = replay.get_snapshot()
    store.commit_episode(make_episode(codec, 0, [0]))
    store.commit_episode(make_episode(codec, 1, [1]))

    delta = store.get_delta(before.cursor, max_bytes=10_000)
    assert delta.full_resync_required
    with pytest.raises(FullResyncRequiredError):
        replay.apply_delta(delta)
    assert replay.get_snapshot() == before

    replay.load_snapshot(store.get_snapshot())
    assert replay.get_snapshot() == store.get_snapshot()


def test_delta_before_snapshot_requires_bootstrap() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="missing-bootstrap",
    )
    initial_cursor = store.cursor
    store.commit_episode(make_episode(codec, 0, [0]))
    delta = store.get_delta(initial_cursor, max_bytes=10_000)

    for replay in (FastReplay(codec), ReferenceFastReplay(codec)):
        with pytest.raises(
            CursorMismatchError,
            match="load a snapshot before applying deltas",
        ):
            replay.apply_delta(delta)
        assert replay.cursor is None
        assert replay.episode_ids == ()


def test_cursor_mismatch_is_rejected_without_mutation() -> None:
    codec = FlatEpisodeCodec()
    replay = FastReplay(codec)
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="cursor-mismatch",
    )
    replay.load_snapshot(store.get_snapshot())
    before = replay.get_snapshot()
    delta = ReplayDelta(
        base_cursor=ReplayCursor("foreign", 0),
        next_cursor=ReplayCursor("foreign", 0),
        transactions=(),
        full_resync_required=False,
        has_more=False,
    )

    with pytest.raises(CursorMismatchError):
        replay.apply_delta(delta)
    assert replay.get_snapshot() == before


@pytest.mark.parametrize("seed", range(20))
def test_randomized_fast_replay_matches_reference_across_deltas_and_resyncs(
    seed: int,
) -> None:
    rng = random.Random(seed)
    codec = FlatEpisodeCodec()
    capacity_transitions = rng.randint(4, 20)
    store = EpisodeStore(
        codec,
        capacity_transitions=capacity_transitions,
        capacity_bytes=100_000,
        journal_capacity=rng.randint(1, 8),
        store_generation=f"randomized-{seed}",
    )
    actual = FastReplay(codec)
    reference = ReferenceFastReplay(codec)
    initial = store.get_snapshot()
    actual.load_snapshot(initial)
    reference.load_snapshot(initial)

    for sequence in range(100):
        transition_count = rng.randint(1, min(7, capacity_transitions))
        episode = make_episode(
            codec,
            sequence,
            [
                {
                    "sequence": sequence,
                    "step": step,
                    "padding": "x" * rng.randint(0, 20),
                }
                for step in range(transition_count)
            ],
        )
        store.commit_episode(episode)
        if rng.random() < 0.2:
            assert store.commit_episode(episode).duplicate

        if rng.random() < 0.45:
            continue

        sync_once_or_resync(
            store,
            actual,
            reference,
            max_bytes=rng.randint(1, 300),
        )
        assert_equivalent(actual, reference, seed=seed * 10_000 + sequence)

    while actual.cursor != store.cursor:
        sync_once_or_resync(
            store,
            actual,
            reference,
            max_bytes=rng.randint(1, 300),
        )
    assert actual.get_snapshot() == store.get_snapshot()
    assert_equivalent(actual, reference, seed=seed)


def test_sampling_is_uniform_over_transitions_not_episodes() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=20,
        capacity_bytes=10_000,
        store_generation="fast-sampling",
    )
    short = make_episode(codec, 0, [("short", 0)])
    long = make_episode(codec, 1, [("long", index) for index in range(9)])
    store.commit_episode(short)
    store.commit_episode(long)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())

    coordinates = replay.sample_coordinates(50_000, rng=random.Random(20260724))
    counts = Counter(coordinates)

    assert len(counts) == 10
    assert all(
        count / len(coordinates) == pytest.approx(0.1, abs=0.01)
        for count in counts.values()
    )
