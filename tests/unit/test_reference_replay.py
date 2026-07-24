from __future__ import annotations

import pickle
import random
from collections import Counter, OrderedDict
from dataclasses import replace

import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
    SchemaMismatchError,
)
from rllib_async.protocols.replay import (
    ReplayCursor,
    ReplayDelta,
    ReplayTransaction,
)
from rllib_async.replay.reference import (
    DuplicateEpisodeConflictError,
    EpisodeStore,
    EpisodeTooLargeError,
    FullResyncRequiredError,
    ReferenceFastReplay,
    ReplayError,
)


def make_episode(
    codec: FlatEpisodeCodec,
    sequence: int,
    transitions: list[object],
) -> EpisodeEnvelope:
    episode_id = f"member-0/runner-0/0/{sequence}"
    payload = codec.encode(transitions)
    count = len(payload.encoded_transitions)
    return EpisodeEnvelope(
        episode_id=episode_id,
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({"default_policy": 5}),
        env_steps=count,
        agent_steps=count,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def sync_all(
    store: EpisodeStore,
    local: ReferenceFastReplay,
    *,
    max_bytes: int = 1,
) -> None:
    while True:
        assert local.cursor is not None
        delta = store.get_delta(local.cursor, max_bytes=max_bytes)
        local.apply_delta(delta)
        if not delta.has_more:
            return


def test_duplicate_commit_is_idempotent_and_conflict_is_explicit() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="test",
    )
    episode = make_episode(codec, 0, [0, 1])

    first = store.commit_episode(episode)
    before = store.get_snapshot()
    duplicate = store.commit_episode(episode)

    assert first.committed and not first.duplicate
    assert not duplicate.committed and duplicate.duplicate
    assert duplicate.cursor == first.cursor
    assert store.get_snapshot() == before

    with pytest.raises(DuplicateEpisodeConflictError):
        store.commit_episode(replace(episode, runner_id="different-runner"))
    assert store.get_snapshot() == before
    stats = store.get_stats()
    assert stats.commit_attempts == 3
    assert stats.committed_episodes == 1
    assert stats.duplicate_commits == 1
    assert stats.rejected_commits == 1
    assert stats.conflicting_commits == 1
    assert stats.evicted_episodes == 0


def test_duplicate_commit_remains_idempotent_after_episode_eviction() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1,
        capacity_bytes=10_000,
        store_generation="evicted-dedup",
    )
    evicted = make_episode(codec, 0, [0])
    retained = make_episode(codec, 1, [1])

    store.commit_episode(evicted)
    store.commit_episode(retained)
    before_retry = store.get_snapshot()

    delayed_retry = pickle.loads(pickle.dumps(evicted))
    duplicate = store.commit_episode(delayed_retry)

    assert not duplicate.committed and duplicate.duplicate
    assert duplicate.cursor == before_retry.cursor
    assert duplicate.evicted_episode_ids == ()
    assert store.get_snapshot() == before_retry

    conflicting_retry = make_episode(codec, 0, ["different-content"])
    with pytest.raises(DuplicateEpisodeConflictError):
        store.commit_episode(conflicting_retry)
    assert store.get_snapshot() == before_retry


def test_oversize_and_schema_mismatch_leave_store_unchanged() -> None:
    codec = FlatEpisodeCodec()
    episode = make_episode(codec, 0, list(range(3)))
    byte_limited = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=episode.estimated_bytes - 1,
        store_generation="bytes",
    )
    transition_limited = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        store_generation="transitions",
    )

    with pytest.raises(EpisodeTooLargeError):
        byte_limited.commit_episode(episode)
    with pytest.raises(EpisodeTooLargeError):
        transition_limited.commit_episode(episode)
    with pytest.raises(SchemaMismatchError):
        transition_limited.commit_episode(replace(episode, schema_version=2))

    assert byte_limited.get_stats().episode_count == 0
    assert transition_limited.get_stats().episode_count == 0
    assert byte_limited.cursor.mutation_seq == 0
    assert transition_limited.cursor.mutation_seq == 0
    assert byte_limited.get_stats().rejected_commits == 1
    assert transition_limited.get_stats().rejected_commits == 2


def test_snapshot_plus_chunked_deltas_exactly_reconstructs_store() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=4,
        capacity_bytes=10_000,
        store_generation="snapshot-delta",
    )
    store.commit_episode(make_episode(codec, 0, [0, 1]))
    store.commit_episode(make_episode(codec, 1, [2, 3, 4]))

    local = ReferenceFastReplay(codec)
    local.load_snapshot(store.get_snapshot())

    store.commit_episode(make_episode(codec, 2, [5, 6, 7]))
    store.commit_episode(make_episode(codec, 3, [8]))
    sync_all(store, local, max_bytes=1)

    assert local.get_snapshot() == store.get_snapshot()
    assert local.episode_ids == (
        "member-0/runner-0/0/2",
        "member-0/runner-0/0/3",
    )


def test_compacted_or_foreign_cursor_requires_full_resync() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        journal_capacity=2,
        store_generation="current",
    )
    original_cursor = store.cursor
    for sequence in range(3):
        store.commit_episode(make_episode(codec, sequence, [sequence]))

    compacted = store.get_delta(original_cursor, max_bytes=10_000)
    foreign = store.get_delta(
        ReplayCursor("other-generation", store.cursor.mutation_seq),
        max_bytes=10_000,
    )

    assert compacted.full_resync_required
    assert foreign.full_resync_required

    local = ReferenceFastReplay(codec)
    local.load_snapshot(store.get_snapshot())
    with pytest.raises(FullResyncRequiredError):
        local.apply_delta(foreign)


def test_invalid_multi_transaction_delta_is_not_partially_applied() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="atomic-delta",
    )
    local = ReferenceFastReplay(codec)
    local.load_snapshot(store.get_snapshot())
    before = local.get_snapshot()
    first = make_episode(codec, 0, [0])
    second = make_episode(codec, 1, [1])
    delta = ReplayDelta(
        base_cursor=before.cursor,
        next_cursor=ReplayCursor("atomic-delta", 2),
        transactions=(
            ReplayTransaction(1, first, ()),
            ReplayTransaction(2, second, ("unknown-episode",)),
        ),
        full_resync_required=False,
        has_more=False,
    )

    with pytest.raises(ReplayError, match="unknown episode_id"):
        local.apply_delta(delta)
    assert local.get_snapshot() == before


@pytest.mark.parametrize("seed", range(20))
def test_randomized_add_evict_matches_a_simple_fifo_model(seed: int) -> None:
    rng = random.Random(seed)
    codec = FlatEpisodeCodec()
    capacity_transitions = rng.randint(4, 18)
    capacity_bytes = rng.randint(90, 360)
    store = EpisodeStore(
        codec,
        capacity_transitions=capacity_transitions,
        capacity_bytes=capacity_bytes,
        journal_capacity=512,
        store_generation=f"property-{seed}",
    )
    local = ReferenceFastReplay(codec)
    local.load_snapshot(store.get_snapshot())
    model: OrderedDict[str, EpisodeEnvelope] = OrderedDict()

    for sequence in range(60):
        transition_count = rng.randint(1, 7)
        transitions = [
            {
                "sequence": sequence,
                "step": step,
                "payload": "x" * rng.randint(0, 18),
            }
            for step in range(transition_count)
        ]
        episode = make_episode(codec, sequence, transitions)
        if (
            transition_count > capacity_transitions
            or episode.estimated_bytes > capacity_bytes
        ):
            with pytest.raises(EpisodeTooLargeError):
                store.commit_episode(episode)
        else:
            ack = store.commit_episode(episode)
            assert ack.committed
            model[episode.episode_id] = episode
            while (
                sum(item.agent_steps for item in model.values()) > capacity_transitions
                or sum(item.estimated_bytes for item in model.values()) > capacity_bytes
            ):
                model.popitem(last=False)

            if sequence % 9 == 0:
                duplicate = store.commit_episode(episode)
                assert duplicate.duplicate

        sync_all(store, local, max_bytes=rng.randint(1, 80))
        authoritative = store.get_snapshot()
        assert tuple(model.values()) == authoritative.episodes
        assert local.get_snapshot() == authoritative


def test_sampling_is_uniform_over_transitions_not_episodes() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=20,
        capacity_bytes=10_000,
        store_generation="sampling",
    )
    short = make_episode(codec, 0, [("short", 0)])
    long = make_episode(codec, 1, [("long", index) for index in range(9)])
    store.commit_episode(short)
    store.commit_episode(long)
    local = ReferenceFastReplay(codec)
    local.load_snapshot(store.get_snapshot())

    coordinates = local.sample_coordinates(50_000, rng=random.Random(20260724))
    counts = Counter(coordinates)

    short_fraction = counts[(short.episode_id, 0)] / len(coordinates)
    assert len(counts) == 10
    assert short_fraction == pytest.approx(0.1, abs=0.01)
    assert all(
        count / len(coordinates) == pytest.approx(0.1, abs=0.01)
        for count in counts.values()
    )


def test_sustained_ingest_bounds_payload_and_journal_but_counts_dedup_growth() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=64,
        capacity_bytes=100_000,
        journal_capacity=32,
        store_generation="sustained-ingest-reference",
    )

    for sequence in range(10_000):
        store.commit_episode(make_episode(codec, sequence, [sequence]))

    stats = store.get_stats()
    assert stats.episode_count == 64
    assert stats.total_transitions == 64
    assert stats.journal_entries == 32
    assert stats.deduplication_entries == 10_000
    assert stats.committed_episodes == 10_000
    assert stats.evicted_episodes == 10_000 - 64

    restored = EpisodeStore.from_state(codec, store.export_state())
    assert restored.get_snapshot() == store.get_snapshot()
    assert restored.get_stats() == stats
