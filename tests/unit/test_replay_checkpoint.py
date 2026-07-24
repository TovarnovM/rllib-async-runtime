from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.replay.checkpoint import (
    InvalidReplayCheckpointError,
    read_replay_checkpoint,
    write_replay_checkpoint,
)
from rllib_async.replay.reference import (
    EpisodeStore,
    InvalidEpisodeStoreStateError,
)


def make_episode(
    codec: FlatEpisodeCodec,
    sequence: int,
    value: object,
) -> EpisodeEnvelope:
    payload = codec.encode([value])
    return EpisodeEnvelope(
        episode_id=f"member-0/runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({"default_policy": 5}),
        env_steps=1,
        agent_steps=1,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def test_store_state_and_checkpoint_round_trip_preserve_all_authoritative_data(
    tmp_path: Path,
) -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        journal_capacity=2,
        store_generation="checkpoint-round-trip",
    )
    episodes = [make_episode(codec, index, index) for index in range(4)]
    store.commit_episode(episodes[0])
    cursor_before_suffix = store.cursor
    store.commit_episode(episodes[1])
    store.commit_episode(episodes[2])
    store.commit_episode(episodes[0])
    checkpoint_path = tmp_path / "replay.snapshot"

    checkpoint = write_replay_checkpoint(checkpoint_path, store.export_state())
    restored = EpisodeStore.from_state(
        codec,
        read_replay_checkpoint(checkpoint_path),
    )

    assert checkpoint.cursor == store.cursor
    assert checkpoint.size_bytes == checkpoint_path.stat().st_size
    assert len(checkpoint.sha256) == 64
    assert restored.get_snapshot() == store.get_snapshot()
    assert restored.get_stats() == store.get_stats()
    assert restored.get_delta(
        cursor_before_suffix,
        max_bytes=10_000,
    ) == store.get_delta(
        cursor_before_suffix,
        max_bytes=10_000,
    )

    delayed_retry = restored.commit_episode(episodes[0])
    assert delayed_retry.duplicate and not delayed_retry.committed
    assert restored.get_snapshot() == store.get_snapshot()


def test_checksum_failure_is_explicit(tmp_path: Path) -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        store_generation="checksum",
    )
    store.commit_episode(make_episode(codec, 0, "payload"))
    checkpoint_path = tmp_path / "replay.snapshot"
    write_replay_checkpoint(checkpoint_path, store.export_state())
    corrupted = bytearray(checkpoint_path.read_bytes())
    corrupted[-1] ^= 0x01
    checkpoint_path.write_bytes(corrupted)

    with pytest.raises(InvalidReplayCheckpointError, match="checksum"):
        read_replay_checkpoint(checkpoint_path)


def test_semantically_invalid_but_checksummed_state_is_rejected(
    tmp_path: Path,
) -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        store_generation="semantic-validation",
    )
    store.commit_episode(make_episode(codec, 0, "payload"))
    invalid_state = replace(store.export_state(), mutation_seq=2)
    checkpoint_path = tmp_path / "replay.snapshot"
    write_replay_checkpoint(checkpoint_path, invalid_state)

    state = read_replay_checkpoint(checkpoint_path)
    with pytest.raises(InvalidEpisodeStoreStateError, match="journal length"):
        EpisodeStore.from_state(codec, state)


def test_checksummed_state_with_incorrect_journal_eviction_is_rejected(
    tmp_path: Path,
) -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1,
        capacity_bytes=10_000,
        journal_capacity=1,
        store_generation="invalid-journal-eviction",
    )
    first = make_episode(codec, 0, "first")
    second = make_episode(codec, 1, "second")
    store.commit_episode(first)
    store.commit_episode(second)

    state = store.export_state()
    corrupted_transaction = replace(
        state.journal[-1],
        evicted_episode_ids=(second.episode_id,),
    )
    invalid_state = replace(
        state,
        journal=(*state.journal[:-1], corrupted_transaction),
    )
    checkpoint_path = tmp_path / "replay.snapshot"
    write_replay_checkpoint(checkpoint_path, invalid_state)

    checksummed_state = read_replay_checkpoint(checkpoint_path)
    with pytest.raises(
        InvalidEpisodeStoreStateError,
        match="journal eviction",
    ):
        EpisodeStore.from_state(codec, checksummed_state)


def test_checksummed_state_with_incomplete_journal_base_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1,
        capacity_bytes=10_000,
        journal_capacity=1,
        store_generation="incomplete-journal-base-manifest",
    )
    first = make_episode(codec, 0, "first")
    second = make_episode(codec, 1, "second")
    store.commit_episode(first)
    store.commit_episode(second)

    state = store.export_state()
    invalid_state = replace(
        state,
        journal_base_manifest=(),
        journal=(
            replace(
                state.journal[0],
                evicted_episode_ids=(),
            ),
        ),
    )
    checkpoint_path = tmp_path / "replay.snapshot"
    write_replay_checkpoint(checkpoint_path, invalid_state)

    checksummed_state = read_replay_checkpoint(checkpoint_path)
    with pytest.raises(
        InvalidEpisodeStoreStateError,
        match="journal base manifest",
    ):
        EpisodeStore.from_state(codec, checksummed_state)


def test_nonempty_incomplete_journal_base_manifest_is_rejected() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        journal_capacity=1,
        store_generation="nonempty-incomplete-journal-base-manifest",
    )
    episodes = [make_episode(codec, index, index) for index in range(3)]
    for episode in episodes:
        store.commit_episode(episode)

    state = store.export_state()
    invalid_state = replace(
        state,
        journal_base_manifest=state.journal_base_manifest[1:],
        journal=(
            replace(
                state.journal[0],
                evicted_episode_ids=(),
            ),
        ),
    )

    with pytest.raises(
        InvalidEpisodeStoreStateError,
        match="journal base manifest",
    ):
        EpisodeStore.from_state(codec, invalid_state)


def test_journal_cannot_move_an_eviction_between_transactions() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        journal_capacity=3,
        store_generation="invalid-eviction-timing",
    )
    episodes = [make_episode(codec, index, index) for index in range(3)]
    for episode in episodes:
        store.commit_episode(episode)

    state = store.export_state()
    invalid_state = replace(
        state,
        journal=(
            state.journal[0],
            replace(
                state.journal[1],
                evicted_episode_ids=(episodes[0].episode_id,),
            ),
            replace(state.journal[2], evicted_episode_ids=()),
        ),
    )

    with pytest.raises(
        InvalidEpisodeStoreStateError,
        match="journal eviction",
    ):
        EpisodeStore.from_state(codec, invalid_state)
