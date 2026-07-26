from __future__ import annotations

import shutil
from dataclasses import replace

import pytest

from rllib_async.protocols import FlatEpisodeCodec, ReplayCursor
from rllib_async.replay.checkpoint import write_replay_checkpoint
from rllib_async.replay.reference import EpisodeStore
from rllib_async.runtime.checkpoint import (
    POPULATION_CHECKPOINT_FILENAME,
    POPULATION_MEMBERS_DIRECTORY,
    RUNTIME_CHECKPOINT_FILENAME,
    RUNTIME_CHECKPOINT_STATE_VERSION,
    RUNTIME_REPLAY_FILENAME,
    InvalidPopulationCheckpointError,
    InvalidRuntimeCheckpointError,
    RuntimeCheckpointState,
    read_population_checkpoint_bundle,
    read_runtime_checkpoint,
    write_population_checkpoint,
    write_runtime_checkpoint,
)


def make_checkpoint_state(
    checkpoint_dir,
    *,
    store_generation: str = "runtime-checkpoint",
) -> RuntimeCheckpointState:
    store = EpisodeStore(
        FlatEpisodeCodec(),
        capacity_transitions=100,
        capacity_bytes=100_000,
        journal_capacity=16,
        store_generation=store_generation,
    )
    replay = write_replay_checkpoint(
        checkpoint_dir / RUNTIME_REPLAY_FILENAME,
        store.export_state(),
    )
    return RuntimeCheckpointState(
        state_version=RUNTIME_CHECKPOINT_STATE_VERSION,
        member_id="member-0",
        runtime_config={"runner_count": 4},
        replay_file=RUNTIME_REPLAY_FILENAME,
        replay_cursor=replay.cursor,
        learner=b"opaque learner state",
        rollout={"runner_generations": {"runner-0": 2}},
        evaluation=None,
        controller={"checkpoint_sequence": 1},
    )


def test_runtime_checkpoint_round_trip_is_relocatable(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    state = make_checkpoint_state(source)

    checkpoint = write_runtime_checkpoint(source, state)
    restored = read_runtime_checkpoint(source)

    assert restored == state
    assert checkpoint.replay_cursor == state.replay_cursor
    assert checkpoint.size_bytes > 0
    assert len(checkpoint.sha256) == 64
    assert checkpoint.directory == str(source)

    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    assert read_runtime_checkpoint(relocated) == state


def test_runtime_checkpoint_rejects_corruption_and_replay_mismatch(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    state = make_checkpoint_state(checkpoint_dir)
    write_runtime_checkpoint(checkpoint_dir, state)

    member_path = checkpoint_dir / "member.snapshot"
    encoded = bytearray(member_path.read_bytes())
    encoded[-1] ^= 0x01
    member_path.write_bytes(encoded)
    with pytest.raises(InvalidRuntimeCheckpointError, match="checksum"):
        read_runtime_checkpoint(checkpoint_dir)

    write_runtime_checkpoint(checkpoint_dir, state)
    different_store = EpisodeStore(
        FlatEpisodeCodec(),
        capacity_transitions=100,
        capacity_bytes=100_000,
        store_generation="different-replay",
    )
    write_replay_checkpoint(
        checkpoint_dir / RUNTIME_REPLAY_FILENAME,
        different_store.export_state(),
    )
    with pytest.raises(InvalidRuntimeCheckpointError, match="cursors"):
        read_runtime_checkpoint(checkpoint_dir)


def test_runtime_checkpoint_rejects_non_relocatable_replay_path(tmp_path) -> None:
    state = make_checkpoint_state(tmp_path)
    invalid = replace(state, replay_file="/absolute/replay.snapshot")

    with pytest.raises(InvalidRuntimeCheckpointError, match="relocatable"):
        write_runtime_checkpoint(tmp_path, invalid)

    assert state.replay_cursor == ReplayCursor("runtime-checkpoint", 0)


def test_population_checkpoint_persists_shared_replay_exactly_once(tmp_path) -> None:
    checkpoint_dir = tmp_path / "population"
    checkpoint_dir.mkdir()
    store = EpisodeStore(
        FlatEpisodeCodec(),
        capacity_transitions=100,
        capacity_bytes=100_000,
        store_generation="population-checkpoint",
    )
    cursor = store.cursor
    member_zero = RuntimeCheckpointState(
        state_version=RUNTIME_CHECKPOINT_STATE_VERSION,
        member_id="member-0",
        runtime_config={"member_id": "member-0"},
        replay_file=RUNTIME_REPLAY_FILENAME,
        replay_cursor=cursor,
        learner=b"member zero",
        rollout={"runner_generations": {}},
        evaluation=None,
        controller={"checkpoint_sequence": 1},
    )
    member_one = replace(
        member_zero,
        member_id="member-1",
        runtime_config={"member_id": "member-1"},
        learner=b"member one",
    )

    checkpoint = write_population_checkpoint(
        checkpoint_dir,
        replay_state=store.export_state(),
        members={
            "member-0": member_zero,
            "member-1": member_one,
        },
    )
    manifest, replay, members = read_population_checkpoint_bundle(checkpoint_dir)

    assert checkpoint.member_ids == ("member-0", "member-1")
    assert checkpoint.replay_cursor == cursor
    assert manifest.replay_cursor == cursor
    assert replay == store.export_state()
    assert members == {
        "member-0": member_zero,
        "member-1": member_one,
    }
    assert (checkpoint_dir / POPULATION_CHECKPOINT_FILENAME).is_file()
    assert (checkpoint_dir / RUNTIME_REPLAY_FILENAME).is_file()
    for member_id in checkpoint.member_ids:
        member_directory = checkpoint_dir / POPULATION_MEMBERS_DIRECTORY / member_id
        assert (member_directory / RUNTIME_CHECKPOINT_FILENAME).is_file()
        assert not (member_directory / RUNTIME_REPLAY_FILENAME).exists()


def test_population_checkpoint_rejects_newer_member_and_corruption(tmp_path) -> None:
    checkpoint_dir = tmp_path / "population"
    checkpoint_dir.mkdir()
    store = EpisodeStore(
        FlatEpisodeCodec(),
        capacity_transitions=100,
        capacity_bytes=100_000,
        store_generation="population-checkpoint",
    )
    member = RuntimeCheckpointState(
        state_version=RUNTIME_CHECKPOINT_STATE_VERSION,
        member_id="member-0",
        runtime_config={"member_id": "member-0"},
        replay_file=RUNTIME_REPLAY_FILENAME,
        replay_cursor=ReplayCursor("population-checkpoint", 1),
        learner=b"member zero",
        rollout={"runner_generations": {}},
        evaluation=None,
        controller={"checkpoint_sequence": 1},
    )
    with pytest.raises(ValueError, match="newer than or foreign"):
        write_population_checkpoint(
            checkpoint_dir,
            replay_state=store.export_state(),
            members={"member-0": member},
        )

    valid = replace(member, replay_cursor=store.cursor)
    write_population_checkpoint(
        checkpoint_dir,
        replay_state=store.export_state(),
        members={"member-0": valid},
    )
    member_path = (
        checkpoint_dir
        / POPULATION_MEMBERS_DIRECTORY
        / "member-0"
        / RUNTIME_CHECKPOINT_FILENAME
    )
    encoded = bytearray(member_path.read_bytes())
    encoded[-1] ^= 0x01
    member_path.write_bytes(encoded)
    with pytest.raises(
        (InvalidPopulationCheckpointError, InvalidRuntimeCheckpointError),
        match="checksum",
    ):
        read_population_checkpoint_bundle(checkpoint_dir)
