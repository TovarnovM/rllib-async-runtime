from __future__ import annotations

import shutil
from dataclasses import replace

import pytest

from rllib_async.protocols import FlatEpisodeCodec, ReplayCursor
from rllib_async.replay.checkpoint import write_replay_checkpoint
from rllib_async.replay.reference import EpisodeStore
from rllib_async.runtime.checkpoint import (
    RUNTIME_CHECKPOINT_STATE_VERSION,
    RUNTIME_REPLAY_FILENAME,
    InvalidRuntimeCheckpointError,
    RuntimeCheckpointState,
    read_runtime_checkpoint,
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
