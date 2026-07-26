"""Atomic, relocatable persistence for one asynchronous SAC member."""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rllib_async.protocols import ReplayCursor
from rllib_async.replay.checkpoint import read_replay_checkpoint
from rllib_async.replay.reference import EpisodeStoreState

RUNTIME_CHECKPOINT_STATE_VERSION = 1
RUNTIME_CHECKPOINT_FILENAME = "member.snapshot"
RUNTIME_REPLAY_FILENAME = "replay.snapshot"

_CHECKPOINT_MAGIC = b"RLLIB_ASYNC_MEMBER\x00\x01"
_DIGEST_SIZE = hashlib.sha256().digest_size


class RuntimeCheckpointError(RuntimeError):
    """Base class for member checkpoint persistence failures."""


class InvalidRuntimeCheckpointError(RuntimeCheckpointError):
    """A member checkpoint is incomplete, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointState:
    """Versioned state persisted next to one authoritative replay snapshot."""

    state_version: int
    member_id: str
    runtime_config: dict[str, Any]
    replay_file: str
    replay_cursor: ReplayCursor
    learner: bytes
    rollout: dict[str, Any]
    evaluation: dict[str, Any] | None
    controller: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeCheckpoint:
    """Metadata for one successfully published member checkpoint."""

    directory: str
    format_version: int
    replay_cursor: ReplayCursor
    size_bytes: int
    sha256: str


def write_runtime_checkpoint(
    directory: str | os.PathLike[str],
    state: RuntimeCheckpointState,
) -> RuntimeCheckpoint:
    """Publish a checksummed member file after its replay snapshot exists."""

    _validate_state(state)
    destination_dir = Path(directory)
    if not destination_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination_dir}"
        )
    _verify_replay_snapshot(destination_dir, state)

    destination = destination_dir / RUNTIME_CHECKPOINT_FILENAME
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).digest()
    encoded = _CHECKPOINT_MAGIC + digest + payload
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return RuntimeCheckpoint(
        directory=str(destination_dir),
        format_version=state.state_version,
        replay_cursor=state.replay_cursor,
        size_bytes=len(encoded),
        sha256=digest.hex(),
    )


def read_runtime_checkpoint(
    directory: str | os.PathLike[str],
) -> RuntimeCheckpointState:
    """Authenticate one trusted-local checkpoint and its replay snapshot."""

    state, _ = read_runtime_checkpoint_bundle(directory)
    return state


def read_runtime_checkpoint_bundle(
    directory: str | os.PathLike[str],
) -> tuple[RuntimeCheckpointState, EpisodeStoreState]:
    """Read a member state and the replay state authenticated against it."""

    source_dir = Path(directory)
    source = source_dir / RUNTIME_CHECKPOINT_FILENAME
    encoded = source.read_bytes()
    header_size = len(_CHECKPOINT_MAGIC) + _DIGEST_SIZE
    if len(encoded) <= header_size or not encoded.startswith(_CHECKPOINT_MAGIC):
        raise InvalidRuntimeCheckpointError(
            f"{source} is not a supported runtime checkpoint"
        )

    expected_digest = encoded[len(_CHECKPOINT_MAGIC) : header_size]
    payload = encoded[header_size:]
    actual_digest = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise InvalidRuntimeCheckpointError(
            f"{source} failed runtime checkpoint checksum validation"
        )

    try:
        state = pickle.loads(payload)
    except Exception as error:
        raise InvalidRuntimeCheckpointError(
            f"{source} contains an unreadable runtime checkpoint payload"
        ) from error
    try:
        _validate_state(state)
        replay_state = _verify_replay_snapshot(source_dir, state)
    except InvalidRuntimeCheckpointError:
        raise
    except Exception as error:
        raise InvalidRuntimeCheckpointError(
            f"{source} contains an invalid runtime checkpoint"
        ) from error
    return state, replay_state


def _validate_state(state: object) -> None:
    if not isinstance(state, RuntimeCheckpointState):
        raise InvalidRuntimeCheckpointError(
            "runtime checkpoint does not contain RuntimeCheckpointState"
        )
    if state.state_version != RUNTIME_CHECKPOINT_STATE_VERSION:
        raise InvalidRuntimeCheckpointError(
            "unsupported runtime checkpoint state version"
        )
    if (
        not isinstance(state.member_id, str)
        or not state.member_id
        or "/" in state.member_id
    ):
        raise InvalidRuntimeCheckpointError(
            "runtime checkpoint member_id must be a path segment"
        )
    if state.replay_file != RUNTIME_REPLAY_FILENAME:
        raise InvalidRuntimeCheckpointError(
            "runtime checkpoint replay path must be relocatable"
        )
    if not isinstance(state.replay_cursor, ReplayCursor):
        raise InvalidRuntimeCheckpointError(
            "runtime checkpoint replay cursor is invalid"
        )
    if not isinstance(state.learner, bytes) or not state.learner:
        raise InvalidRuntimeCheckpointError(
            "runtime checkpoint learner state must be non-empty bytes"
        )
    for name, value in (
        ("runtime_config", state.runtime_config),
        ("rollout", state.rollout),
        ("controller", state.controller),
    ):
        if not isinstance(value, Mapping):
            raise InvalidRuntimeCheckpointError(
                f"runtime checkpoint {name} state must be a mapping"
            )
    if state.evaluation is not None and not isinstance(state.evaluation, Mapping):
        raise InvalidRuntimeCheckpointError(
            "runtime checkpoint evaluation state must be a mapping or None"
        )


def _verify_replay_snapshot(
    directory: Path,
    state: RuntimeCheckpointState,
) -> EpisodeStoreState:
    replay_state = read_replay_checkpoint(directory / state.replay_file)
    cursor = ReplayCursor(
        replay_state.store_generation,
        replay_state.mutation_seq,
    )
    if cursor != state.replay_cursor:
        raise InvalidRuntimeCheckpointError(
            "member and replay checkpoint cursors do not match"
        )
    return replay_state
