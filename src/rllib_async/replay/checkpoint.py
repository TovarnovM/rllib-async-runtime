"""Atomic trusted-local persistence for authoritative replay state."""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import tempfile
from pathlib import Path

from rllib_async.protocols.replay import ReplayCheckpoint, ReplayCursor
from rllib_async.replay.reference import (
    EPISODE_STORE_STATE_VERSION,
    EpisodeStoreState,
)

_CHECKPOINT_MAGIC = b"RLLIB_ASYNC_REPLAY\x00\x01"
_DIGEST_SIZE = hashlib.sha256().digest_size


class ReplayCheckpointError(RuntimeError):
    """Base class for replay checkpoint failures."""


class InvalidReplayCheckpointError(ReplayCheckpointError):
    """A checkpoint is truncated, corrupt, or has an unsupported format."""


def write_replay_checkpoint(
    path: str | os.PathLike[str],
    state: EpisodeStoreState,
) -> ReplayCheckpoint:
    """Atomically replace ``path`` with one checksummed checkpoint."""
    if not isinstance(state, EpisodeStoreState):
        raise TypeError("state must be an EpisodeStoreState")

    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {parent}")

    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).digest()
    encoded = _CHECKPOINT_MAGIC + digest + payload
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
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
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ReplayCheckpoint(
        path=str(destination),
        format_version=EPISODE_STORE_STATE_VERSION,
        cursor=_state_cursor(state),
        size_bytes=len(encoded),
        sha256=digest.hex(),
    )


def read_replay_checkpoint(
    path: str | os.PathLike[str],
) -> EpisodeStoreState:
    """Read and authenticate trusted-local checkpoint bytes."""
    source = Path(path)
    encoded = source.read_bytes()
    header_size = len(_CHECKPOINT_MAGIC) + _DIGEST_SIZE
    if len(encoded) <= header_size or not encoded.startswith(_CHECKPOINT_MAGIC):
        raise InvalidReplayCheckpointError(
            f"{source} is not a supported replay checkpoint"
        )

    expected_digest = encoded[len(_CHECKPOINT_MAGIC) : header_size]
    payload = encoded[header_size:]
    actual_digest = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise InvalidReplayCheckpointError(
            f"{source} failed replay checkpoint checksum validation"
        )

    try:
        state = pickle.loads(payload)
    except Exception as error:
        raise InvalidReplayCheckpointError(
            f"{source} contains an unreadable replay checkpoint payload"
        ) from error
    if not isinstance(state, EpisodeStoreState):
        raise InvalidReplayCheckpointError(
            f"{source} does not contain an EpisodeStoreState"
        )
    return state


def _state_cursor(state: EpisodeStoreState) -> ReplayCursor:
    return ReplayCursor(state.store_generation, state.mutation_seq)
