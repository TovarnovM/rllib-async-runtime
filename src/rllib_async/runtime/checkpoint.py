"""Atomic, relocatable persistence for one asynchronous SAC member."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rllib_async.protocols import ReplayCursor
from rllib_async.replay.checkpoint import (
    read_replay_checkpoint,
    write_replay_checkpoint,
)
from rllib_async.replay.reference import EpisodeStoreState

RUNTIME_CHECKPOINT_STATE_VERSION = 1
RUNTIME_CHECKPOINT_FILENAME = "member.snapshot"
RUNTIME_REPLAY_FILENAME = "replay.snapshot"
POPULATION_CHECKPOINT_STATE_VERSION = 1
POPULATION_CHECKPOINT_FILENAME = "population.snapshot"
POPULATION_MEMBERS_DIRECTORY = "members"
PBT_STATE_FILENAME = "pbt_state.json"

_CHECKPOINT_MAGIC = b"RLLIB_ASYNC_MEMBER\x00\x01"
_POPULATION_CHECKPOINT_MAGIC = b"RLLIB_ASYNC_POPULATION\x00\x01"
_DIGEST_SIZE = hashlib.sha256().digest_size


class RuntimeCheckpointError(RuntimeError):
    """Base class for member checkpoint persistence failures."""


class InvalidRuntimeCheckpointError(RuntimeCheckpointError):
    """A member checkpoint is incomplete, corrupt, or incompatible."""


class InvalidPopulationCheckpointError(RuntimeCheckpointError):
    """A population checkpoint is incomplete, corrupt, or incompatible."""


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


@dataclass(frozen=True, slots=True)
class PopulationMemberRecord:
    """One independently restorable member cut inside a population bundle."""

    member_id: str
    file: str
    replay_cursor: ReplayCursor
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PopulationCheckpointState:
    """Manifest published after one shared replay and all member states."""

    state_version: int
    replay_file: str
    replay_cursor: ReplayCursor
    members: tuple[PopulationMemberRecord, ...]


@dataclass(frozen=True, slots=True)
class PopulationCheckpoint:
    """Metadata for one successfully published population checkpoint."""

    directory: str
    format_version: int
    replay_cursor: ReplayCursor
    member_ids: tuple[str, ...]
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
    return write_runtime_member_checkpoint(destination_dir, state)


def write_runtime_member_checkpoint(
    directory: str | os.PathLike[str],
    state: RuntimeCheckpointState,
) -> RuntimeCheckpoint:
    """Publish member state without duplicating an externally owned replay."""

    _validate_state(state)
    destination_dir = Path(directory)
    if not destination_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination_dir}"
        )
    destination = destination_dir / RUNTIME_CHECKPOINT_FILENAME
    size_bytes, digest = _write_checksummed_pickle(
        destination,
        state,
        magic=_CHECKPOINT_MAGIC,
    )
    return RuntimeCheckpoint(
        directory=str(destination_dir),
        format_version=state.state_version,
        replay_cursor=state.replay_cursor,
        size_bytes=size_bytes,
        sha256=digest,
    )


def write_population_checkpoint(
    directory: str | os.PathLike[str],
    *,
    replay_state: EpisodeStoreState,
    members: Mapping[str, RuntimeCheckpointState],
    pbt_metadata: Mapping[str, Any] | None = None,
) -> PopulationCheckpoint:
    """Publish one replay snapshot plus independent member state files."""

    destination_dir = Path(directory)
    if not destination_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination_dir}"
        )
    if not isinstance(members, Mapping) or not members:
        raise ValueError("population checkpoint requires at least one member")

    ordered_members = tuple(sorted(members.items()))
    for member_id, state in ordered_members:
        _validate_state(state)
        if member_id != state.member_id:
            raise ValueError("population member key does not match checkpoint state")

    replay_cursor = ReplayCursor(
        replay_state.store_generation,
        replay_state.mutation_seq,
    )
    for _, state in ordered_members:
        _validate_member_replay_cursor(
            state.replay_cursor,
            replay_cursor,
            error_type=ValueError,
        )

    members_directory = destination_dir / POPULATION_MEMBERS_DIRECTORY
    reserved = (
        destination_dir / POPULATION_CHECKPOINT_FILENAME,
        destination_dir / RUNTIME_REPLAY_FILENAME,
        members_directory,
    )
    if pbt_metadata is not None:
        if not isinstance(pbt_metadata, Mapping):
            raise TypeError("PBT checkpoint metadata must be a mapping")
        reserved = (*reserved, destination_dir / PBT_STATE_FILENAME)
    if any(path.exists() for path in reserved):
        raise FileExistsError(
            "checkpoint directory already contains a population checkpoint"
        )

    members_directory.mkdir()
    replay_checkpoint = write_replay_checkpoint(
        destination_dir / RUNTIME_REPLAY_FILENAME,
        replay_state,
    )
    if replay_checkpoint.cursor != replay_cursor:
        raise RuntimeError("population replay checkpoint cursor changed during write")

    records: list[PopulationMemberRecord] = []
    for member_id, state in ordered_members:
        member_directory = members_directory / member_id
        member_directory.mkdir()
        checkpoint = write_runtime_member_checkpoint(member_directory, state)
        records.append(
            PopulationMemberRecord(
                member_id=member_id,
                file=(
                    Path(POPULATION_MEMBERS_DIRECTORY)
                    / member_id
                    / RUNTIME_CHECKPOINT_FILENAME
                ).as_posix(),
                replay_cursor=state.replay_cursor,
                size_bytes=checkpoint.size_bytes,
                sha256=checkpoint.sha256,
            )
        )

    if pbt_metadata is not None:
        _write_atomic_json(
            destination_dir / PBT_STATE_FILENAME,
            dict(pbt_metadata),
        )

    manifest = PopulationCheckpointState(
        state_version=POPULATION_CHECKPOINT_STATE_VERSION,
        replay_file=RUNTIME_REPLAY_FILENAME,
        replay_cursor=replay_cursor,
        members=tuple(records),
    )
    _validate_population_state(manifest)
    size_bytes, digest = _write_checksummed_pickle(
        destination_dir / POPULATION_CHECKPOINT_FILENAME,
        manifest,
        magic=_POPULATION_CHECKPOINT_MAGIC,
    )
    return PopulationCheckpoint(
        directory=str(destination_dir),
        format_version=manifest.state_version,
        replay_cursor=replay_cursor,
        member_ids=tuple(record.member_id for record in records),
        size_bytes=size_bytes,
        sha256=digest,
    )


def read_runtime_checkpoint(
    directory: str | os.PathLike[str],
) -> RuntimeCheckpointState:
    """Authenticate one trusted-local checkpoint and its replay snapshot."""

    state, _ = read_runtime_checkpoint_bundle(directory)
    return state


def read_runtime_member_checkpoint(
    directory: str | os.PathLike[str],
) -> RuntimeCheckpointState:
    """Read member state whose authoritative replay is managed externally."""

    state, _ = _read_runtime_member_file(Path(directory) / RUNTIME_CHECKPOINT_FILENAME)
    return state


def read_runtime_checkpoint_bundle(
    directory: str | os.PathLike[str],
) -> tuple[RuntimeCheckpointState, EpisodeStoreState]:
    """Read a member state and the replay state authenticated against it."""

    source_dir = Path(directory)
    state, _ = _read_runtime_member_file(source_dir / RUNTIME_CHECKPOINT_FILENAME)
    try:
        replay_state = _verify_replay_snapshot(source_dir, state)
    except InvalidRuntimeCheckpointError:
        raise
    except Exception as error:
        raise InvalidRuntimeCheckpointError(
            f"{source_dir} contains an invalid runtime checkpoint"
        ) from error
    return state, replay_state


def read_population_checkpoint_bundle(
    directory: str | os.PathLike[str],
) -> tuple[
    PopulationCheckpointState,
    EpisodeStoreState,
    dict[str, RuntimeCheckpointState],
]:
    """Authenticate one shared replay and every member state in its bundle."""

    source_dir = Path(directory)
    source = source_dir / POPULATION_CHECKPOINT_FILENAME
    manifest = _read_checksummed_pickle(
        source,
        magic=_POPULATION_CHECKPOINT_MAGIC,
        error_type=InvalidPopulationCheckpointError,
        label="population",
    )
    try:
        _validate_population_state(manifest)
        replay_state = read_replay_checkpoint(source_dir / manifest.replay_file)
        replay_cursor = ReplayCursor(
            replay_state.store_generation,
            replay_state.mutation_seq,
        )
        if replay_cursor != manifest.replay_cursor:
            raise InvalidPopulationCheckpointError(
                "population manifest and replay cursors do not match"
            )

        members: dict[str, RuntimeCheckpointState] = {}
        for record in manifest.members:
            member_path = source_dir / record.file
            state, checkpoint = _read_runtime_member_file(member_path)
            if checkpoint.sha256 != record.sha256:
                raise InvalidPopulationCheckpointError(
                    f"population member {record.member_id!r} checksum does not match"
                )
            if checkpoint.size_bytes != record.size_bytes:
                raise InvalidPopulationCheckpointError(
                    f"population member {record.member_id!r} size does not match"
                )
            if (
                state.member_id != record.member_id
                or state.replay_cursor != record.replay_cursor
            ):
                raise InvalidPopulationCheckpointError(
                    f"population member {record.member_id!r} manifest does not match"
                )
            _validate_member_replay_cursor(
                state.replay_cursor,
                replay_cursor,
                error_type=InvalidPopulationCheckpointError,
            )
            members[record.member_id] = state
    except InvalidPopulationCheckpointError:
        raise
    except InvalidRuntimeCheckpointError as error:
        raise InvalidPopulationCheckpointError(str(error)) from error
    except Exception as error:
        raise InvalidPopulationCheckpointError(
            f"{source} contains an invalid population checkpoint"
        ) from error
    return manifest, replay_state, members


def read_pbt_checkpoint_metadata(
    directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read required single-trial PBT metadata from a population checkpoint."""

    source = Path(directory) / PBT_STATE_FILENAME
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidPopulationCheckpointError(
            f"{source} is not valid PBT checkpoint metadata"
        ) from error
    if not isinstance(value, Mapping):
        raise InvalidPopulationCheckpointError(
            "PBT checkpoint metadata must be a JSON object"
        )
    return dict(value)


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
        or state.member_id in {".", ".."}
        or "/" in state.member_id
        or "\\" in state.member_id
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


def _write_checksummed_pickle(
    destination: Path,
    value: object,
    *,
    magic: bytes,
) -> tuple[int, str]:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).digest()
    encoded = magic + digest + payload
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
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
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(encoded), digest.hex()


def _write_atomic_json(destination: Path, value: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(encoded)
            file.write(b"\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_checksummed_pickle(
    source: Path,
    *,
    magic: bytes,
    error_type: type[RuntimeCheckpointError],
    label: str,
) -> object:
    try:
        encoded = source.read_bytes()
    except OSError as error:
        raise error_type(f"{source} is not a complete {label} checkpoint") from error
    header_size = len(magic) + _DIGEST_SIZE
    if len(encoded) <= header_size or not encoded.startswith(magic):
        raise error_type(f"{source} is not a supported {label} checkpoint")

    expected_digest = encoded[len(magic) : header_size]
    payload = encoded[header_size:]
    actual_digest = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise error_type(f"{source} failed {label} checkpoint checksum validation")
    try:
        return pickle.loads(payload)
    except Exception as error:
        raise error_type(
            f"{source} contains an unreadable {label} checkpoint payload"
        ) from error


def _read_runtime_member_file(
    source: Path,
) -> tuple[RuntimeCheckpointState, RuntimeCheckpoint]:
    state = _read_checksummed_pickle(
        source,
        magic=_CHECKPOINT_MAGIC,
        error_type=InvalidRuntimeCheckpointError,
        label="runtime",
    )
    try:
        _validate_state(state)
    except InvalidRuntimeCheckpointError:
        raise
    except Exception as error:
        raise InvalidRuntimeCheckpointError(
            f"{source} contains an invalid runtime checkpoint"
        ) from error

    encoded = source.read_bytes()
    header_size = len(_CHECKPOINT_MAGIC) + _DIGEST_SIZE
    digest = encoded[len(_CHECKPOINT_MAGIC) : header_size].hex()
    assert isinstance(state, RuntimeCheckpointState)
    return state, RuntimeCheckpoint(
        directory=str(source.parent),
        format_version=state.state_version,
        replay_cursor=state.replay_cursor,
        size_bytes=len(encoded),
        sha256=digest,
    )


def _validate_population_state(state: object) -> None:
    if not isinstance(state, PopulationCheckpointState):
        raise InvalidPopulationCheckpointError(
            "population checkpoint does not contain PopulationCheckpointState"
        )
    if state.state_version != POPULATION_CHECKPOINT_STATE_VERSION:
        raise InvalidPopulationCheckpointError(
            "unsupported population checkpoint state version"
        )
    if state.replay_file != RUNTIME_REPLAY_FILENAME:
        raise InvalidPopulationCheckpointError(
            "population checkpoint replay path must be relocatable"
        )
    if not isinstance(state.replay_cursor, ReplayCursor):
        raise InvalidPopulationCheckpointError(
            "population checkpoint replay cursor is invalid"
        )
    if not isinstance(state.members, tuple) or not state.members:
        raise InvalidPopulationCheckpointError(
            "population checkpoint must contain member records"
        )

    member_ids: list[str] = []
    files: set[str] = set()
    for record in state.members:
        if not isinstance(record, PopulationMemberRecord):
            raise InvalidPopulationCheckpointError(
                "population checkpoint contains an invalid member record"
            )
        if (
            not isinstance(record.member_id, str)
            or not record.member_id
            or record.member_id in {".", ".."}
            or "/" in record.member_id
            or "\\" in record.member_id
        ):
            raise InvalidPopulationCheckpointError(
                "population checkpoint member_id must be a path segment"
            )
        expected_file = (
            Path(POPULATION_MEMBERS_DIRECTORY)
            / record.member_id
            / RUNTIME_CHECKPOINT_FILENAME
        ).as_posix()
        if record.file != expected_file or record.file in files:
            raise InvalidPopulationCheckpointError(
                "population checkpoint member path must be unique and relocatable"
            )
        if not isinstance(record.replay_cursor, ReplayCursor):
            raise InvalidPopulationCheckpointError(
                "population checkpoint member cursor is invalid"
            )
        if (
            not isinstance(record.size_bytes, int)
            or isinstance(record.size_bytes, bool)
            or record.size_bytes < 1
        ):
            raise InvalidPopulationCheckpointError(
                "population checkpoint member size is invalid"
            )
        if (
            not isinstance(record.sha256, str)
            or len(record.sha256) != hashlib.sha256().digest_size * 2
        ):
            raise InvalidPopulationCheckpointError(
                "population checkpoint member checksum is invalid"
            )
        try:
            bytes.fromhex(record.sha256)
        except ValueError as error:
            raise InvalidPopulationCheckpointError(
                "population checkpoint member checksum is invalid"
            ) from error
        _validate_member_replay_cursor(
            record.replay_cursor,
            state.replay_cursor,
            error_type=InvalidPopulationCheckpointError,
        )
        member_ids.append(record.member_id)
        files.add(record.file)
    if member_ids != sorted(member_ids) or len(member_ids) != len(set(member_ids)):
        raise InvalidPopulationCheckpointError(
            "population checkpoint member IDs must be unique and sorted"
        )


def _validate_member_replay_cursor(
    member_cursor: ReplayCursor,
    population_cursor: ReplayCursor,
    *,
    error_type: type[Exception],
) -> None:
    if (
        not isinstance(member_cursor, ReplayCursor)
        or member_cursor.store_generation != population_cursor.store_generation
        or member_cursor.mutation_seq > population_cursor.mutation_seq
    ):
        raise error_type(
            "population member replay cursor is newer than or foreign to shared replay"
        )
