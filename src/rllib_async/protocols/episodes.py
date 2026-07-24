"""Episode contracts and the reference flat-episode codec."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class EpisodeValidationError(ValueError):
    """An episode does not satisfy the replay contract."""


class SchemaMismatchError(EpisodeValidationError):
    """An episode was encoded with an unsupported schema version."""


@dataclass(frozen=True, slots=True, init=False)
class FrozenVersions(Mapping[str, int]):
    """A small, deterministic, pickle-safe immutable module-version mapping."""

    _items: tuple[tuple[str, int], ...]

    def __init__(
        self,
        values: Mapping[str, int] | Iterable[tuple[str, int]],
    ) -> None:
        items = list(values.items() if isinstance(values, Mapping) else values)
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise EpisodeValidationError("behavior version keys must be unique")
        if any(not isinstance(key, str) or not key for key in keys):
            raise EpisodeValidationError(
                "behavior version keys must be non-empty strings"
            )
        if any(
            not isinstance(version, int) or isinstance(version, bool) or version < 0
            for _, version in items
        ):
            raise EpisodeValidationError(
                "behavior versions must be non-negative integers"
            )
        object.__setattr__(self, "_items", tuple(sorted(items)))

    def __getitem__(self, key: str) -> int:
        for item_key, version in self._items:
            if item_key == key:
                return version
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True, slots=True)
class FlatEpisodePayload:
    """Immutable reference payload containing one pickle per transition.

    The reference codec favors unambiguous ownership over throughput. Optimized
    RLlib episode and array layouts belong behind later codec implementations.
    Payloads are trusted internal data and must not be loaded from untrusted
    sources.
    """

    encoded_transitions: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.encoded_transitions) is not tuple:
            raise EpisodeValidationError("encoded_transitions must be a tuple")
        if not self.encoded_transitions:
            raise EpisodeValidationError("an episode must contain a transition")
        if any(type(item) is not bytes for item in self.encoded_transitions):
            raise EpisodeValidationError("encoded transitions must be bytes")

    @property
    def estimated_bytes(self) -> int:
        return sum(map(len, self.encoded_transitions))


@dataclass(frozen=True, slots=True)
class EpisodeEnvelope:
    """The atomic commit, eviction, and synchronization unit."""

    episode_id: str
    schema_version: int
    producer_member_id: str
    runner_id: str
    runner_generation: int
    local_episode_seq: int
    behavior_versions: Mapping[str, int]
    env_steps: int
    agent_steps: int
    terminated: bool
    truncated: bool
    estimated_bytes: int
    payload: object

    def __post_init__(self) -> None:
        identifiers = {
            "episode_id": self.episode_id,
            "producer_member_id": self.producer_member_id,
            "runner_id": self.runner_id,
        }
        for name, value in identifiers.items():
            if not isinstance(value, str) or not value:
                raise EpisodeValidationError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version < 1
        ):
            raise EpisodeValidationError("schema_version must be a positive integer")
        if (
            not isinstance(self.runner_generation, int)
            or isinstance(self.runner_generation, bool)
            or self.runner_generation < 0
        ):
            raise EpisodeValidationError("runner_generation must be non-negative")
        if (
            not isinstance(self.local_episode_seq, int)
            or isinstance(self.local_episode_seq, bool)
            or self.local_episode_seq < 0
        ):
            raise EpisodeValidationError("local_episode_seq must be non-negative")
        if (
            not isinstance(self.env_steps, int)
            or isinstance(self.env_steps, bool)
            or self.env_steps < 1
        ):
            raise EpisodeValidationError("env_steps must be positive")
        if (
            not isinstance(self.agent_steps, int)
            or isinstance(self.agent_steps, bool)
            or self.agent_steps < 1
        ):
            raise EpisodeValidationError("agent_steps must be positive")
        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise EpisodeValidationError("terminated and truncated must be booleans")
        if not self.terminated and not self.truncated:
            raise EpisodeValidationError(
                "a committed episode must be terminated or truncated"
            )
        if (
            not isinstance(self.estimated_bytes, int)
            or isinstance(self.estimated_bytes, bool)
            or self.estimated_bytes < 1
        ):
            raise EpisodeValidationError("estimated_bytes must be positive")

        versions = self.behavior_versions
        if not isinstance(versions, FrozenVersions):
            try:
                versions = FrozenVersions(versions)
            except (TypeError, ValueError) as error:
                raise EpisodeValidationError(
                    "behavior_versions must be a mapping of module IDs to versions"
                ) from error
            object.__setattr__(self, "behavior_versions", versions)
        if not versions:
            raise EpisodeValidationError("behavior_versions must not be empty")


@runtime_checkable
class EpisodeCodec(Protocol):
    """Storage-independent access to an episode payload."""

    schema_version: int

    def validate(self, episode: EpisodeEnvelope) -> None: ...

    def transition_count(self, episode: EpisodeEnvelope) -> int: ...

    def get_transition(self, episode: EpisodeEnvelope, index: int) -> object: ...

    def estimate_bytes(self, episode: EpisodeEnvelope) -> int: ...


class FlatEpisodeCodec:
    """Correctness-first codec for single-agent flat transitions."""

    schema_version = 1

    def encode(self, transitions: Iterable[Any]) -> FlatEpisodePayload:
        return FlatEpisodePayload(
            tuple(
                pickle.dumps(transition, protocol=pickle.HIGHEST_PROTOCOL)
                for transition in transitions
            )
        )

    def validate(self, episode: EpisodeEnvelope) -> None:
        if episode.schema_version != self.schema_version:
            raise SchemaMismatchError(
                f"expected schema {self.schema_version}, got {episode.schema_version}"
            )
        if not isinstance(episode.payload, FlatEpisodePayload):
            raise EpisodeValidationError("FlatEpisodeCodec requires FlatEpisodePayload")
        count = len(episode.payload.encoded_transitions)
        if episode.env_steps != count or episode.agent_steps != count:
            raise EpisodeValidationError(
                "flat single-agent env_steps and agent_steps must match "
                "the payload transition count"
            )
        estimated_bytes = self.estimate_bytes(episode)
        if episode.estimated_bytes != estimated_bytes:
            raise EpisodeValidationError(
                f"estimated_bytes={episode.estimated_bytes} does not match "
                f"the codec estimate {estimated_bytes}"
            )

    def transition_count(self, episode: EpisodeEnvelope) -> int:
        self.validate(episode)
        assert isinstance(episode.payload, FlatEpisodePayload)
        return len(episode.payload.encoded_transitions)

    def get_transition(self, episode: EpisodeEnvelope, index: int) -> object:
        self.validate(episode)
        assert isinstance(episode.payload, FlatEpisodePayload)
        if index < 0 or index >= len(episode.payload.encoded_transitions):
            raise IndexError(index)
        return pickle.loads(episode.payload.encoded_transitions[index])

    def estimate_bytes(self, episode: EpisodeEnvelope) -> int:
        if not isinstance(episode.payload, FlatEpisodePayload):
            raise EpisodeValidationError("FlatEpisodeCodec requires FlatEpisodePayload")
        return episode.payload.estimated_bytes
