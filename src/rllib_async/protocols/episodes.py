"""Episode contracts and the reference flat-episode codec."""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
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
class MultiModuleTransition:
    """One sparse agent transition with explicit environment/module provenance."""

    env_t: int
    agent_t: int
    agent_id: str
    module_id: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, value in (("env_t", self.env_t), ("agent_t", self.agent_t)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EpisodeValidationError(f"{name} must be a non-negative integer")
        for name, value in (
            ("agent_id", self.agent_id),
            ("module_id", self.module_id),
        ):
            if not isinstance(value, str) or not value:
                raise EpisodeValidationError(f"{name} must be a non-empty string")
        if not isinstance(self.data, Mapping) or not self.data:
            raise EpisodeValidationError(
                "multi-module transition data must be a non-empty mapping"
            )
        if any(not isinstance(key, str) or not key for key in self.data):
            raise EpisodeValidationError(
                "multi-module transition data keys must be non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class EncodedModuleTransition:
    """Pickled transition data plus queryable sparse-timeline metadata."""

    env_t: int
    agent_t: int
    agent_id: str
    encoded_data: bytes

    def __post_init__(self) -> None:
        for name, value in (("env_t", self.env_t), ("agent_t", self.agent_t)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EpisodeValidationError(f"{name} must be a non-negative integer")
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise EpisodeValidationError("agent_id must be a non-empty string")
        if type(self.encoded_data) is not bytes:
            raise EpisodeValidationError("encoded transition data must be bytes")


@dataclass(frozen=True, slots=True)
class MultiModuleEpisodePayload:
    """Immutable transitions grouped by module for direct per-module sampling."""

    encoded_module_transitions: tuple[
        tuple[str, tuple[EncodedModuleTransition, ...]],
        ...,
    ]
    _transition_count: int = field(init=False, repr=False)
    _env_steps: int = field(init=False, repr=False)
    _estimated_bytes: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.encoded_module_transitions) is not tuple:
            raise EpisodeValidationError("encoded_module_transitions must be a tuple")
        if not self.encoded_module_transitions:
            raise EpisodeValidationError("an episode must contain a transition")
        module_ids: list[str] = []
        transition_count = 0
        estimated_bytes = 0
        observed_env_ts: set[int] = set()
        next_agent_t: dict[str, int] = {}
        agent_modules: dict[str, str] = {}
        previous_agent_env_t: dict[str, int] = {}
        for item in self.encoded_module_transitions:
            if type(item) is not tuple or len(item) != 2:
                raise EpisodeValidationError(
                    "module transition groups must be (module_id, transitions) tuples"
                )
            module_id, transitions = item
            if not isinstance(module_id, str) or not module_id:
                raise EpisodeValidationError("module IDs must be non-empty strings")
            if type(transitions) is not tuple or not transitions:
                raise EpisodeValidationError(
                    "every encoded module group must contain a transition"
                )
            if any(
                not isinstance(transition, EncodedModuleTransition)
                for transition in transitions
            ):
                raise EpisodeValidationError(
                    "module groups must contain EncodedModuleTransition values"
                )
            module_ids.append(module_id)
            transition_count += len(transitions)
            for transition in transitions:
                estimated_bytes += len(transition.encoded_data)
                observed_env_ts.add(transition.env_t)
                expected_agent_t = next_agent_t.get(transition.agent_id, 0)
                if transition.agent_t != expected_agent_t:
                    raise EpisodeValidationError(
                        "agent_t must be contiguous from zero for each agent"
                    )
                previous_env_t = previous_agent_env_t.get(
                    transition.agent_id,
                    -1,
                )
                if transition.env_t <= previous_env_t:
                    raise EpisodeValidationError(
                        "environment timesteps must increase for each agent"
                    )
                previous_module = agent_modules.setdefault(
                    transition.agent_id,
                    module_id,
                )
                if previous_module != module_id:
                    raise EpisodeValidationError(
                        "one agent_id cannot change module within an episode"
                    )
                next_agent_t[transition.agent_id] = expected_agent_t + 1
                previous_agent_env_t[transition.agent_id] = transition.env_t
        if len(module_ids) != len(set(module_ids)):
            raise EpisodeValidationError("module transition groups must be unique")
        if module_ids != sorted(module_ids):
            raise EpisodeValidationError(
                "module transition groups must be sorted by module ID"
            )
        env_steps = max(observed_env_ts) + 1
        if observed_env_ts != set(range(env_steps)):
            raise EpisodeValidationError(
                "every environment timestep must contain an agent transition"
            )
        object.__setattr__(self, "_transition_count", transition_count)
        object.__setattr__(self, "_env_steps", env_steps)
        object.__setattr__(self, "_estimated_bytes", estimated_bytes)

    @property
    def module_ids(self) -> tuple[str, ...]:
        return tuple(module_id for module_id, _ in self.encoded_module_transitions)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def env_steps(self) -> int:
        return self._env_steps

    @property
    def estimated_bytes(self) -> int:
        return self._estimated_bytes


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
    """Storage-independent access to an episode payload.

    Phase 3B may call ``get_transition`` concurrently from multiple reader
    threads. Codec reads must therefore be side-effect free and thread-safe for
    immutable validated envelopes.
    """

    codec_id: str
    schema_version: int

    def validate(self, episode: EpisodeEnvelope) -> None: ...

    def transition_count(self, episode: EpisodeEnvelope) -> int: ...

    def get_transition(self, episode: EpisodeEnvelope, index: int) -> object: ...

    def estimate_bytes(self, episode: EpisodeEnvelope) -> int: ...


@runtime_checkable
class ModuleEpisodeCodec(EpisodeCodec, Protocol):
    """An episode codec supporting direct module-specific transition views."""

    def module_ids(self, episode: EpisodeEnvelope) -> tuple[str, ...]: ...

    def module_transition_count(
        self,
        episode: EpisodeEnvelope,
        module_id: str,
    ) -> int: ...

    def get_module_transition(
        self,
        episode: EpisodeEnvelope,
        module_id: str,
        index: int,
    ) -> MultiModuleTransition: ...


class FlatEpisodeCodec:
    """Correctness-first codec for single-agent flat transitions."""

    codec_id = "flat-pickle-v1"
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


class MultiModuleEpisodeCodec:
    """Correctness-first codec for sparse multi-agent/module transitions."""

    codec_id = "multi-module-pickle-v1"
    schema_version = 1

    def encode(
        self,
        transitions: Iterable[MultiModuleTransition],
    ) -> MultiModuleEpisodePayload:
        grouped: dict[str, list[EncodedModuleTransition]] = defaultdict(list)
        for transition in transitions:
            if not isinstance(transition, MultiModuleTransition):
                raise EpisodeValidationError(
                    "MultiModuleEpisodeCodec requires MultiModuleTransition values"
                )
            try:
                encoded_data = pickle.dumps(
                    dict(transition.data),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            except (pickle.PickleError, TypeError) as error:
                raise EpisodeValidationError(
                    "multi-module transition data must be pickle-safe"
                ) from error
            grouped[transition.module_id].append(
                EncodedModuleTransition(
                    env_t=transition.env_t,
                    agent_t=transition.agent_t,
                    agent_id=transition.agent_id,
                    encoded_data=encoded_data,
                )
            )
        return MultiModuleEpisodePayload(
            tuple(
                (module_id, tuple(grouped[module_id])) for module_id in sorted(grouped)
            )
        )

    def validate(self, episode: EpisodeEnvelope) -> None:
        if episode.schema_version != self.schema_version:
            raise SchemaMismatchError(
                f"expected schema {self.schema_version}, got {episode.schema_version}"
            )
        payload = self._payload(episode)
        if episode.agent_steps != payload.transition_count:
            raise EpisodeValidationError(
                "multi-module agent_steps must match the payload transition count"
            )
        if set(episode.behavior_versions) != set(payload.module_ids):
            raise EpisodeValidationError(
                "behavior version module IDs must match payload module IDs"
            )
        if episode.env_steps != payload.env_steps:
            raise EpisodeValidationError(
                "episode env_steps must match the payload environment timeline"
            )
        estimated_bytes = self.estimate_bytes(episode)
        if episode.estimated_bytes != estimated_bytes:
            raise EpisodeValidationError(
                f"estimated_bytes={episode.estimated_bytes} does not match "
                f"the codec estimate {estimated_bytes}"
            )

    def transition_count(self, episode: EpisodeEnvelope) -> int:
        self.validate(episode)
        return self._payload(episode).transition_count

    def get_transition(self, episode: EpisodeEnvelope, index: int) -> object:
        self.validate(episode)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise IndexError(index)
        offset = index
        for module_id, transitions in self._payload(episode).encoded_module_transitions:
            if offset < len(transitions):
                return self._decode(module_id, transitions[offset])
            offset -= len(transitions)
        raise IndexError(index)

    def estimate_bytes(self, episode: EpisodeEnvelope) -> int:
        return self._payload(episode).estimated_bytes

    def module_ids(self, episode: EpisodeEnvelope) -> tuple[str, ...]:
        self.validate(episode)
        return self._payload(episode).module_ids

    def module_transition_count(
        self,
        episode: EpisodeEnvelope,
        module_id: str,
    ) -> int:
        self.validate(episode)
        return len(self._module_transitions(episode, module_id))

    def get_module_transition(
        self,
        episode: EpisodeEnvelope,
        module_id: str,
        index: int,
    ) -> MultiModuleTransition:
        self.validate(episode)
        transitions = self._module_transitions(episode, module_id)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(transitions)
        ):
            raise IndexError(index)
        return self._decode(module_id, transitions[index])

    @staticmethod
    def _decode(
        module_id: str,
        transition: EncodedModuleTransition,
    ) -> MultiModuleTransition:
        data = pickle.loads(transition.encoded_data)
        if not isinstance(data, Mapping):
            raise EpisodeValidationError(
                "encoded multi-module transition data must decode to a mapping"
            )
        return MultiModuleTransition(
            env_t=transition.env_t,
            agent_t=transition.agent_t,
            agent_id=transition.agent_id,
            module_id=module_id,
            data=data,
        )

    @staticmethod
    def _payload(episode: EpisodeEnvelope) -> MultiModuleEpisodePayload:
        if not isinstance(episode.payload, MultiModuleEpisodePayload):
            raise EpisodeValidationError(
                "MultiModuleEpisodeCodec requires MultiModuleEpisodePayload"
            )
        return episode.payload

    @staticmethod
    def _module_transitions(
        episode: EpisodeEnvelope,
        module_id: str,
    ) -> tuple[EncodedModuleTransition, ...]:
        if not isinstance(module_id, str) or not module_id:
            raise KeyError(module_id)
        payload = MultiModuleEpisodeCodec._payload(episode)
        for candidate, transitions in payload.encoded_module_transitions:
            if candidate == module_id:
                return transitions
        raise KeyError(module_id)
