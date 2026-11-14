"""Consumer-local prioritized sequence starts over one shared replay read view."""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ray.rllib.execution.segment_tree import MinSegmentTree, SumSegmentTree

from rllib_async.protocols.replay import ReplayCursor
from rllib_async.replay.fast import FastReplay
from rllib_async.replay.sequences import (
    SequenceConfig,
    SequenceReadView,
    SequenceWindow,
)


@dataclass(frozen=True, slots=True)
class PrioritizedReplayConfig:
    alpha: float = 0.6
    beta: float = 0.4
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"PER {name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"PER {name} must be a finite number")
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("PER alpha and beta must be between zero and one")
        if self.epsilon <= 0:
            raise ValueError("PER epsilon must be positive")


@dataclass(frozen=True, slots=True)
class PriorityHandle:
    store_generation: str
    episode_id: str
    module_id: str
    agent_id: str
    agent_t: int
    learning_length: int
    burn_in: int | None
    n_step: int


@dataclass(frozen=True, slots=True)
class PrioritizedSequenceSample:
    windows: tuple[SequenceWindow, ...]
    handles: tuple[PriorityHandle, ...]
    probabilities: np.ndarray
    window_weights: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        """Repeat each start's IS weight over its actual valid learning rows."""
        return np.repeat(
            self.window_weights, [window.valid_count for window in self.windows]
        ).astype(np.float32, copy=False)


class PrioritizedSequenceSampler:
    """Priority metadata only; payloads belong to the shared FastReplay.

    The base measure is uniform window starts, including short tails. A window's
    normalized importance weight is repeated over its valid rows; RLlib keeps
    its existing per-row mean and algorithm-specific use of those weights.
    This does not claim an unbiased uniform-transition objective when windows
    overlap or have different lengths. The caller owns/checkpoints its RNG.
    """

    def __init__(
        self,
        replay: FastReplay,
        module_id: str,
        *,
        sequence_config: SequenceConfig | None = None,
        n_step: int = 1,
        config: PrioritizedReplayConfig | None = None,
    ) -> None:
        if not isinstance(module_id, str) or not module_id:
            raise ValueError("module_id must be a non-empty string")
        if type(n_step) is not int or n_step < 1:
            raise ValueError("n_step must be a positive integer")
        self._replay = replay
        self._module_id = module_id
        self._sequence_config = sequence_config or SequenceConfig()
        self._n_step = n_step
        self._config = config or PrioritizedReplayConfig()
        if not isinstance(self._sequence_config, SequenceConfig):
            raise TypeError("sequence_config must be a SequenceConfig")
        if not isinstance(self._config, PrioritizedReplayConfig):
            raise TypeError("config must be a PrioritizedReplayConfig")
        self._cursor: ReplayCursor | None = None
        self._reset()

    def _reset(self) -> None:
        self._handles: list[PriorityHandle | None] = [None]
        self._coordinates: list[tuple[str, int] | None] = [None]
        self._priorities = [0.0]
        self._slots: dict[PriorityHandle, int] = {}
        self._episode_slots: dict[str, tuple[int, ...]] = {}
        self._free: list[int] = []
        self._next_slot = 0
        self._max_priority = 1.0
        self._sum = SumSegmentTree(1)
        self._min = MinSegmentTree(1)

    @property
    def sample_count(self) -> int:
        self._sync()
        return len(self._slots)

    def _handle(
        self, view: SequenceReadView, episode_id: str, index: int
    ) -> PriorityHandle:
        agent_id, agent_t = view.decision_ids[episode_id][index]
        return PriorityHandle(
            view.cursor.store_generation,
            episode_id,
            self._module_id,
            agent_id,
            agent_t,
            self._sequence_config.learning_length,
            self._sequence_config.burn_in,
            self._n_step,
        )

    def _sync(self) -> SequenceReadView:
        view = self._replay.get_sequence_view(self._module_id)
        if self._cursor == view.cursor:
            return view
        if self._cursor is not None and (
            self._cursor.store_generation != view.cursor.store_generation
        ):
            self._reset()
        for episode_id in tuple(self._episode_slots):
            if episode_id not in view.episodes:
                for slot in self._episode_slots.pop(episode_id):
                    handle = self._handles[slot]
                    del self._slots[handle]
                    self._handles[slot] = None
                    self._coordinates[slot] = None
                    self._priorities[slot] = 0.0
                    self._sum[slot] = 0.0
                    self._min[slot] = float("inf")
                    heapq.heappush(self._free, slot)
        for episode_id in view.episode_ids:
            if episode_id in self._episode_slots:
                continue
            slots = []
            for index in range(len(view.decision_ids[episode_id])):
                if self._free:
                    slot = heapq.heappop(self._free)
                else:
                    if self._next_slot == len(self._handles):
                        self._grow()
                    slot = self._next_slot
                    self._next_slot += 1
                handle = self._handle(view, episode_id, index)
                self._handles[slot] = handle
                self._coordinates[slot] = (episode_id, index)
                self._slots[handle] = slot
                self._set_priority(slot, self._max_priority)
                slots.append(slot)
            self._episode_slots[episode_id] = tuple(slots)
        self._cursor = view.cursor
        return view

    def _grow(self) -> None:
        old_size = len(self._handles)
        self._handles.extend([None] * old_size)
        self._coordinates.extend([None] * old_size)
        self._priorities.extend([0.0] * old_size)
        self._sum = SumSegmentTree(old_size * 2)
        self._min = MinSegmentTree(old_size * 2)
        for slot, priority in enumerate(self._priorities[:old_size]):
            if priority:
                self._set_priority(slot, priority)

    def _set_priority(self, slot: int, priority: float) -> None:
        mass = priority**self._config.alpha
        self._priorities[slot] = priority
        self._sum[slot] = mass
        self._min[slot] = mass

    def sample(self, count: int, *, rng: random.Random) -> PrioritizedSequenceSample:
        if type(count) is not int or count < 1:
            raise ValueError("sample count must be a positive integer")
        view = self._sync()
        if not self._slots:
            raise ValueError("cannot sample an empty priority view")
        total = self._sum.sum()
        if not math.isfinite(total) or total <= 0:
            raise ValueError("priority mass must have a finite positive total")
        minimum = self._min.min()
        windows, handles, probabilities, weights = [], [], [], []
        for _ in range(count):
            if self._config.alpha == 0:
                coordinate = view.coordinate(rng.randrange(view.total_starts))
                slot = self._episode_slots[coordinate[0]][coordinate[1]]
            else:
                mass = min(rng.random() * total, math.nextafter(total, 0.0))
                slot = self._sum.find_prefixsum_idx(mass)
                coordinate = self._coordinates[slot]
            probability = self._sum[slot] / total
            windows.append(
                view.window(
                    *coordinate,
                    sequence_config=self._sequence_config,
                    n_step=self._n_step,
                )
            )
            handles.append(self._handles[slot])
            probabilities.append(probability)
            weights.append((minimum / self._sum[slot]) ** self._config.beta)
        return PrioritizedSequenceSample(
            tuple(windows),
            tuple(handles),
            np.asarray(probabilities, dtype=np.float64),
            np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(
        self, handles: Sequence[PriorityHandle], priorities: Sequence[float]
    ) -> int:
        """Apply max absolute valid-row error; repeated starts use their maximum."""
        if len(handles) != len(priorities):
            raise ValueError("priority handles and values must have the same length")
        self._sync()
        updates: dict[PriorityHandle, float] = {}
        for handle, value in zip(handles, priorities, strict=True):
            if handle not in self._slots or not self._replay.episode_retained(
                handle.episode_id, handle.store_generation
            ):
                continue
            priority = abs(float(value)) + self._config.epsilon
            if not math.isfinite(priority):
                raise ValueError("priority errors must be finite")
            updates[handle] = max(updates.get(handle, 0.0), priority)
        for handle, priority in updates.items():
            self._set_priority(self._slots[handle], priority)
            self._max_priority = max(self._max_priority, priority)
        return len(updates)

    def get_state(self) -> dict[str, Any]:
        """Persist slot order as well as values so the next RNG draw is exact."""
        self._sync()
        return {
            "format_version": 1,
            "module_id": self._module_id,
            "sequence_config": asdict(self._sequence_config),
            "n_step": self._n_step,
            "config": asdict(self._config),
            "cursor": self._cursor,
            "entries": tuple(
                (handle, self._coordinates[slot][1], self._priorities[slot])
                if handle is not None
                else None
                for slot, handle in enumerate(self._handles)
            ),
            "next_slot": self._next_slot,
            "max_priority": self._max_priority,
        }

    def set_state(
        self, state: Mapping[str, Any], *, allow_replay_ahead: bool = False
    ) -> None:
        """Validate the checkpoint before replacing this consumer's live priorities."""
        view = self._replay.get_sequence_view(self._module_id)
        if (
            state.get("format_version") != 1
            or state.get("module_id") != self._module_id
            or state.get("sequence_config") != asdict(self._sequence_config)
            or state.get("n_step") != self._n_step
            or state.get("config") != asdict(self._config)
        ):
            raise ValueError("priority checkpoint does not match the consumer contract")
        cursor = state.get("cursor")
        if not isinstance(cursor, ReplayCursor) or (
            cursor.store_generation != view.cursor.store_generation
            or cursor.mutation_seq > view.cursor.mutation_seq
            or (not allow_replay_ahead and cursor != view.cursor)
        ):
            raise ValueError("priority checkpoint replay frontier does not match")
        entries = state.get("entries")
        next_slot = state.get("next_slot")
        maximum = state.get("max_priority")
        if (
            not isinstance(entries, tuple)
            or not entries
            or len(entries) & (len(entries) - 1)
            or type(next_slot) is not int
            or not 0 <= next_slot <= len(entries)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int | float)
            or not math.isfinite(maximum)
            or maximum <= 0
        ):
            raise ValueError("priority checkpoint has invalid storage metadata")
        candidate = PrioritizedSequenceSampler(
            self._replay,
            self._module_id,
            sequence_config=self._sequence_config,
            n_step=self._n_step,
            config=self._config,
        )
        while len(candidate._handles) < len(entries):
            candidate._grow()
        episode_slots: dict[str, dict[int, int]] = {}
        for slot, entry in enumerate(entries):
            if entry is None:
                if slot < next_slot:
                    heapq.heappush(candidate._free, slot)
                continue
            if slot >= next_slot or not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError("priority checkpoint has an invalid slot")
            handle, index, priority = entry
            if (
                not isinstance(handle, PriorityHandle)
                or handle.store_generation != cursor.store_generation
                or handle.module_id != self._module_id
                or handle.learning_length != self._sequence_config.learning_length
                or handle.burn_in != self._sequence_config.burn_in
                or handle.n_step != self._n_step
                or type(index) is not int
                or index < 0
                or isinstance(priority, bool)
                or not isinstance(priority, int | float)
                or not math.isfinite(priority)
                or not 0 < priority <= maximum
                or handle in candidate._slots
            ):
                raise ValueError("priority checkpoint contains an invalid sample")
            if handle.episode_id in view.episodes:
                decisions = view.decision_ids[handle.episode_id]
                if index >= len(decisions) or handle != self._handle(
                    view, handle.episode_id, index
                ):
                    raise ValueError(
                        "priority checkpoint sample identity does not match replay"
                    )
            elif not allow_replay_ahead:
                raise ValueError("priority checkpoint references an absent episode")
            locations = episode_slots.setdefault(handle.episode_id, {})
            if index in locations:
                raise ValueError("priority checkpoint repeats a sample coordinate")
            locations[index] = slot
            candidate._handles[slot] = handle
            candidate._coordinates[slot] = (handle.episode_id, index)
            candidate._slots[handle] = slot
            candidate._set_priority(slot, priority)
        for episode_id, locations in episode_slots.items():
            if set(locations) != set(range(len(locations))):
                raise ValueError("priority checkpoint episode starts are incomplete")
            if episode_id in view.episodes and len(locations) != len(
                view.decision_ids[episode_id]
            ):
                raise ValueError(
                    "priority checkpoint episode starts do not match replay"
                )
            candidate._episode_slots[episode_id] = tuple(
                locations[index] for index in range(len(locations))
            )
        if not allow_replay_ahead and set(episode_slots) != set(view.episode_ids):
            raise ValueError("priority checkpoint omits retained episodes")
        candidate._next_slot = next_slot
        candidate._max_priority = float(maximum)
        candidate._cursor = cursor
        if cursor != view.cursor:
            candidate._sync()
        self.__dict__.update(candidate.__dict__)
