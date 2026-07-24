"""Correctness-first learner-local materialized replay."""

from __future__ import annotations

import random
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rllib_async.protocols.episodes import EpisodeCodec, EpisodeEnvelope
from rllib_async.protocols.replay import (
    ReplayCursor,
    ReplayDelta,
    ReplaySnapshot,
)
from rllib_async.replay.reference import (
    CursorMismatchError,
    FullResyncRequiredError,
    ReplayError,
)


@dataclass(frozen=True, slots=True)
class _SamplingIndex:
    episode_ids: tuple[str, ...]
    cumulative_lengths: tuple[int, ...]
    total_transitions: int


@dataclass(frozen=True, slots=True)
class _MaterializedView:
    cursor: ReplayCursor
    episodes: Mapping[str, EpisodeEnvelope]
    sampling_index: _SamplingIndex
    total_estimated_bytes: int


class FastReplay:
    """Learner-local payload view with a materialized transition index.

    Phase 3A rebuilds the immutable view synchronously. Episode envelopes are
    reused by reference; only the manifest and sampling index are rebuilt.
    Publishing one replacement view keeps snapshot and delta application
    atomic for readers.
    """

    def __init__(self, codec: EpisodeCodec) -> None:
        self._codec = codec
        self._view: _MaterializedView | None = None

    @property
    def cursor(self) -> ReplayCursor | None:
        return self._view.cursor if self._view is not None else None

    @property
    def episode_ids(self) -> tuple[str, ...]:
        if self._view is None:
            return ()
        return self._view.sampling_index.episode_ids

    @property
    def total_transitions(self) -> int:
        if self._view is None:
            return 0
        return self._view.sampling_index.total_transitions

    @property
    def total_estimated_bytes(self) -> int:
        if self._view is None:
            return 0
        return self._view.total_estimated_bytes

    def load_snapshot(self, snapshot: ReplaySnapshot) -> None:
        """Validate and atomically replace the complete local view."""
        candidate = self._materialize(snapshot.cursor, snapshot.episodes)
        if candidate.sampling_index.total_transitions != snapshot.total_transitions:
            raise ReplayError("snapshot transition total is inconsistent")
        if candidate.total_estimated_bytes != snapshot.total_estimated_bytes:
            raise ReplayError("snapshot byte total is inconsistent")
        self._view = candidate

    def apply_delta(self, delta: ReplayDelta) -> None:
        """Validate one complete delta before atomically publishing it."""
        if delta.full_resync_required:
            raise FullResyncRequiredError("authoritative replay requires a snapshot")
        if self._view is None:
            raise CursorMismatchError("load a snapshot before applying deltas")
        view = self._require_view()
        if delta.base_cursor != view.cursor:
            raise CursorMismatchError(
                f"local cursor {view.cursor!r} does not match "
                f"delta base {delta.base_cursor!r}"
            )

        episodes = OrderedDict(view.episodes.items())
        expected_mutation_seq = view.cursor.mutation_seq
        for transaction in delta.transactions:
            expected_mutation_seq += 1
            if transaction.mutation_seq != expected_mutation_seq:
                raise CursorMismatchError(
                    f"expected mutation {expected_mutation_seq}, "
                    f"got {transaction.mutation_seq}"
                )
            self._codec.validate(transaction.added)
            episode_id = transaction.added.episode_id
            if episode_id in episodes:
                raise ReplayError(f"delta adds existing episode_id {episode_id!r}")
            episodes[episode_id] = transaction.added
            for evicted_id in transaction.evicted_episode_ids:
                if evicted_id not in episodes:
                    raise ReplayError(f"delta evicts unknown episode_id {evicted_id!r}")
                if next(iter(episodes)) != evicted_id:
                    raise ReplayError(
                        f"delta transaction {transaction.mutation_seq} "
                        "does not evict FIFO"
                    )
                del episodes[evicted_id]

        if delta.next_cursor.store_generation != view.cursor.store_generation:
            raise CursorMismatchError("delta changes store generation")
        if delta.next_cursor.mutation_seq != expected_mutation_seq:
            raise CursorMismatchError(
                "delta next cursor does not match its transaction suffix"
            )
        if not delta.transactions:
            return

        candidate = self._materialize(delta.next_cursor, episodes.values())
        self._view = candidate

    def apply_deltas(self, deltas: Sequence[ReplayDelta]) -> None:
        for delta in deltas:
            self.apply_delta(delta)

    def get_snapshot(self) -> ReplaySnapshot:
        view = self._require_view()
        return ReplaySnapshot(
            cursor=view.cursor,
            episodes=tuple(view.episodes.values()),
            total_transitions=view.sampling_index.total_transitions,
            total_estimated_bytes=view.total_estimated_bytes,
        )

    def sample_coordinates(
        self,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[str, int]]:
        view = self._require_view()
        return self._sample_coordinates(view, batch_size, rng=rng)

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[object]:
        view = self._require_view()
        return [
            self._codec.get_transition(view.episodes[episode_id], transition_index)
            for episode_id, transition_index in self._sample_coordinates(
                view,
                batch_size,
                rng=rng,
            )
        ]

    def _materialize(
        self,
        cursor: ReplayCursor,
        source_episodes: Iterable[EpisodeEnvelope],
    ) -> _MaterializedView:
        episodes: OrderedDict[str, EpisodeEnvelope] = OrderedDict()
        episode_ids: list[str] = []
        cumulative_lengths: list[int] = []
        total_transitions = 0
        total_estimated_bytes = 0

        for episode in source_episodes:
            self._codec.validate(episode)
            if episode.episode_id in episodes:
                raise ReplayError(
                    f"materialized view contains duplicate episode_id "
                    f"{episode.episode_id!r}"
                )
            episodes[episode.episode_id] = episode
            total_transitions += self._codec.transition_count(episode)
            total_estimated_bytes += episode.estimated_bytes
            episode_ids.append(episode.episode_id)
            cumulative_lengths.append(total_transitions)

        return _MaterializedView(
            cursor=cursor,
            episodes=MappingProxyType(episodes),
            sampling_index=_SamplingIndex(
                episode_ids=tuple(episode_ids),
                cumulative_lengths=tuple(cumulative_lengths),
                total_transitions=total_transitions,
            ),
            total_estimated_bytes=total_estimated_bytes,
        )

    def _require_view(self) -> _MaterializedView:
        if self._view is None:
            raise ReplayError("load a snapshot before reading the local view")
        return self._view

    @staticmethod
    def _sample_coordinates(
        view: _MaterializedView,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[str, int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        index = view.sampling_index
        if index.total_transitions == 0:
            raise ReplayError("cannot sample an empty replay")

        coordinates: list[tuple[str, int]] = []
        for _ in range(batch_size):
            flat_index = rng.randrange(index.total_transitions)
            episode_index = bisect_right(index.cumulative_lengths, flat_index)
            episode_start = (
                index.cumulative_lengths[episode_index - 1] if episode_index else 0
            )
            coordinates.append(
                (
                    index.episode_ids[episode_index],
                    flat_index - episode_start,
                )
            )
        return coordinates
