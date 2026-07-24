"""Deterministic in-process replay used as the correctness oracle."""

from __future__ import annotations

import hashlib
import pickle
import random
import uuid
from bisect import bisect_right
from collections import OrderedDict, deque
from collections.abc import Sequence

from rllib_async.protocols.episodes import EpisodeCodec, EpisodeEnvelope
from rllib_async.protocols.replay import (
    CommitAck,
    ReplayCursor,
    ReplayDelta,
    ReplaySnapshot,
    ReplayStats,
    ReplayTransaction,
)


class ReplayError(RuntimeError):
    """Base class for replay contract failures."""


class EpisodeTooLargeError(ReplayError):
    """An episode cannot fit in an otherwise empty store."""


class DuplicateEpisodeConflictError(ReplayError):
    """An existing episode ID was reused for different content."""


class FullResyncRequiredError(ReplayError):
    """A local view cannot safely apply the supplied delta."""


class CursorMismatchError(ReplayError):
    """A delta does not start at the local view's current cursor."""


class SnapshotTooLargeError(ReplayError):
    """A complete snapshot exceeds the caller's requested byte budget."""


class EpisodeStore:
    """FIFO authoritative episode store without Ray process machinery."""

    def __init__(
        self,
        codec: EpisodeCodec,
        *,
        capacity_transitions: int,
        capacity_bytes: int,
        journal_capacity: int = 1_024,
        store_generation: str | None = None,
    ) -> None:
        if capacity_transitions < 1:
            raise ValueError("capacity_transitions must be positive")
        if capacity_bytes < 1:
            raise ValueError("capacity_bytes must be positive")
        if journal_capacity < 1:
            raise ValueError("journal_capacity must be positive")

        self._codec = codec
        self._capacity_transitions = capacity_transitions
        self._capacity_bytes = capacity_bytes
        self._journal_capacity = journal_capacity
        self._store_generation = store_generation or uuid.uuid4().hex
        self._mutation_seq = 0
        self._episodes: OrderedDict[str, EpisodeEnvelope] = OrderedDict()
        self._transition_counts: dict[str, int] = {}
        self._episode_fingerprints: dict[str, bytes] = {}
        self._total_transitions = 0
        self._total_estimated_bytes = 0
        self._journal: deque[ReplayTransaction] = deque()

    @property
    def cursor(self) -> ReplayCursor:
        return ReplayCursor(self._store_generation, self._mutation_seq)

    def commit_episode(self, episode: EpisodeEnvelope) -> CommitAck:
        """Atomically add an episode and evict whole oldest episodes."""
        self._codec.validate(episode)
        transition_count = self._codec.transition_count(episode)
        if (
            transition_count > self._capacity_transitions
            or episode.estimated_bytes > self._capacity_bytes
        ):
            raise EpisodeTooLargeError(
                f"episode {episode.episode_id!r} requires "
                f"{transition_count} transitions/{episode.estimated_bytes} bytes; "
                f"store capacity is {self._capacity_transitions} "
                f"transitions/{self._capacity_bytes} bytes"
            )

        fingerprint = self._fingerprint(episode)
        existing_fingerprint = self._episode_fingerprints.get(episode.episode_id)
        if existing_fingerprint is not None:
            if existing_fingerprint != fingerprint:
                raise DuplicateEpisodeConflictError(
                    f"episode_id {episode.episode_id!r} already has different content"
                )
            return CommitAck(
                cursor=self.cursor,
                committed=False,
                duplicate=True,
            )

        episodes = self._episodes.copy()
        transition_counts = self._transition_counts.copy()
        total_transitions = self._total_transitions + transition_count
        total_estimated_bytes = self._total_estimated_bytes + episode.estimated_bytes
        episodes[episode.episode_id] = episode
        transition_counts[episode.episode_id] = transition_count

        evicted_episode_ids: list[str] = []
        while (
            total_transitions > self._capacity_transitions
            or total_estimated_bytes > self._capacity_bytes
        ):
            evicted_id, evicted = episodes.popitem(last=False)
            evicted_episode_ids.append(evicted_id)
            total_transitions -= transition_counts.pop(evicted_id)
            total_estimated_bytes -= evicted.estimated_bytes

        mutation_seq = self._mutation_seq + 1
        transaction = ReplayTransaction(
            mutation_seq=mutation_seq,
            added=episode,
            evicted_episode_ids=tuple(evicted_episode_ids),
        )

        self._episodes = episodes
        self._transition_counts = transition_counts
        self._total_transitions = total_transitions
        self._total_estimated_bytes = total_estimated_bytes
        self._mutation_seq = mutation_seq
        self._episode_fingerprints[episode.episode_id] = fingerprint
        self._journal.append(transaction)
        while len(self._journal) > self._journal_capacity:
            self._journal.popleft()

        return CommitAck(
            cursor=self.cursor,
            committed=True,
            duplicate=False,
            evicted_episode_ids=tuple(evicted_episode_ids),
        )

    @staticmethod
    def _fingerprint(episode: EpisodeEnvelope) -> bytes:
        encoded = pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(encoded).digest()

    def get_snapshot(self, max_bytes: int | None = None) -> ReplaySnapshot:
        if max_bytes is not None:
            if max_bytes < 1:
                raise ValueError("max_bytes must be positive")
            if self._total_estimated_bytes > max_bytes:
                raise SnapshotTooLargeError(
                    f"snapshot requires {self._total_estimated_bytes} bytes; "
                    f"budget is {max_bytes}"
                )
        return ReplaySnapshot(
            cursor=self.cursor,
            episodes=tuple(self._episodes.values()),
            total_transitions=self._total_transitions,
            total_estimated_bytes=self._total_estimated_bytes,
        )

    def get_delta(self, cursor: ReplayCursor, *, max_bytes: int) -> ReplayDelta:
        """Return a bounded suffix of journal transactions.

        ``max_bytes`` is a soft bound for the first pending transaction so a
        caller can always make progress. Subsequent transactions are included
        only while their estimated combined size remains within the budget.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if self._requires_full_resync(cursor):
            return ReplayDelta(
                base_cursor=cursor,
                next_cursor=self.cursor,
                transactions=(),
                full_resync_required=True,
                has_more=False,
            )

        pending = [
            transaction
            for transaction in self._journal
            if transaction.mutation_seq > cursor.mutation_seq
        ]
        selected: list[ReplayTransaction] = []
        selected_bytes = 0
        for transaction in pending:
            if selected and selected_bytes + transaction.estimated_bytes > max_bytes:
                break
            selected.append(transaction)
            selected_bytes += transaction.estimated_bytes

        next_cursor = (
            ReplayCursor(self._store_generation, selected[-1].mutation_seq)
            if selected
            else cursor
        )
        return ReplayDelta(
            base_cursor=cursor,
            next_cursor=next_cursor,
            transactions=tuple(selected),
            full_resync_required=False,
            has_more=next_cursor.mutation_seq < self._mutation_seq,
        )

    def get_stats(self) -> ReplayStats:
        oldest_available = (
            self._journal[0].mutation_seq if self._journal else self._mutation_seq + 1
        )
        return ReplayStats(
            cursor=self.cursor,
            episode_count=len(self._episodes),
            total_transitions=self._total_transitions,
            total_estimated_bytes=self._total_estimated_bytes,
            oldest_available_mutation_seq=oldest_available,
        )

    def _requires_full_resync(self, cursor: ReplayCursor) -> bool:
        if cursor.store_generation != self._store_generation:
            return True
        if cursor.mutation_seq > self._mutation_seq:
            return True
        if not self._journal:
            return cursor.mutation_seq != self._mutation_seq
        oldest_replayable_cursor = self._journal[0].mutation_seq - 1
        return cursor.mutation_seq < oldest_replayable_cursor


class ReferenceFastReplay:
    """Simple materialized replay view and uniform transition sampler."""

    def __init__(self, codec: EpisodeCodec) -> None:
        self._codec = codec
        self._cursor: ReplayCursor | None = None
        self._episodes: OrderedDict[str, EpisodeEnvelope] = OrderedDict()

    @property
    def cursor(self) -> ReplayCursor | None:
        return self._cursor

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._episodes)

    def load_snapshot(self, snapshot: ReplaySnapshot) -> None:
        episodes: OrderedDict[str, EpisodeEnvelope] = OrderedDict()
        total_transitions = 0
        total_estimated_bytes = 0
        for episode in snapshot.episodes:
            self._codec.validate(episode)
            if episode.episode_id in episodes:
                raise ReplayError(
                    f"snapshot contains duplicate episode_id {episode.episode_id!r}"
                )
            episodes[episode.episode_id] = episode
            total_transitions += self._codec.transition_count(episode)
            total_estimated_bytes += episode.estimated_bytes
        if total_transitions != snapshot.total_transitions:
            raise ReplayError("snapshot transition total is inconsistent")
        if total_estimated_bytes != snapshot.total_estimated_bytes:
            raise ReplayError("snapshot byte total is inconsistent")
        self._episodes = episodes
        self._cursor = snapshot.cursor

    def apply_delta(self, delta: ReplayDelta) -> None:
        if delta.full_resync_required:
            raise FullResyncRequiredError("authoritative replay requires a snapshot")
        if self._cursor is None:
            raise CursorMismatchError("load a snapshot before applying deltas")
        if delta.base_cursor != self._cursor:
            raise CursorMismatchError(
                f"local cursor {self._cursor!r} does not match "
                f"delta base {delta.base_cursor!r}"
            )

        episodes = self._episodes.copy()
        expected_mutation_seq = self._cursor.mutation_seq
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
                del episodes[evicted_id]

        if delta.next_cursor.store_generation != self._cursor.store_generation:
            raise CursorMismatchError("delta changes store generation")
        if delta.next_cursor.mutation_seq != expected_mutation_seq:
            raise CursorMismatchError(
                "delta next cursor does not match its transaction suffix"
            )
        self._episodes = episodes
        self._cursor = delta.next_cursor

    def get_snapshot(self) -> ReplaySnapshot:
        if self._cursor is None:
            raise ReplayError("load a snapshot before reading the local view")
        episodes = tuple(self._episodes.values())
        return ReplaySnapshot(
            cursor=self._cursor,
            episodes=episodes,
            total_transitions=sum(
                self._codec.transition_count(episode) for episode in episodes
            ),
            total_estimated_bytes=sum(episode.estimated_bytes for episode in episodes),
        )

    def sample_coordinates(
        self,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[str, int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        episode_ids: list[str] = []
        cumulative_lengths: list[int] = []
        total_transitions = 0
        for episode_id, episode in self._episodes.items():
            total_transitions += self._codec.transition_count(episode)
            episode_ids.append(episode_id)
            cumulative_lengths.append(total_transitions)
        if total_transitions == 0:
            raise ReplayError("cannot sample an empty replay")

        coordinates: list[tuple[str, int]] = []
        for _ in range(batch_size):
            flat_index = rng.randrange(total_transitions)
            episode_index = bisect_right(cumulative_lengths, flat_index)
            episode_start = (
                cumulative_lengths[episode_index - 1] if episode_index else 0
            )
            coordinates.append((episode_ids[episode_index], flat_index - episode_start))
        return coordinates

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[object]:
        return [
            self._codec.get_transition(self._episodes[episode_id], index)
            for episode_id, index in self.sample_coordinates(batch_size, rng=rng)
        ]

    def apply_deltas(self, deltas: Sequence[ReplayDelta]) -> None:
        for delta in deltas:
            self.apply_delta(delta)
