"""Deterministic in-process replay used as the correctness oracle."""

from __future__ import annotations

import hashlib
import pickle
import random
import uuid
from bisect import bisect_right
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from rllib_async.protocols.episodes import EpisodeCodec, EpisodeEnvelope
from rllib_async.protocols.replay import (
    CommitAck,
    ReplayCursor,
    ReplayDelta,
    ReplaySnapshot,
    ReplayStats,
    ReplayTransaction,
)

EPISODE_STORE_STATE_VERSION = 2


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


class InvalidEpisodeStoreStateError(ReplayError):
    """A checkpoint cannot safely reconstruct an authoritative store."""


@dataclass(frozen=True, slots=True)
class EpisodeStoreState:
    """Versioned, pickle-safe state owned by an authoritative replay actor."""

    format_version: int
    codec_id: str
    codec_schema_version: int
    capacity_transitions: int
    capacity_bytes: int
    journal_capacity: int
    store_generation: str
    mutation_seq: int
    episodes: tuple[EpisodeEnvelope, ...]
    episode_records: tuple[tuple[str, bytes, int, int], ...]
    journal_base_manifest: tuple[tuple[str, int, int], ...]
    journal: tuple[ReplayTransaction, ...]
    commit_attempts: int
    committed_episodes: int
    duplicate_commits: int
    rejected_commits: int
    conflicting_commits: int
    evicted_episodes: int


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
        if not _is_positive_integer(capacity_transitions):
            raise ValueError("capacity_transitions must be positive")
        if not _is_positive_integer(capacity_bytes):
            raise ValueError("capacity_bytes must be positive")
        if not _is_positive_integer(journal_capacity):
            raise ValueError("journal_capacity must be positive")
        if store_generation is not None and (
            not isinstance(store_generation, str) or not store_generation
        ):
            raise ValueError("store_generation must be a non-empty string")

        self._codec = codec
        self._capacity_transitions = capacity_transitions
        self._capacity_bytes = capacity_bytes
        self._journal_capacity = journal_capacity
        self._store_generation = store_generation or uuid.uuid4().hex
        self._mutation_seq = 0
        self._episodes: OrderedDict[str, EpisodeEnvelope] = OrderedDict()
        self._transition_counts: dict[str, int] = {}
        self._episode_records: dict[str, tuple[bytes, int, int]] = {}
        self._total_transitions = 0
        self._total_estimated_bytes = 0
        self._journal_base_manifest: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._journal: deque[ReplayTransaction] = deque()
        self._commit_attempts = 0
        self._committed_episodes = 0
        self._duplicate_commits = 0
        self._rejected_commits = 0
        self._conflicting_commits = 0
        self._evicted_episodes = 0

    @property
    def cursor(self) -> ReplayCursor:
        return ReplayCursor(self._store_generation, self._mutation_seq)

    def commit_episode(self, episode: EpisodeEnvelope) -> CommitAck:
        """Atomically add an episode and evict whole oldest episodes."""
        self._commit_attempts += 1
        try:
            self._codec.validate(episode)
            transition_count = self._codec.transition_count(episode)
            fingerprint = self._fingerprint(episode)
        except Exception:
            self._rejected_commits += 1
            raise

        if (
            transition_count > self._capacity_transitions
            or episode.estimated_bytes > self._capacity_bytes
        ):
            self._rejected_commits += 1
            raise EpisodeTooLargeError(
                f"episode {episode.episode_id!r} requires "
                f"{transition_count} transitions/{episode.estimated_bytes} bytes; "
                f"store capacity is {self._capacity_transitions} "
                f"transitions/{self._capacity_bytes} bytes"
            )

        existing_record = self._episode_records.get(episode.episode_id)
        if existing_record is not None:
            if existing_record[0] != fingerprint:
                self._rejected_commits += 1
                self._conflicting_commits += 1
                raise DuplicateEpisodeConflictError(
                    f"episode_id {episode.episode_id!r} already has different content"
                )
            self._duplicate_commits += 1
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
        journal_base_manifest = self._journal_base_manifest.copy()
        journal = deque(self._journal)
        if not journal:
            journal_base_manifest = _retention_manifest(
                self._episodes,
                self._transition_counts,
            )
        journal.append(transaction)
        while len(journal) > self._journal_capacity:
            compacted = journal.popleft()
            _advance_retention_manifest(
                journal_base_manifest,
                compacted,
                codec=self._codec,
            )

        self._episodes = episodes
        self._transition_counts = transition_counts
        self._total_transitions = total_transitions
        self._total_estimated_bytes = total_estimated_bytes
        self._mutation_seq = mutation_seq
        self._episode_records[episode.episode_id] = (
            fingerprint,
            transition_count,
            episode.estimated_bytes,
        )
        self._journal_base_manifest = journal_base_manifest
        self._journal = journal
        self._committed_episodes += 1
        self._evicted_episodes += len(evicted_episode_ids)

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
        producer_episode_counts: Counter[str] = Counter()
        producer_transition_counts: Counter[str] = Counter()
        for episode_id, episode in self._episodes.items():
            producer_episode_counts[episode.producer_member_id] += 1
            producer_transition_counts[episode.producer_member_id] += (
                self._transition_counts[episode_id]
            )
        return ReplayStats(
            cursor=self.cursor,
            episode_count=len(self._episodes),
            total_transitions=self._total_transitions,
            total_estimated_bytes=self._total_estimated_bytes,
            producer_episode_counts=tuple(sorted(producer_episode_counts.items())),
            producer_transition_counts=tuple(
                sorted(producer_transition_counts.items())
            ),
            oldest_available_mutation_seq=oldest_available,
            journal_entries=len(self._journal),
            deduplication_entries=len(self._episode_records),
            commit_attempts=self._commit_attempts,
            committed_episodes=self._committed_episodes,
            duplicate_commits=self._duplicate_commits,
            rejected_commits=self._rejected_commits,
            conflicting_commits=self._conflicting_commits,
            evicted_episodes=self._evicted_episodes,
        )

    def export_state(self) -> EpisodeStoreState:
        """Return a self-contained immutable checkpoint state."""
        return EpisodeStoreState(
            format_version=EPISODE_STORE_STATE_VERSION,
            codec_id=self._codec.codec_id,
            codec_schema_version=self._codec.schema_version,
            capacity_transitions=self._capacity_transitions,
            capacity_bytes=self._capacity_bytes,
            journal_capacity=self._journal_capacity,
            store_generation=self._store_generation,
            mutation_seq=self._mutation_seq,
            episodes=tuple(self._episodes.values()),
            episode_records=tuple(
                (
                    episode_id,
                    fingerprint,
                    transition_count,
                    estimated_bytes,
                )
                for episode_id, (
                    fingerprint,
                    transition_count,
                    estimated_bytes,
                ) in self._episode_records.items()
            ),
            journal_base_manifest=tuple(
                (episode_id, transition_count, estimated_bytes)
                for episode_id, (
                    transition_count,
                    estimated_bytes,
                ) in self._journal_base_manifest.items()
            ),
            journal=tuple(self._journal),
            commit_attempts=self._commit_attempts,
            committed_episodes=self._committed_episodes,
            duplicate_commits=self._duplicate_commits,
            rejected_commits=self._rejected_commits,
            conflicting_commits=self._conflicting_commits,
            evicted_episodes=self._evicted_episodes,
        )

    @classmethod
    def from_state(
        cls,
        codec: EpisodeCodec,
        state: EpisodeStoreState,
    ) -> EpisodeStore:
        """Validate checkpoint state before constructing a replacement store."""
        if not isinstance(state, EpisodeStoreState):
            raise InvalidEpisodeStoreStateError(
                "checkpoint payload is not an EpisodeStoreState"
            )
        if state.format_version != EPISODE_STORE_STATE_VERSION:
            raise InvalidEpisodeStoreStateError(
                f"unsupported store state version {state.format_version}"
            )
        if (
            state.codec_id != codec.codec_id
            or state.codec_schema_version != codec.schema_version
        ):
            raise InvalidEpisodeStoreStateError(
                "checkpoint codec does not match the actor codec"
            )

        try:
            store = cls(
                codec,
                capacity_transitions=state.capacity_transitions,
                capacity_bytes=state.capacity_bytes,
                journal_capacity=state.journal_capacity,
                store_generation=state.store_generation,
            )
        except (TypeError, ValueError) as error:
            raise InvalidEpisodeStoreStateError(
                "checkpoint retention configuration is invalid"
            ) from error

        if not _is_non_negative_integer(state.mutation_seq):
            raise InvalidEpisodeStoreStateError("mutation_seq must be non-negative")

        episodes: OrderedDict[str, EpisodeEnvelope] = OrderedDict()
        transition_counts: dict[str, int] = {}
        total_transitions = 0
        total_estimated_bytes = 0
        for episode in state.episodes:
            try:
                codec.validate(episode)
                transition_count = codec.transition_count(episode)
            except Exception as error:
                raise InvalidEpisodeStoreStateError(
                    f"checkpoint episode {getattr(episode, 'episode_id', None)!r} "
                    "is invalid"
                ) from error
            if episode.episode_id in episodes:
                raise InvalidEpisodeStoreStateError(
                    f"duplicate live episode_id {episode.episode_id!r}"
                )
            episodes[episode.episode_id] = episode
            transition_counts[episode.episode_id] = transition_count
            total_transitions += transition_count
            total_estimated_bytes += episode.estimated_bytes

        if (
            total_transitions > state.capacity_transitions
            or total_estimated_bytes > state.capacity_bytes
        ):
            raise InvalidEpisodeStoreStateError(
                "checkpoint manifest exceeds its retention capacity"
            )

        fingerprints: dict[str, bytes] = {}
        retention_metadata: OrderedDict[str, tuple[int, int]] = OrderedDict()
        if type(state.episode_records) is not tuple:
            raise InvalidEpisodeStoreStateError(
                "checkpoint episode records must be a tuple"
            )
        for entry in state.episode_records:
            if (
                type(entry) is not tuple
                or len(entry) != 4
                or not isinstance(entry[0], str)
                or not entry[0]
                or type(entry[1]) is not bytes
                or len(entry[1]) != hashlib.sha256().digest_size
                or not _is_positive_integer(entry[2])
                or not _is_positive_integer(entry[3])
            ):
                raise InvalidEpisodeStoreStateError(
                    "checkpoint contains an invalid episode record"
                )
            episode_id, fingerprint, transition_count, estimated_bytes = entry
            if episode_id in fingerprints:
                raise InvalidEpisodeStoreStateError(
                    f"duplicate episode record for episode_id {episode_id!r}"
                )
            if (
                transition_count > state.capacity_transitions
                or estimated_bytes > state.capacity_bytes
            ):
                raise InvalidEpisodeStoreStateError(
                    f"episode {episode_id!r} retention metadata exceeds capacity"
                )
            fingerprints[episode_id] = fingerprint
            retention_metadata[episode_id] = (
                transition_count,
                estimated_bytes,
            )

        for episode in episodes.values():
            if fingerprints.get(episode.episode_id) != cls._fingerprint(episode):
                raise InvalidEpisodeStoreStateError(
                    f"live episode {episode.episode_id!r} has no matching fingerprint"
                )
            if retention_metadata[episode.episode_id] != (
                transition_counts[episode.episode_id],
                episode.estimated_bytes,
            ):
                raise InvalidEpisodeStoreStateError(
                    f"live episode {episode.episode_id!r} has inconsistent "
                    "retention metadata"
                )

        if type(state.journal) is not tuple:
            raise InvalidEpisodeStoreStateError("checkpoint journal must be a tuple")
        journal = state.journal
        expected_journal_length = min(state.mutation_seq, state.journal_capacity)
        if len(journal) != expected_journal_length:
            raise InvalidEpisodeStoreStateError(
                "checkpoint journal length is inconsistent"
            )
        journal_base_mutation_seq = state.mutation_seq - len(journal)
        journal_base_manifest: OrderedDict[str, tuple[int, int]] = OrderedDict()
        if type(state.journal_base_manifest) is not tuple:
            raise InvalidEpisodeStoreStateError(
                "checkpoint journal base manifest must be a tuple"
            )
        for entry in state.journal_base_manifest:
            if (
                type(entry) is not tuple
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not entry[0]
                or not _is_positive_integer(entry[1])
                or not _is_positive_integer(entry[2])
            ):
                raise InvalidEpisodeStoreStateError(
                    "checkpoint journal base manifest contains an invalid entry"
                )
            episode_id, transition_count, estimated_bytes = entry
            if episode_id in journal_base_manifest:
                raise InvalidEpisodeStoreStateError(
                    f"duplicate journal base episode_id {episode_id!r}"
                )
            if episode_id not in fingerprints:
                raise InvalidEpisodeStoreStateError(
                    f"journal base episode {episode_id!r} has no fingerprint"
                )
            if retention_metadata[episode_id] != (
                transition_count,
                estimated_bytes,
            ):
                raise InvalidEpisodeStoreStateError(
                    f"journal base episode {episode_id!r} has inconsistent "
                    "retention metadata"
                )
            journal_base_manifest[episode_id] = (
                transition_count,
                estimated_bytes,
            )
        expected_journal_base_manifest = _retention_manifest_from_history(
            tuple(retention_metadata.items())[:journal_base_mutation_seq],
            capacity_transitions=state.capacity_transitions,
            capacity_bytes=state.capacity_bytes,
        )
        if journal_base_manifest != expected_journal_base_manifest:
            raise InvalidEpisodeStoreStateError(
                "checkpoint journal base manifest is incomplete or inconsistent"
            )

        expected_sequences = range(
            state.mutation_seq - len(journal) + 1,
            state.mutation_seq + 1,
        )
        fingerprint_ids = tuple(fingerprints)
        expected_journal_episode_ids = fingerprint_ids[journal_base_mutation_seq:]
        actual_journal_episode_ids: list[str] = []
        reconstructed_manifest = journal_base_manifest.copy()
        reconstructed_transitions = sum(
            item[0] for item in reconstructed_manifest.values()
        )
        reconstructed_bytes = sum(item[1] for item in reconstructed_manifest.values())
        for transaction, expected_sequence in zip(
            journal,
            expected_sequences,
            strict=True,
        ):
            if not isinstance(transaction, ReplayTransaction):
                raise InvalidEpisodeStoreStateError(
                    "checkpoint journal contains an invalid transaction"
                )
            if transaction.mutation_seq != expected_sequence:
                raise InvalidEpisodeStoreStateError(
                    "checkpoint journal is not a contiguous mutation suffix"
                )
            try:
                codec.validate(transaction.added)
            except Exception as error:
                raise InvalidEpisodeStoreStateError(
                    f"journal episode {transaction.added.episode_id!r} is invalid"
                ) from error
            if fingerprints.get(transaction.added.episode_id) != cls._fingerprint(
                transaction.added
            ):
                raise InvalidEpisodeStoreStateError(
                    f"journal episode {transaction.added.episode_id!r} "
                    "has no matching fingerprint"
                )
            if retention_metadata[transaction.added.episode_id] != (
                codec.transition_count(transaction.added),
                transaction.added.estimated_bytes,
            ):
                raise InvalidEpisodeStoreStateError(
                    f"journal episode {transaction.added.episode_id!r} has "
                    "inconsistent retention metadata"
                )
            if type(transaction.evicted_episode_ids) is not tuple or any(
                not isinstance(episode_id, str) or not episode_id
                for episode_id in transaction.evicted_episode_ids
            ):
                raise InvalidEpisodeStoreStateError(
                    "checkpoint journal contains invalid eviction IDs"
                )
            actual_journal_episode_ids.append(transaction.added.episode_id)
            try:
                reconstructed_transitions, reconstructed_bytes = (
                    _apply_retention_transaction(
                        reconstructed_manifest,
                        transaction,
                        codec=codec,
                        capacity_transitions=state.capacity_transitions,
                        capacity_bytes=state.capacity_bytes,
                        total_transitions=reconstructed_transitions,
                        total_estimated_bytes=reconstructed_bytes,
                    )
                )
            except ReplayError as error:
                raise InvalidEpisodeStoreStateError(
                    "checkpoint journal eviction semantics are inconsistent"
                ) from error

        if tuple(actual_journal_episode_ids) != expected_journal_episode_ids:
            raise InvalidEpisodeStoreStateError(
                "checkpoint journal additions do not match commit order"
            )
        expected_live_manifest = OrderedDict(
            (
                episode_id,
                (transition_counts[episode_id], episode.estimated_bytes),
            )
            for episode_id, episode in episodes.items()
        )
        if reconstructed_manifest != expected_live_manifest:
            raise InvalidEpisodeStoreStateError(
                "checkpoint journal suffix does not reconstruct the retained manifest"
            )

        metric_names = (
            "commit_attempts",
            "committed_episodes",
            "duplicate_commits",
            "rejected_commits",
            "conflicting_commits",
            "evicted_episodes",
        )
        if any(
            not _is_non_negative_integer(getattr(state, name)) for name in metric_names
        ):
            raise InvalidEpisodeStoreStateError(
                "checkpoint commit metrics must be non-negative integers"
            )
        if state.committed_episodes != state.mutation_seq:
            raise InvalidEpisodeStoreStateError(
                "committed episode count does not match mutation_seq"
            )
        if state.commit_attempts != (
            state.committed_episodes + state.duplicate_commits + state.rejected_commits
        ):
            raise InvalidEpisodeStoreStateError(
                "commit attempt metrics are inconsistent"
            )
        if state.conflicting_commits > state.rejected_commits:
            raise InvalidEpisodeStoreStateError(
                "conflicting commit count exceeds rejected commits"
            )
        if state.evicted_episodes != state.committed_episodes - len(episodes):
            raise InvalidEpisodeStoreStateError(
                "eviction count does not match the retained manifest"
            )
        if len(fingerprints) != state.committed_episodes:
            raise InvalidEpisodeStoreStateError(
                "deduplication state does not cover every committed episode"
            )

        store._episodes = episodes
        store._transition_counts = transition_counts
        store._episode_records = {
            episode_id: (
                fingerprint,
                *retention_metadata[episode_id],
            )
            for episode_id, fingerprint in fingerprints.items()
        }
        store._total_transitions = total_transitions
        store._total_estimated_bytes = total_estimated_bytes
        store._mutation_seq = state.mutation_seq
        store._journal_base_manifest = journal_base_manifest
        store._journal = deque(journal)
        store._commit_attempts = state.commit_attempts
        store._committed_episodes = state.committed_episodes
        store._duplicate_commits = state.duplicate_commits
        store._rejected_commits = state.rejected_commits
        store._conflicting_commits = state.conflicting_commits
        store._evicted_episodes = state.evicted_episodes
        return store

    def _requires_full_resync(self, cursor: ReplayCursor) -> bool:
        if cursor.store_generation != self._store_generation:
            return True
        if cursor.mutation_seq > self._mutation_seq:
            return True
        if not self._journal:
            return cursor.mutation_seq != self._mutation_seq
        oldest_replayable_cursor = self._journal[0].mutation_seq - 1
        return cursor.mutation_seq < oldest_replayable_cursor


def _is_non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _retention_manifest_from_history(
    history: Sequence[tuple[str, tuple[int, int]]],
    *,
    capacity_transitions: int,
    capacity_bytes: int,
) -> OrderedDict[str, tuple[int, int]]:
    manifest: OrderedDict[str, tuple[int, int]] = OrderedDict()
    total_transitions = 0
    total_estimated_bytes = 0
    for episode_id, (transition_count, estimated_bytes) in history:
        manifest[episode_id] = (transition_count, estimated_bytes)
        total_transitions += transition_count
        total_estimated_bytes += estimated_bytes
        while (
            total_transitions > capacity_transitions
            or total_estimated_bytes > capacity_bytes
        ):
            _, (evicted_transitions, evicted_bytes) = manifest.popitem(last=False)
            total_transitions -= evicted_transitions
            total_estimated_bytes -= evicted_bytes
    return manifest


def _retention_manifest(
    episodes: OrderedDict[str, EpisodeEnvelope],
    transition_counts: dict[str, int],
) -> OrderedDict[str, tuple[int, int]]:
    return OrderedDict(
        (
            episode_id,
            (transition_counts[episode_id], episode.estimated_bytes),
        )
        for episode_id, episode in episodes.items()
    )


def _apply_retention_transaction(
    manifest: OrderedDict[str, tuple[int, int]],
    transaction: ReplayTransaction,
    *,
    codec: EpisodeCodec,
    capacity_transitions: int,
    capacity_bytes: int,
    total_transitions: int,
    total_estimated_bytes: int,
) -> tuple[int, int]:
    episode_id = transaction.added.episode_id
    if episode_id in manifest:
        raise ReplayError(f"journal adds existing episode_id {episode_id!r}")

    transition_count = codec.transition_count(transaction.added)
    estimated_bytes = transaction.added.estimated_bytes
    manifest[episode_id] = (transition_count, estimated_bytes)
    total_transitions += transition_count
    total_estimated_bytes += estimated_bytes
    expected_evictions: list[str] = []
    while (
        total_transitions > capacity_transitions
        or total_estimated_bytes > capacity_bytes
    ):
        evicted_id, (transition_count, estimated_bytes) = manifest.popitem(last=False)
        expected_evictions.append(evicted_id)
        total_transitions -= transition_count
        total_estimated_bytes -= estimated_bytes

    if transaction.evicted_episode_ids != tuple(expected_evictions):
        raise ReplayError(
            f"journal transaction {transaction.mutation_seq} evicts "
            f"{transaction.evicted_episode_ids!r}; expected "
            f"{tuple(expected_evictions)!r}"
        )
    return total_transitions, total_estimated_bytes


def _advance_retention_manifest(
    manifest: OrderedDict[str, tuple[int, int]],
    transaction: ReplayTransaction,
    *,
    codec: EpisodeCodec,
) -> None:
    episode_id = transaction.added.episode_id
    if episode_id in manifest:
        raise ReplayError(f"journal adds existing episode_id {episode_id!r}")
    manifest[episode_id] = (
        codec.transition_count(transaction.added),
        transaction.added.estimated_bytes,
    )
    for evicted_id in transaction.evicted_episode_ids:
        if not manifest or next(iter(manifest)) != evicted_id:
            raise ReplayError(
                f"journal transaction {transaction.mutation_seq} does not evict FIFO"
            )
        del manifest[evicted_id]


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
