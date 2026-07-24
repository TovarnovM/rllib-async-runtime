"""Serializable replay synchronization values."""

from __future__ import annotations

from dataclasses import dataclass

from rllib_async.protocols.episodes import EpisodeEnvelope


@dataclass(frozen=True, slots=True)
class ReplayCursor:
    store_generation: str
    mutation_seq: int

    def __post_init__(self) -> None:
        if not self.store_generation:
            raise ValueError("store_generation must not be empty")
        if self.mutation_seq < 0:
            raise ValueError("mutation_seq must be non-negative")


@dataclass(frozen=True, slots=True)
class ReplayTransaction:
    """One atomic addition and every FIFO eviction caused by it."""

    mutation_seq: int
    added: EpisodeEnvelope
    evicted_episode_ids: tuple[str, ...]

    @property
    def estimated_bytes(self) -> int:
        evicted_id_bytes = sum(
            len(episode_id.encode("utf-8")) for episode_id in self.evicted_episode_ids
        )
        return self.added.estimated_bytes + evicted_id_bytes


@dataclass(frozen=True, slots=True)
class CommitAck:
    cursor: ReplayCursor
    committed: bool
    duplicate: bool
    evicted_episode_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    cursor: ReplayCursor
    episodes: tuple[EpisodeEnvelope, ...]
    total_transitions: int
    total_estimated_bytes: int


@dataclass(frozen=True, slots=True)
class ReplayDelta:
    base_cursor: ReplayCursor
    next_cursor: ReplayCursor
    transactions: tuple[ReplayTransaction, ...]
    full_resync_required: bool
    has_more: bool


@dataclass(frozen=True, slots=True)
class ReplayStats:
    cursor: ReplayCursor
    episode_count: int
    total_transitions: int
    total_estimated_bytes: int
    oldest_available_mutation_seq: int
    journal_entries: int
    deduplication_entries: int
    commit_attempts: int
    committed_episodes: int
    duplicate_commits: int
    rejected_commits: int
    conflicting_commits: int
    evicted_episodes: int


@dataclass(frozen=True, slots=True)
class ReplayCheckpoint:
    path: str
    format_version: int
    cursor: ReplayCursor
    size_bytes: int
    sha256: str
