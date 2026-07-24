"""Single-owner Ray actor for authoritative whole-episode replay."""

from __future__ import annotations

import ray

from rllib_async.protocols.episodes import EpisodeCodec, EpisodeEnvelope
from rllib_async.protocols.replay import (
    CommitAck,
    ReplayCheckpoint,
    ReplayCursor,
    ReplayDelta,
    ReplaySnapshot,
    ReplayStats,
)
from rllib_async.replay.checkpoint import (
    read_replay_checkpoint,
    write_replay_checkpoint,
)
from rllib_async.replay.reference import EpisodeStore


@ray.remote(max_concurrency=1)
class ReplayActor:
    """Serialize authoritative replay mutations behind one Ray actor."""

    def __init__(
        self,
        codec: EpisodeCodec,
        *,
        capacity_transitions: int,
        capacity_bytes: int,
        journal_capacity: int = 1_024,
        store_generation: str | None = None,
    ) -> None:
        self._codec = codec
        self._store = EpisodeStore(
            codec,
            capacity_transitions=capacity_transitions,
            capacity_bytes=capacity_bytes,
            journal_capacity=journal_capacity,
            store_generation=store_generation,
        )

    def commit_episode(self, episode: EpisodeEnvelope) -> CommitAck:
        return self._store.commit_episode(episode)

    def get_delta(self, cursor: ReplayCursor, *, max_bytes: int) -> ReplayDelta:
        return self._store.get_delta(cursor, max_bytes=max_bytes)

    def get_snapshot(self, max_bytes: int | None = None) -> ReplaySnapshot:
        return self._store.get_snapshot(max_bytes=max_bytes)

    def get_stats(self) -> ReplayStats:
        return self._store.get_stats()

    def save_snapshot(self, path: str) -> ReplayCheckpoint:
        return write_replay_checkpoint(path, self._store.export_state())

    def load_snapshot(self, path: str) -> ReplayStats:
        state = read_replay_checkpoint(path)
        restored = EpisodeStore.from_state(self._codec, state)
        self._store = restored
        return self._store.get_stats()
