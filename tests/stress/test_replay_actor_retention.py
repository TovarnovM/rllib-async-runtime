from __future__ import annotations

import pytest
import ray

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.replay.actor import ReplayActor


def make_episode(codec: FlatEpisodeCodec, sequence: int) -> EpisodeEnvelope:
    payload = codec.encode([sequence])
    return EpisodeEnvelope(
        episode_id=f"member-0/runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({"default_policy": 1}),
        env_steps=1,
        agent_steps=1,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


@pytest.mark.stress
def test_fifo_payload_and_journal_remain_bounded_under_sustained_ingest(
    ray_runtime: None,
) -> None:
    codec = FlatEpisodeCodec()
    replay = ReplayActor.remote(
        codec,
        capacity_transitions=64,
        capacity_bytes=100_000,
        journal_capacity=32,
        store_generation="sustained-ingest",
    )

    try:
        for sequence in range(10_000):
            ray.get(replay.commit_episode.remote(make_episode(codec, sequence)))

        stats = ray.get(replay.get_stats.remote())
        assert stats.episode_count == 64
        assert stats.total_transitions == 64
        assert stats.journal_entries == 32
        assert stats.deduplication_entries == 10_000
        assert stats.committed_episodes == 10_000
        assert stats.evicted_episodes == 10_000 - 64
    finally:
        ray.kill(replay)
