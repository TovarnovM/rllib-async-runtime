from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import ray

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.replay.actor import ReplayActor
from rllib_async.replay.checkpoint import InvalidReplayCheckpointError


def make_episode(
    codec: FlatEpisodeCodec,
    *,
    runner_index: int,
    sequence: int,
    value: object,
) -> EpisodeEnvelope:
    payload = codec.encode([value])
    return EpisodeEnvelope(
        episode_id=f"member-0/runner-{runner_index}/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id=f"runner-{runner_index}",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({"default_policy": 5}),
        env_steps=1,
        agent_steps=1,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


@ray.remote(num_cpus=0)
def commit_from_producer(
    replay: object,
    episodes: Sequence[EpisodeEnvelope],
) -> list[tuple[object, object]]:
    acknowledgements = []
    for episode in episodes:
        committed = ray.get(replay.commit_episode.remote(episode))
        duplicate = ray.get(replay.commit_episode.remote(episode))
        acknowledgements.append((committed, duplicate))
    return acknowledgements


@pytest.mark.integration
def test_sixteen_concurrent_producers_preserve_atomic_manifest_and_metrics(
    ray_runtime: None,
) -> None:
    codec = FlatEpisodeCodec()
    producer_count = 16
    episodes_per_producer = 4
    expected_episodes = [
        make_episode(
            codec,
            runner_index=runner_index,
            sequence=sequence,
            value=(runner_index, sequence),
        )
        for runner_index in range(producer_count)
        for sequence in range(episodes_per_producer)
    ]
    replay = ReplayActor.remote(
        codec,
        capacity_transitions=len(expected_episodes),
        capacity_bytes=100_000,
        journal_capacity=len(expected_episodes),
        store_generation="concurrent-producers",
    )

    try:
        producer_results = ray.get(
            [
                commit_from_producer.remote(
                    replay,
                    [
                        episode
                        for episode in expected_episodes
                        if episode.runner_id == f"runner-{runner_index}"
                    ],
                )
                for runner_index in range(producer_count)
            ]
        )
        acknowledgement_pairs = [
            pair for producer_result in producer_results for pair in producer_result
        ]
        committed = [pair[0] for pair in acknowledgement_pairs]
        duplicates = [pair[1] for pair in acknowledgement_pairs]

        assert all(ack.committed and not ack.duplicate for ack in committed)
        assert all(not ack.committed and ack.duplicate for ack in duplicates)
        assert sorted(ack.cursor.mutation_seq for ack in committed) == list(
            range(1, len(expected_episodes) + 1)
        )

        snapshot = ray.get(replay.get_snapshot.remote())
        stats = ray.get(replay.get_stats.remote())
        assert {episode.episode_id for episode in snapshot.episodes} == {
            episode.episode_id for episode in expected_episodes
        }
        assert snapshot.total_transitions == len(expected_episodes)
        assert snapshot.cursor.mutation_seq == len(expected_episodes)
        assert stats.commit_attempts == len(expected_episodes) * 2
        assert stats.committed_episodes == len(expected_episodes)
        assert stats.duplicate_commits == len(expected_episodes)
        assert stats.rejected_commits == 0
        assert stats.evicted_episodes == 0
    finally:
        ray.kill(replay)


@pytest.mark.integration
def test_checkpoint_restore_preserves_payload_journal_and_evicted_deduplication(
    ray_runtime: None,
    tmp_path: Path,
) -> None:
    codec = FlatEpisodeCodec()
    source = ReplayActor.remote(
        codec,
        capacity_transitions=2,
        capacity_bytes=100_000,
        journal_capacity=8,
        store_generation="checkpoint-source",
    )
    restored = ReplayActor.remote(
        codec,
        capacity_transitions=99,
        capacity_bytes=999_999,
        journal_capacity=99,
        store_generation="discarded-on-restore",
    )
    episodes = [
        make_episode(codec, runner_index=0, sequence=index, value=index)
        for index in range(4)
    ]
    checkpoint_path = tmp_path / "replay.snapshot"

    try:
        ray.get(source.commit_episode.remote(episodes[0]))
        cursor_before_suffix = ray.get(source.get_stats.remote()).cursor
        ray.get(source.commit_episode.remote(episodes[1]))
        ray.get(source.commit_episode.remote(episodes[2]))
        expected_snapshot = ray.get(source.get_snapshot.remote())
        expected_delta = ray.get(
            source.get_delta.remote(cursor_before_suffix, max_bytes=100_000)
        )

        checkpoint = ray.get(source.save_snapshot.remote(str(checkpoint_path)))
        assert checkpoint.path == str(checkpoint_path)
        assert checkpoint.cursor == expected_snapshot.cursor
        assert checkpoint.format_version == 2
        assert checkpoint.size_bytes == checkpoint_path.stat().st_size
        assert len(checkpoint.sha256) == 64

        restored_stats = ray.get(restored.load_snapshot.remote(str(checkpoint_path)))
        restored_snapshot = ray.get(restored.get_snapshot.remote())
        restored_delta = ray.get(
            restored.get_delta.remote(cursor_before_suffix, max_bytes=100_000)
        )
        assert restored_snapshot == expected_snapshot
        assert restored_delta == expected_delta
        assert restored_stats.cursor == expected_snapshot.cursor

        evicted_retry = ray.get(restored.commit_episode.remote(episodes[0]))
        assert not evicted_retry.committed and evicted_retry.duplicate
        assert ray.get(restored.get_snapshot.remote()) == expected_snapshot

        new_commit = ray.get(restored.commit_episode.remote(episodes[3]))
        assert new_commit.evicted_episode_ids == (episodes[1].episode_id,)
    finally:
        ray.kill(source)
        ray.kill(restored)


@pytest.mark.integration
def test_invalid_checkpoint_does_not_partially_replace_live_actor_state(
    ray_runtime: None,
    tmp_path: Path,
) -> None:
    codec = FlatEpisodeCodec()
    replay = ReplayActor.remote(
        codec,
        capacity_transitions=4,
        capacity_bytes=100_000,
        store_generation="unchanged-after-invalid-restore",
    )
    checkpoint_path = tmp_path / "corrupted.snapshot"
    checkpoint_path.write_bytes(b"not a replay checkpoint")

    try:
        ray.get(
            replay.commit_episode.remote(
                make_episode(codec, runner_index=0, sequence=0, value="retained")
            )
        )
        before = ray.get(replay.get_snapshot.remote())

        with pytest.raises(ray.exceptions.RayTaskError) as error:
            ray.get(replay.load_snapshot.remote(str(checkpoint_path)))

        assert error.value.as_instanceof_cause() is not None
        assert isinstance(
            error.value.as_instanceof_cause(),
            InvalidReplayCheckpointError,
        )
        assert ray.get(replay.get_snapshot.remote()) == before
    finally:
        ray.kill(replay)
