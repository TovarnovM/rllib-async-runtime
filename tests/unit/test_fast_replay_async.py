from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.replay import (
    FastReplay,
    IndexRebuildError,
    ReplayClosedError,
)
from rllib_async.replay.reference import EpisodeStore


def make_episode(
    codec: FlatEpisodeCodec,
    sequence: int,
    transitions: list[object],
) -> EpisodeEnvelope:
    payload = codec.encode(transitions)
    transition_count = len(payload.encoded_transitions)
    return EpisodeEnvelope(
        episode_id=f"member-0/runner-0/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({"default_policy": 5}),
        env_steps=transition_count,
        agent_steps=transition_count,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


class ControlledBuildFastReplay(FastReplay):
    def __init__(self, codec: FlatEpisodeCodec) -> None:
        super().__init__(codec)
        self._build_gates: dict[int, tuple[threading.Event, threading.Event]] = {}
        self._fail_once: set[int] = set()

    def block_build(
        self,
        mutation_seq: int,
    ) -> tuple[threading.Event, threading.Event]:
        started = threading.Event()
        release = threading.Event()
        self._build_gates[mutation_seq] = (started, release)
        return started, release

    def fail_build_once(self, mutation_seq: int) -> None:
        self._fail_once.add(mutation_seq)

    def _build_view(self, request):  # type: ignore[no-untyped-def]
        if request.cursor.mutation_seq in self._fail_once:
            self._fail_once.remove(request.cursor.mutation_seq)
            raise RuntimeError("controlled rebuild failure")
        gate = self._build_gates.get(request.cursor.mutation_seq)
        if gate is not None:
            started, release = gate
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("controlled rebuild was not released")
        return super()._build_view(request)


class BlockingSampleCodec(FlatEpisodeCodec):
    def __init__(self) -> None:
        self.block_episode_id: str | None = None
        self.sample_started = threading.Event()
        self.release_sample = threading.Event()

    def get_transition(self, episode: EpisodeEnvelope, index: int) -> object:
        if episode.episode_id == self.block_episode_id:
            self.sample_started.set()
            if not self.release_sample.wait(timeout=5):
                raise TimeoutError("controlled sample was not released")
        return super().get_transition(episode, index)


def test_sampling_continues_on_old_view_while_index_rebuilds() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1,
        capacity_bytes=10_000,
        store_generation="background-sampling",
    )
    old = make_episode(codec, 0, [("old", 0)])
    new = make_episode(codec, 1, [("new", 0)])
    store.commit_episode(old)
    replay = ControlledBuildFastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    started, release = replay.block_build(2)

    store.commit_episode(new)
    assert replay.cursor is not None
    replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))

    assert started.wait(timeout=2)
    assert replay.cursor == store.cursor
    assert replay.active_cursor != store.cursor
    assert replay.sample(10, rng=random.Random(1)) == [("old", 0)] * 10

    release.set()
    replay.wait_for_idle(timeout=2)
    assert replay.active_cursor == store.cursor
    assert replay.sample(10, rng=random.Random(1)) == [("new", 0)] * 10
    replay.close()


def test_stale_background_build_is_never_published() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1,
        capacity_bytes=10_000,
        store_generation="stale-build",
    )
    first = make_episode(codec, 0, [("first", 0)])
    second = make_episode(codec, 1, [("second", 0)])
    third = make_episode(codec, 2, [("third", 0)])
    store.commit_episode(first)
    replay = ControlledBuildFastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    second_started, release_second = replay.block_build(2)
    third_started, release_third = replay.block_build(3)

    store.commit_episode(second)
    assert replay.cursor is not None
    replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))
    assert second_started.wait(timeout=2)

    store.commit_episode(third)
    assert replay.cursor is not None
    replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))
    release_second.set()

    assert third_started.wait(timeout=2)
    assert replay.active_cursor is not None
    assert replay.active_cursor.mutation_seq == 1
    assert replay.sample(1, rng=random.Random(2)) == [("first", 0)]

    release_third.set()
    replay.wait_for_idle(timeout=2)
    assert replay.active_cursor == store.cursor
    assert replay.sample(1, rng=random.Random(2)) == [("third", 0)]
    assert replay.get_stats().discarded_rebuilds == 1
    replay.close()


def test_reader_lease_keeps_evicted_payload_alive_through_swap() -> None:
    codec = BlockingSampleCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=1,
        capacity_bytes=10_000,
        store_generation="reader-lease",
    )
    old = make_episode(codec, 0, [("old", 0)])
    new = make_episode(codec, 1, [("new", 0)])
    store.commit_episode(old)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    codec.block_episode_id = old.episode_id

    with ThreadPoolExecutor(max_workers=1) as executor:
        in_flight = executor.submit(replay.sample, 1, rng=random.Random(3))
        assert codec.sample_started.wait(timeout=2)

        store.commit_episode(new)
        assert replay.cursor is not None
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))
        replay.wait_for_idle(timeout=2)
        codec.block_episode_id = None
        assert replay.sample(1, rng=random.Random(3)) == [("new", 0)]

        codec.release_sample.set()
        assert in_flight.result(timeout=2) == [("old", 0)]

    replay.close()


def test_rebuild_failure_preserves_active_view_and_snapshot_can_recover() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=2,
        capacity_bytes=10_000,
        store_generation="failed-build",
    )
    first = make_episode(codec, 0, [("first", 0)])
    second = make_episode(codec, 1, [("second", 0)])
    store.commit_episode(first)
    replay = ControlledBuildFastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    replay.fail_build_once(2)

    store.commit_episode(second)
    assert replay.cursor is not None
    replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))

    with pytest.raises(IndexRebuildError, match="controlled rebuild failure"):
        replay.wait_for_idle(timeout=2)
    assert replay.active_cursor is not None
    assert replay.active_cursor.mutation_seq == 1
    assert replay.sample(1, rng=random.Random(4)) == [("first", 0)]
    failed_stats = replay.get_stats()
    assert failed_stats.rebuild_failures == 1
    assert failed_stats.delta_lag_mutations == 1

    replay.load_snapshot(store.get_snapshot())
    assert replay.active_cursor == store.cursor
    assert replay.get_stats().delta_lag_mutations == 0
    replay.close()


def test_metrics_report_delta_lag_and_successful_rebuild_time() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="rebuild-metrics",
    )
    store.commit_episode(make_episode(codec, 0, [0]))
    replay = ControlledBuildFastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    started, release = replay.block_build(2)

    added = make_episode(codec, 1, [1, 2, 3])
    store.commit_episode(added)
    assert replay.cursor is not None
    replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))
    assert started.wait(timeout=2)

    lagging = replay.get_stats()
    assert lagging.rebuild_in_progress
    assert lagging.delta_lag_mutations == 1
    assert lagging.delta_lag_agent_steps == 3

    release.set()
    replay.wait_for_idle(timeout=2)
    current = replay.get_stats()
    assert not current.rebuild_in_progress
    assert current.delta_lag_mutations == 0
    assert current.delta_lag_agent_steps == 0
    assert current.completed_rebuilds == 1
    assert current.last_rebuild_ms >= 0
    replay.close()


def test_delta_reuses_one_canonical_payload_map_and_episode_objects() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="single-payload-map",
    )
    retained = make_episode(codec, 0, [("retained", 0)])
    store.commit_episode(retained)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    canonical_records = replay._records

    store.commit_episode(make_episode(codec, 1, [("added", 0)]))
    assert replay.cursor is not None
    replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))
    replay.wait_for_idle(timeout=2)

    assert replay._records is canonical_records
    assert replay.get_snapshot().episodes[0] is retained
    replay.close()


def test_close_waits_for_rebuild_and_rejects_future_mutations() -> None:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=10,
        capacity_bytes=10_000,
        store_generation="rebuild-close",
    )
    store.commit_episode(make_episode(codec, 0, [0]))
    replay = ControlledBuildFastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    started, release = replay.block_build(2)

    store.commit_episode(make_episode(codec, 1, [1]))
    assert replay.cursor is not None
    delta = store.get_delta(replay.cursor, max_bytes=10_000)
    replay.apply_delta(delta)
    assert started.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(replay.close, wait=True, timeout=2)
        assert not closing.done()
        release.set()
        closing.result(timeout=2)

    assert replay.get_stats().closed
    with pytest.raises(ReplayClosedError):
        replay.apply_delta(delta)


@pytest.mark.parametrize("seed", range(5))
def test_concurrent_readers_survive_randomized_fifo_churn(seed: int) -> None:
    rng = random.Random(seed)
    codec = FlatEpisodeCodec()
    capacity = rng.randint(5, 15)
    store = EpisodeStore(
        codec,
        capacity_transitions=capacity,
        capacity_bytes=100_000,
        journal_capacity=200,
        store_generation=f"concurrent-churn-{seed}",
    )
    store.commit_episode(make_episode(codec, 0, [{"sequence": 0, "step": 0}]))
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    barrier = threading.Barrier(5)
    stop = threading.Event()

    def read_loop(reader: int) -> int:
        reader_rng = random.Random(seed * 1_000 + reader)
        samples = 0
        barrier.wait(timeout=2)
        while not stop.is_set():
            transition = replay.sample(1, rng=reader_rng)[0]
            assert isinstance(transition, dict)
            assert set(transition) == {"sequence", "step"}
            samples += 1
        return samples

    with ThreadPoolExecutor(max_workers=4) as executor:
        readers = [executor.submit(read_loop, reader) for reader in range(4)]
        barrier.wait(timeout=2)
        for sequence in range(1, 101):
            count = rng.randint(1, min(4, capacity))
            store.commit_episode(
                make_episode(
                    codec,
                    sequence,
                    [{"sequence": sequence, "step": step} for step in range(count)],
                )
            )
            assert replay.cursor is not None
            replay.apply_delta(store.get_delta(replay.cursor, max_bytes=100_000))
            if sequence % 5 == 0:
                time.sleep(0)
        replay.wait_for_idle(timeout=5)
        stop.set()
        assert all(future.result(timeout=2) > 0 for future in readers)

    assert replay.get_snapshot() == store.get_snapshot()
    retained_ids = set(replay.episode_ids)
    assert {
        episode_id
        for episode_id, _ in replay.sample_coordinates(
            1_000,
            rng=random.Random(seed),
        )
    } <= retained_ids
    replay.close()
