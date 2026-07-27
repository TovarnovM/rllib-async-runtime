from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.replay import FastReplay
from rllib_async.replay.batching import (
    BatchCollationError,
    BatchProducer,
    BatchProducerError,
    BatchProducerState,
    BatchQueueEmptyError,
    FlatBatchCollator,
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


def make_replay(transitions: list[object]) -> tuple[EpisodeStore, FastReplay]:
    codec = FlatEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=100,
        capacity_bytes=100_000,
        store_generation="batching",
    )
    if transitions:
        store.commit_episode(make_episode(codec, 0, transitions))
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    return store, replay


def wait_until(predicate: Callable[[], bool], *, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        time.sleep(0.005)


def test_flat_batch_collator_stacks_numeric_mapping_columns() -> None:
    transitions = [
        {
            "obs": np.array([1.0, 2.0], dtype=np.float32),
            "action": 0,
            "reward": 1.5,
            "terminated": False,
        },
        {
            "obs": np.array([3.0, 4.0], dtype=np.float32),
            "action": 1,
            "reward": -0.5,
            "terminated": True,
        },
    ]

    batch = FlatBatchCollator().collate(transitions)

    assert tuple(batch) == ("obs", "action", "reward", "terminated")
    np.testing.assert_array_equal(
        batch["obs"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(batch["action"], np.array([0, 1]))
    np.testing.assert_array_equal(batch["reward"], np.array([1.5, -0.5]))
    np.testing.assert_array_equal(batch["terminated"], np.array([False, True]))
    assert all(array.flags.c_contiguous for array in batch.values())


@pytest.mark.parametrize(
    ("transitions", "message"),
    [
        ([], "at least one transition"),
        ([1], "must be a mapping"),
        ([{"obs": 1}, {"action": 1}], "identical keys"),
        ([{"obs": [1]}, {"obs": [1, 2]}], "compatible shapes"),
        ([{"metadata": {"runner": "a"}}], "numeric or boolean"),
    ],
)
def test_flat_batch_collator_rejects_ambiguous_inputs(
    transitions: Sequence[object],
    message: str,
) -> None:
    with pytest.raises(BatchCollationError, match=message):
        FlatBatchCollator().collate(transitions)


def test_bounded_queue_applies_backpressure_without_exceeding_capacity() -> None:
    _, replay = make_replay(
        [
            {"value": index, "vector": np.array([index], dtype=np.float32)}
            for index in range(8)
        ]
    )
    producer = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=4,
        queue_capacity=2,
        seed=17,
    )
    try:
        producer.start()
        wait_until(
            lambda: producer.get_stats().queue_full_events > 0,
        )
        stats = producer.get_stats()
        assert stats.queue_size == 2
        assert stats.queue_high_watermark == 2
        assert stats.queue_high_watermark <= stats.queue_capacity
        assert stats.backpressure_s > 0

        batch = producer.get(timeout=1)
        assert batch["value"].shape == (4,)
        assert batch["vector"].shape == (4, 1)
        assert producer.get_stats().batches_consumed == 1
    finally:
        producer.stop(timeout=2)
        replay.close()


def test_zero_capacity_builds_batches_synchronously_without_a_queue() -> None:
    _, replay = make_replay(
        [
            {"value": index, "vector": np.array([index], dtype=np.float32)}
            for index in range(8)
        ]
    )
    producer = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=4,
        queue_capacity=0,
        seed=19,
    )
    try:
        producer.start()
        batch = producer.get(timeout=1)
        stats = producer.get_stats()

        assert batch["value"].shape == (4,)
        assert not stats.prefetch_enabled
        assert stats.queue_size == 0
        assert stats.queue_capacity == 0
        assert stats.queue_high_watermark == 0
        assert stats.batches_produced == 1
        assert stats.batches_consumed == 1
        assert stats.batch_builds == 1
        assert stats.batch_build_s > 0
        assert stats.data_wait_calls == 1

        producer.pause(timeout=1)
        assert producer.get_stats().state is BatchProducerState.PAUSED
        assert producer.drain() == []
        producer.resume()
        assert producer.get_stats().state is BatchProducerState.RUNNING
    finally:
        producer.stop(timeout=2)
        replay.close()


def test_zero_capacity_sampler_rng_round_trips_at_a_pause_boundary() -> None:
    _, replay = make_replay([{"value": index} for index in range(32)])
    source = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=8,
        queue_capacity=0,
        seed=21,
    )
    restored = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=8,
        queue_capacity=0,
        seed=999,
    )
    try:
        source.start()
        source.get(timeout=1)
        source.pause(timeout=1)
        restored.set_rng_state(source.get_rng_state())

        source.resume()
        restored.start()
        expected = source.get(timeout=1)
        actual = restored.get(timeout=1)

        np.testing.assert_array_equal(actual["value"], expected["value"])
    finally:
        source.stop(timeout=2)
        restored.stop(timeout=2)
        replay.close()


def test_empty_queue_records_data_wait_then_recovers_when_replay_fills() -> None:
    store, replay = make_replay([])
    producer = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=1,
        queue_capacity=1,
        seed=23,
    )
    try:
        producer.start()
        with pytest.raises(BatchQueueEmptyError):
            producer.get(timeout=0.05)
        waiting = producer.get_stats()
        assert waiting.data_wait_calls == 1
        assert waiting.data_wait_timeouts == 1
        assert waiting.data_wait_s > 0

        codec = FlatEpisodeCodec()
        store.commit_episode(make_episode(codec, 0, [{"value": 9}]))
        assert replay.cursor is not None
        replay.apply_delta(store.get_delta(replay.cursor, max_bytes=10_000))
        replay.wait_for_idle(timeout=2)

        batch = producer.get(timeout=2)
        np.testing.assert_array_equal(batch["value"], np.array([9]))
        recovered = producer.get_stats()
        assert recovered.batches_consumed == 1
        assert recovered.data_wait_calls == 2
    finally:
        producer.stop(timeout=2)
        replay.close()


def test_pause_drain_resume_and_stop_have_explicit_lifecycle() -> None:
    _, replay = make_replay([{"value": index} for index in range(4)])
    producer = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=2,
        queue_capacity=2,
        seed=29,
    )
    try:
        producer.start()
        wait_until(lambda: producer.get_stats().queue_size == 2)

        producer.pause(timeout=2)
        assert producer.get_stats().state is BatchProducerState.PAUSED
        drained = producer.drain()
        assert len(drained) == 2
        assert producer.get_stats().queue_size == 0
        assert producer.get_stats().batches_dropped == 2

        producer.resume()
        batch = producer.get(timeout=2)
        assert isinstance(batch, Mapping)
        producer.stop(timeout=2)
        assert producer.get_stats().state is BatchProducerState.STOPPED
        with pytest.raises(BatchProducerError, match="cannot start"):
            producer.start()
    finally:
        producer.stop(timeout=2)
        replay.close()


class FailingCollator:
    def collate(self, transitions: Sequence[object]) -> object:
        raise ValueError("controlled collator failure")


@pytest.mark.parametrize("queue_capacity", [0, 1])
def test_producer_failure_is_visible_to_consumer_and_metrics(
    queue_capacity: int,
) -> None:
    _, replay = make_replay([{"value": 1}])
    producer = BatchProducer(
        replay,
        FailingCollator(),
        batch_size=1,
        queue_capacity=queue_capacity,
        seed=31,
    )
    try:
        producer.start()
        with pytest.raises(BatchProducerError, match="controlled collator failure"):
            producer.get(timeout=2)
        stats = producer.get_stats()
        assert stats.state is BatchProducerState.FAILED
        assert stats.producer_failures == 1
        assert stats.queue_size == 0
    finally:
        producer.stop(timeout=2)
        assert producer.get_stats().state is BatchProducerState.FAILED
        replay.close()


def test_get_before_start_fails_instead_of_waiting_forever() -> None:
    _, replay = make_replay([{"value": 1}])
    producer = BatchProducer(
        replay,
        FlatBatchCollator(),
        batch_size=1,
        queue_capacity=1,
        seed=37,
    )
    try:
        with pytest.raises(BatchProducerError, match="start"):
            producer.get()
    finally:
        producer.stop(timeout=2)
        replay.close()
