"""Bounded learner-local batch construction pipeline."""

from __future__ import annotations

import queue
import random
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeAlias, TypeVar, cast

import numpy as np

from rllib_async.protocols.batches import BatchCollator
from rllib_async.replay.fast import (
    FastReplay,
    IndexRebuildError,
    ReplayClosedError,
)
from rllib_async.replay.reference import ReplayError

BatchT = TypeVar("BatchT")
FlatBatch: TypeAlias = dict[str, np.ndarray]
_WAIT_SLICE_S = 0.01
_NO_BATCH = object()


class BatchCollationError(ValueError):
    """Sampled transitions cannot form one unambiguous flat batch."""


class BatchProducerError(RuntimeError):
    """The batch producer cannot continue or satisfy a lifecycle request."""


class BatchQueueEmptyError(TimeoutError):
    """No batch became available within the requested wait."""


class BatchProducerState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BatchProducerStats:
    state: BatchProducerState
    queue_size: int
    queue_capacity: int
    queue_high_watermark: int
    batches_produced: int
    batches_consumed: int
    batches_dropped: int
    producer_failures: int
    queue_full_events: int
    backpressure_s: float
    data_wait_calls: int
    data_wait_timeouts: int
    data_wait_s: float
    last_data_wait_ms: float


class FlatBatchCollator:
    """Stack flat numeric mapping transitions into contiguous NumPy columns."""

    def collate(self, transitions: Sequence[object]) -> FlatBatch:
        if not transitions:
            raise BatchCollationError("collation requires at least one transition")
        first = transitions[0]
        if not isinstance(first, Mapping):
            raise BatchCollationError("every flat transition must be a mapping")
        keys = tuple(first)
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise BatchCollationError("flat transition keys must be non-empty strings")

        mappings: list[Mapping[str, object]] = []
        expected_keys = set(keys)
        for transition in transitions:
            if not isinstance(transition, Mapping):
                raise BatchCollationError("every flat transition must be a mapping")
            if set(transition) != expected_keys:
                raise BatchCollationError(
                    "all flat transitions must have identical keys"
                )
            mappings.append(transition)

        batch: FlatBatch = {}
        for key in keys:
            try:
                arrays = [np.asarray(transition[key]) for transition in mappings]
            except (TypeError, ValueError) as error:
                raise BatchCollationError(
                    f"column {key!r} must contain numeric or boolean values"
                ) from error
            if any(array.dtype.kind not in "biufc" for array in arrays):
                raise BatchCollationError(
                    f"column {key!r} must contain numeric or boolean values"
                )
            shapes = {array.shape for array in arrays}
            if len(shapes) != 1:
                raise BatchCollationError(f"column {key!r} must have compatible shapes")
            try:
                stacked = np.stack(arrays, axis=0)
            except (TypeError, ValueError) as error:
                raise BatchCollationError(
                    f"column {key!r} must have compatible shapes"
                ) from error
            if stacked.dtype.kind not in "biufc":
                raise BatchCollationError(
                    f"column {key!r} must contain numeric or boolean values"
                )
            batch[key] = np.ascontiguousarray(stacked)
        return batch


class BatchProducer(Generic[BatchT]):
    """Build batches on one background thread into a bounded FIFO queue."""

    def __init__(
        self,
        replay: FastReplay,
        collator: BatchCollator[BatchT],
        *,
        batch_size: int,
        queue_capacity: int,
        seed: int,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            not isinstance(queue_capacity, int)
            or isinstance(queue_capacity, bool)
            or queue_capacity < 1
        ):
            raise ValueError("queue_capacity must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")

        self._replay = replay
        self._collator = collator
        self._batch_size = batch_size
        self._queue: queue.Queue[BatchT] = queue.Queue(maxsize=queue_capacity)
        self._rng = random.Random(seed)
        self._condition = threading.Condition()
        self._state = BatchProducerState.CREATED
        self._pause_acknowledged = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._queue_high_watermark = 0
        self._batches_produced = 0
        self._batches_consumed = 0
        self._batches_dropped = 0
        self._producer_failures = 0
        self._queue_full_events = 0
        self._backpressure_s = 0.0
        self._active_backpressure_started: float | None = None
        self._data_wait_calls = 0
        self._data_wait_timeouts = 0
        self._data_wait_s = 0.0
        self._last_data_wait_ms = 0.0

    def start(self) -> None:
        """Start a new producer or resume a paused one."""
        with self._condition:
            if self._state is BatchProducerState.RUNNING:
                return
            if self._state is BatchProducerState.PAUSED:
                self._state = BatchProducerState.RUNNING
                self._pause_acknowledged = False
                self._condition.notify_all()
                return
            if self._state is not BatchProducerState.CREATED:
                raise BatchProducerError(
                    f"cannot start batch producer in state {self._state.value!r}"
                )

            self._state = BatchProducerState.RUNNING
            thread = threading.Thread(
                target=self._run,
                name=f"batch-producer-{id(self):x}",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except Exception as error:
                self._state = BatchProducerState.FAILED
                self._thread = None
                self._error = error
                self._producer_failures += 1
                self._condition.notify_all()
                raise BatchProducerError(
                    f"could not start batch producer: {error}"
                ) from error

    def pause(self, *, timeout: float | None = None) -> None:
        """Pause after the current sample/collation step reaches a safe point."""
        deadline = self._deadline(timeout)
        with self._condition:
            if self._state is BatchProducerState.PAUSED:
                return
            if self._state is not BatchProducerState.RUNNING:
                raise BatchProducerError(
                    f"cannot pause batch producer in state {self._state.value!r}"
                )
            self._state = BatchProducerState.PAUSED
            self._condition.notify_all()
            while not self._pause_acknowledged:
                if self._state is BatchProducerState.FAILED:
                    self._raise_failure_locked()
                remaining = self._remaining(deadline)
                if remaining == 0:
                    raise TimeoutError("timed out waiting for batch producer pause")
                self._condition.wait(remaining)

    def resume(self) -> None:
        self.start()

    def get(self, *, timeout: float | None = None) -> BatchT:
        """Return one batch and account for consumer-side data wait."""
        deadline = self._deadline(timeout)
        started = time.monotonic()
        with self._condition:
            if self._state is BatchProducerState.CREATED:
                self._record_data_wait_locked(started, timed_out=False)
                raise BatchProducerError("start the batch producer before reading")
        while True:
            remaining = self._remaining(deadline)
            wait_slice = (
                _WAIT_SLICE_S if remaining is None else min(_WAIT_SLICE_S, remaining)
            )
            try:
                batch = self._queue.get(timeout=wait_slice)
            except queue.Empty:
                with self._condition:
                    if self._queue.empty():
                        if self._state is BatchProducerState.FAILED:
                            self._record_data_wait_locked(started, timed_out=False)
                            self._raise_failure_locked()
                        if self._state is BatchProducerState.STOPPED:
                            self._record_data_wait_locked(started, timed_out=False)
                            raise BatchProducerError(
                                "batch producer stopped with an empty queue"
                            ) from None
                if self._remaining(deadline) == 0:
                    with self._condition:
                        self._record_data_wait_locked(started, timed_out=True)
                    raise BatchQueueEmptyError(
                        "timed out waiting for a learner batch"
                    ) from None
                continue

            with self._condition:
                self._batches_consumed += 1
                self._record_data_wait_locked(started, timed_out=False)
            return batch

    def drain(self) -> list[BatchT]:
        """Remove and return every batch currently queued."""
        drained: list[BatchT] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        with self._condition:
            self._batches_dropped += len(drained)
        return drained

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the producer thread; queued batches remain available to drain."""
        deadline = self._deadline(timeout)
        with self._condition:
            if self._state is BatchProducerState.CREATED:
                self._state = BatchProducerState.STOPPED
                self._condition.notify_all()
                return
            if self._state in {
                BatchProducerState.STOPPED,
                BatchProducerState.FAILED,
            }:
                thread = self._thread
            else:
                self._state = BatchProducerState.STOPPING
                self._condition.notify_all()
                thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(self._remaining(deadline))
            if thread.is_alive():
                raise TimeoutError("timed out waiting for batch producer shutdown")

        with self._condition:
            if self._state is BatchProducerState.STOPPING:
                self._state = BatchProducerState.STOPPED
                self._condition.notify_all()

    def get_stats(self) -> BatchProducerStats:
        with self._condition:
            backpressure_s = self._backpressure_s
            if self._active_backpressure_started is not None:
                backpressure_s += time.monotonic() - self._active_backpressure_started
            return BatchProducerStats(
                state=self._state,
                queue_size=self._queue.qsize(),
                queue_capacity=self._queue.maxsize,
                queue_high_watermark=self._queue_high_watermark,
                batches_produced=self._batches_produced,
                batches_consumed=self._batches_consumed,
                batches_dropped=self._batches_dropped,
                producer_failures=self._producer_failures,
                queue_full_events=self._queue_full_events,
                backpressure_s=backpressure_s,
                data_wait_calls=self._data_wait_calls,
                data_wait_timeouts=self._data_wait_timeouts,
                data_wait_s=self._data_wait_s,
                last_data_wait_ms=self._last_data_wait_ms,
            )

    def get_rng_state(self) -> object:
        """Return sampler RNG state only at a producer safe point."""

        with self._condition:
            if self._state not in {
                BatchProducerState.CREATED,
                BatchProducerState.PAUSED,
            }:
                raise BatchProducerError(
                    "sampler RNG state requires a created or paused producer"
                )
            return self._rng.getstate()

    def set_rng_state(self, state: object) -> None:
        """Restore sampler RNG state before the producer thread starts."""

        with self._condition:
            if self._state is not BatchProducerState.CREATED:
                raise BatchProducerError(
                    "sampler RNG state can only be restored before start"
                )
            probe = random.Random()
            try:
                probe.setstate(state)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid sampler RNG state") from error
            self._rng.setstate(state)

    def __enter__(self) -> BatchProducer[BatchT]:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _run(self) -> None:
        pending: BatchT | object = _NO_BATCH
        try:
            while self._wait_until_running():
                if pending is _NO_BATCH:
                    if not self._replay.wait_for_transitions(timeout=_WAIT_SLICE_S):
                        if self._replay.get_stats().closed:
                            raise ReplayClosedError(
                                "learner-local replay closed while producing batches"
                            )
                        continue
                    if not self._wait_until_running():
                        break
                    try:
                        transitions = self._replay.sample(
                            self._batch_size,
                            rng=self._rng,
                        )
                    except (IndexRebuildError, ReplayClosedError):
                        raise
                    except ReplayError:
                        continue
                    pending = self._collator.collate(transitions)

                if not self._put_when_running(cast(BatchT, pending)):
                    break
                pending = _NO_BATCH
        except Exception as error:
            with self._condition:
                self._error = error
                self._producer_failures += 1
                self._state = BatchProducerState.FAILED
                self._condition.notify_all()
            return

        with self._condition:
            if self._state is BatchProducerState.STOPPING:
                self._state = BatchProducerState.STOPPED
            self._condition.notify_all()

    def _wait_until_running(self) -> bool:
        with self._condition:
            while self._state is BatchProducerState.PAUSED:
                self._pause_acknowledged = True
                self._condition.notify_all()
                self._condition.wait()
            self._pause_acknowledged = False
            return self._state is BatchProducerState.RUNNING

    def _put_when_running(self, batch: BatchT) -> bool:
        blocked_started: float | None = None
        while True:
            with self._condition:
                if (
                    self._state is BatchProducerState.PAUSED
                    and blocked_started is not None
                ):
                    self._finish_backpressure_locked(blocked_started)
                    blocked_started = None
            if not self._wait_until_running():
                break
            if self._queue.full() and blocked_started is None:
                blocked_started = time.monotonic()
                with self._condition:
                    self._queue_full_events += 1
                    self._active_backpressure_started = blocked_started
            try:
                self._queue.put(batch, timeout=_WAIT_SLICE_S)
            except queue.Full:
                continue

            with self._condition:
                if blocked_started is not None:
                    self._finish_backpressure_locked(blocked_started)
                self._batches_produced += 1
                self._queue_high_watermark = max(
                    self._queue_high_watermark,
                    self._queue.qsize(),
                )
            return True
        if blocked_started is not None:
            with self._condition:
                self._finish_backpressure_locked(blocked_started)
        return False

    def _finish_backpressure_locked(self, started: float) -> None:
        self._backpressure_s += time.monotonic() - started
        if self._active_backpressure_started == started:
            self._active_backpressure_started = None

    def _raise_failure_locked(self) -> None:
        assert self._error is not None
        raise BatchProducerError(
            f"batch producer failed: {self._error}"
        ) from self._error

    def _record_data_wait_locked(
        self,
        started: float,
        *,
        timed_out: bool,
    ) -> None:
        elapsed = time.monotonic() - started
        self._data_wait_calls += 1
        self._data_wait_s += elapsed
        self._last_data_wait_ms = elapsed * 1_000
        if timed_out:
            self._data_wait_timeouts += 1

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        return time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
