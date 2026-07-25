"""Finite-call learner host owning SAC and its local replay hot path."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Any

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.learner import SACLearnerAdapter
from rllib_async.protocols import (
    FlatEpisodeCodec,
    ReplayDelta,
    ReplaySnapshot,
    WeightsDescriptor,
)
from rllib_async.replay import (
    BatchProducer,
    BatchProducerStats,
    BatchQueueEmptyError,
    FastReplay,
    FastReplayStats,
    FlatBatch,
    FlatBatchCollator,
)


class LearnerHostError(RuntimeError):
    """The learner host cannot safely satisfy a runtime request."""


class LearnerHostState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LearnerHostTick:
    """Finite progress made by one learner-host event-pump call."""

    synced_transactions: int
    sync_has_more: bool
    updates_performed: int
    updates_skipped_learning_start: int
    published_weights: WeightsDescriptor | None
    learner_metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class LearnerHostStats:
    """Cumulative learner, sync, and local replay metrics."""

    state: LearnerHostState
    sync_requests: int
    snapshot_loads: int
    delta_transactions: int
    full_resyncs: int
    learner_updates: int
    updates_skipped_learning_start: int
    weight_publications: int
    updates_per_s: float
    latest_module_version: int
    last_learner_metrics: tuple[tuple[str, float], ...]
    fast_replay: FastReplayStats
    batch_producer: BatchProducerStats


class LearnerHost:
    """Own a local replay view, bounded batches, and one RLlib SAC learner."""

    def __init__(
        self,
        config: SACConfig,
        spaces: Mapping[str, tuple[gym.Space, gym.Space]],
        replay_actor: Any,
        codec: FlatEpisodeCodec,
        *,
        member_id: str,
        publication_interval_updates: int,
        batch_size: int,
        batch_queue_capacity: int,
        batch_seed: int,
        replay_sync_max_bytes: int,
    ) -> None:
        if not isinstance(config, SACConfig):
            raise TypeError("config must be an SACConfig")
        if not isinstance(codec, FlatEpisodeCodec):
            raise TypeError("LearnerHost requires FlatEpisodeCodec")
        if (
            not isinstance(replay_sync_max_bytes, int)
            or isinstance(replay_sync_max_bytes, bool)
            or replay_sync_max_bytes < 1
        ):
            raise ValueError("replay_sync_max_bytes must be positive")

        self._replay_actor = replay_actor
        self._codec = codec
        self._sync_max_bytes = replay_sync_max_bytes
        self._fast_replay = FastReplay(codec)
        snapshot = ray.get(self._replay_actor.get_snapshot.remote())
        if not isinstance(snapshot, ReplaySnapshot):
            self._fast_replay.close()
            raise LearnerHostError("replay actor returned an invalid snapshot")
        self._fast_replay.load_snapshot(snapshot)

        self._adapter: SACLearnerAdapter | None = None
        self._producer: BatchProducer[FlatBatch] | None = None
        try:
            self._adapter = SACLearnerAdapter(
                config,
                spaces=spaces,
                member_id=member_id,
                publication_interval_updates=publication_interval_updates,
            )
            self._producer = BatchProducer(
                self._fast_replay,
                FlatBatchCollator(),
                batch_size=batch_size,
                queue_capacity=batch_queue_capacity,
                seed=batch_seed,
            )
        except Exception:
            if self._adapter is not None:
                self._adapter.close()
            self._fast_replay.close()
            raise

        self._state = LearnerHostState.CREATED
        self._started_at: float | None = None
        self._sync_requests = 0
        self._snapshot_loads = 1
        self._delta_transactions = 0
        self._full_resyncs = 0
        self._updates_skipped_learning_start = 0
        self._weight_publications = 0
        self._last_learner_metrics: tuple[tuple[str, float], ...] = ()
        initial_weights = self._adapter.get_published_weights()
        self._latest_module_version = max(
            initial_weights.module_versions.values(),
            default=0,
        )

    def start(self) -> None:
        if self._state is LearnerHostState.RUNNING:
            return
        if self._state is LearnerHostState.PAUSED:
            self.resume()
            return
        if self._state is not LearnerHostState.CREATED:
            raise LearnerHostError(
                f"cannot start learner host in state {self._state.value!r}"
            )
        assert self._producer is not None
        self._producer.start()
        self._started_at = time.monotonic()
        self._state = LearnerHostState.RUNNING

    def tick(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        max_updates: int,
    ) -> LearnerHostTick:
        """Synchronize once, then consume several purely local batches."""

        self._require_state(LearnerHostState.RUNNING)
        if (
            not isinstance(max_updates, int)
            or isinstance(max_updates, bool)
            or max_updates < 1
        ):
            raise ValueError("max_updates must be a positive integer")
        try:
            transaction_count, has_more = self._sync_once()
            return self._consume_available_batches(
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
                max_updates=max_updates,
                synced_transactions=transaction_count,
                sync_has_more=has_more,
            )
        except Exception:
            self._state = LearnerHostState.FAILED
            raise

    def pause(self, *, timeout_s: float | None = None) -> None:
        if self._state is LearnerHostState.PAUSED:
            return
        self._require_state(LearnerHostState.RUNNING)
        assert self._producer is not None
        self._producer.pause(timeout=timeout_s)
        self._state = LearnerHostState.PAUSED

    def resume(self) -> None:
        if self._state is LearnerHostState.RUNNING:
            return
        self._require_state(LearnerHostState.PAUSED)
        assert self._producer is not None
        self._producer.resume()
        self._state = LearnerHostState.RUNNING

    def drain(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        timeout_s: float | None = None,
    ) -> LearnerHostTick:
        """Reach current replay, pause production, and consume queued batches."""

        if self._state not in {
            LearnerHostState.RUNNING,
            LearnerHostState.PAUSED,
        }:
            raise LearnerHostError(
                f"cannot drain learner host in state {self._state.value!r}"
            )
        deadline = self._deadline(timeout_s)
        synced_transactions = 0
        has_more = True
        try:
            while has_more:
                transaction_count, has_more = self._sync_once()
                synced_transactions += transaction_count
                if has_more and self._remaining(deadline) == 0:
                    raise TimeoutError("timed out draining learner replay sync")
            self._fast_replay.wait_for_idle(timeout=self._remaining(deadline))

            assert self._producer is not None
            if self._state is LearnerHostState.RUNNING:
                self._producer.pause(timeout=self._remaining(deadline))
                self._state = LearnerHostState.PAUSED
            queued = self._drain_queued_batches()
            tick = self._consume_batches(
                queued,
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
                synced_transactions=synced_transactions,
                sync_has_more=False,
            )
            assert self._adapter is not None
            forced = self._adapter.maybe_publish_weights(force=True)
            if forced is None:
                return tick
            self._weight_publications += 1
            self._latest_module_version = max(
                forced.module_versions.values(),
                default=self._latest_module_version,
            )
            return LearnerHostTick(
                synced_transactions=tick.synced_transactions,
                sync_has_more=False,
                updates_performed=tick.updates_performed,
                updates_skipped_learning_start=(tick.updates_skipped_learning_start),
                published_weights=forced,
                learner_metrics=tick.learner_metrics,
            )
        except Exception:
            self._state = LearnerHostState.FAILED
            raise

    def get_published_weights(self) -> WeightsDescriptor:
        self._require_not_stopped()
        assert self._adapter is not None
        return self._adapter.get_published_weights()

    def get_stats(self) -> LearnerHostStats:
        self._require_not_stopped()
        assert self._adapter is not None
        assert self._producer is not None
        elapsed = (
            max(time.monotonic() - self._started_at, 0.0)
            if self._started_at is not None
            else 0.0
        )
        return LearnerHostStats(
            state=self._state,
            sync_requests=self._sync_requests,
            snapshot_loads=self._snapshot_loads,
            delta_transactions=self._delta_transactions,
            full_resyncs=self._full_resyncs,
            learner_updates=self._adapter.learner_updates,
            updates_skipped_learning_start=(self._updates_skipped_learning_start),
            weight_publications=self._weight_publications,
            updates_per_s=(self._adapter.learner_updates / elapsed if elapsed else 0.0),
            latest_module_version=self._latest_module_version,
            last_learner_metrics=self._last_learner_metrics,
            fast_replay=self._fast_replay.get_stats(),
            batch_producer=self._producer.get_stats(),
        )

    def stop(self, *, timeout_s: float | None = None) -> None:
        if self._state is LearnerHostState.STOPPED:
            return
        errors: list[BaseException] = []
        if self._producer is not None:
            try:
                self._producer.stop(timeout=timeout_s)
            except BaseException as error:
                errors.append(error)
        try:
            self._fast_replay.close(timeout=timeout_s)
        except BaseException as error:
            errors.append(error)
        if self._adapter is not None:
            try:
                self._adapter.close()
            except BaseException as error:
                errors.append(error)
        self._state = LearnerHostState.STOPPED
        if errors:
            raise LearnerHostError(
                f"learner host shutdown failed: {errors[0]}"
            ) from errors[0]

    def _sync_once(self) -> tuple[int, bool]:
        cursor = self._fast_replay.cursor
        if cursor is None:
            raise LearnerHostError("local replay has no synchronization cursor")
        self._sync_requests += 1
        delta = ray.get(
            self._replay_actor.get_delta.remote(
                cursor,
                max_bytes=self._sync_max_bytes,
            )
        )
        if not isinstance(delta, ReplayDelta):
            raise LearnerHostError("replay actor returned an invalid delta")
        if delta.full_resync_required:
            snapshot = ray.get(self._replay_actor.get_snapshot.remote())
            if not isinstance(snapshot, ReplaySnapshot):
                raise LearnerHostError("replay actor returned an invalid snapshot")
            self._fast_replay.load_snapshot(snapshot)
            self._snapshot_loads += 1
            self._full_resyncs += 1
            return 0, False
        self._fast_replay.apply_delta(delta)
        transaction_count = len(delta.transactions)
        self._delta_transactions += transaction_count
        return transaction_count, delta.has_more

    def _consume_available_batches(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        max_updates: int,
        synced_transactions: int,
        sync_has_more: bool,
    ) -> LearnerHostTick:
        assert self._producer is not None
        batches: list[FlatBatch] = []
        for _ in range(max_updates):
            try:
                batches.append(self._producer.get(timeout=0.0))
            except BatchQueueEmptyError:
                break
        return self._consume_batches(
            batches,
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
            synced_transactions=synced_transactions,
            sync_has_more=sync_has_more,
        )

    def _consume_batches(
        self,
        batches: list[FlatBatch],
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        synced_transactions: int,
        sync_has_more: bool,
    ) -> LearnerHostTick:
        assert self._adapter is not None
        updates_performed = 0
        skipped = 0
        publication: WeightsDescriptor | None = None
        last_metrics = self._last_learner_metrics
        for batch in batches:
            update = self._adapter.update(
                batch,
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
            )
            if update.performed:
                updates_performed += 1
                last_metrics = self._extract_numeric_metrics(update.learner_results)
            else:
                skipped += 1
            if update.published_weights is not None:
                publication = update.published_weights
                self._weight_publications += 1
                self._latest_module_version = max(
                    publication.module_versions.values(),
                    default=self._latest_module_version,
                )
        self._updates_skipped_learning_start += skipped
        self._last_learner_metrics = last_metrics
        return LearnerHostTick(
            synced_transactions=synced_transactions,
            sync_has_more=sync_has_more,
            updates_performed=updates_performed,
            updates_skipped_learning_start=skipped,
            published_weights=publication,
            learner_metrics=last_metrics,
        )

    def _drain_queued_batches(self) -> list[FlatBatch]:
        assert self._producer is not None
        batches: list[FlatBatch] = []
        while True:
            try:
                batches.append(self._producer.get(timeout=0.0))
            except BatchQueueEmptyError:
                return batches

    @classmethod
    def _extract_numeric_metrics(
        cls,
        value: object,
    ) -> tuple[tuple[str, float], ...]:
        metrics: dict[str, float] = {}

        def visit(item: object, path: tuple[str, ...]) -> None:
            if hasattr(item, "peek") and callable(item.peek):
                try:
                    visit(item.peek(), path)
                except Exception:
                    return
            elif isinstance(item, Mapping):
                for key, nested in item.items():
                    visit(nested, (*path, str(key)))
            elif isinstance(item, list | tuple):
                for index, nested in enumerate(item):
                    visit(nested, (*path, str(index)))
            elif isinstance(item, np.ndarray):
                if item.size == 1:
                    visit(item.reshape(()).item(), path)
            elif isinstance(item, Number) and not isinstance(item, bool):
                numeric = float(item)
                if math.isfinite(numeric):
                    metrics["/".join(path) or "value"] = numeric

        visit(value, ())
        return tuple(sorted(metrics.items()))

    def _require_state(self, expected: LearnerHostState) -> None:
        if self._state is not expected:
            raise LearnerHostError(
                f"learner host must be {expected.value!r}, not {self._state.value!r}"
            )

    def _require_not_stopped(self) -> None:
        if self._state is LearnerHostState.STOPPED:
            raise LearnerHostError("learner host is stopped")

    @staticmethod
    def _deadline(timeout_s: float | None) -> float | None:
        if timeout_s is None:
            return None
        if (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and non-negative or None")
        return time.monotonic() + timeout_s

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(deadline - time.monotonic(), 0.0)


@ray.remote(max_concurrency=1)
class LearnerHostActor:
    """Ray process boundary around the finite-call learner host."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._host = LearnerHost(*args, **kwargs)

    def start(self) -> None:
        self._host.start()

    def tick(self, **kwargs: Any) -> LearnerHostTick:
        return self._host.tick(**kwargs)

    def pause(self, **kwargs: Any) -> None:
        self._host.pause(**kwargs)

    def resume(self) -> None:
        self._host.resume()

    def drain(self, **kwargs: Any) -> LearnerHostTick:
        return self._host.drain(**kwargs)

    def get_published_weights(self) -> WeightsDescriptor:
        return self._host.get_published_weights()

    def get_stats(self) -> LearnerHostStats:
        return self._host.get_stats()

    def stop(self, **kwargs: Any) -> None:
        self._host.stop(**kwargs)
