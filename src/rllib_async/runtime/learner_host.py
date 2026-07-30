"""Finite-call learner host owning SAC and its local replay hot path."""

from __future__ import annotations

import math
import pickle
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Number
from typing import Any

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.learner import PBTModelState, SACLearnerAdapter
from rllib_async.protocols import (
    FlatEpisodeCodec,
    ReplayCursor,
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

LEARNER_HOST_CHECKPOINT_VERSION = 1


def encode_learner_host_checkpoint(state: Mapping[str, Any]) -> bytes:
    """Serialize learner-owned tensors inside the learner process."""

    if not isinstance(state, Mapping):
        raise TypeError("learner host checkpoint state must be a mapping")
    return pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)


def decode_learner_host_checkpoint(payload: bytes) -> dict[str, Any]:
    """Deserialize trusted learner state on the learner's assigned device."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("learner host checkpoint payload must be non-empty bytes")
    try:
        state = pickle.loads(payload)
    except Exception as error:
        raise ValueError("learner host checkpoint payload is unreadable") from error
    if not isinstance(state, Mapping):
        raise ValueError("learner host checkpoint payload is not a mapping")
    return dict(state)


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
class LearnerHostCheckpoint:
    """Opaque learner-owned state plus its authoritative replay cursor."""

    replay_cursor: ReplayCursor
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.replay_cursor, ReplayCursor):
            raise TypeError("learner checkpoint replay_cursor is invalid")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("learner checkpoint payload must be non-empty bytes")


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
    samples_per_s: float
    data_wait_fraction: float
    batch_queue_empty_fraction: float
    update_time_ms_p50: float
    update_time_ms_p95: float
    latest_module_version: int
    last_learner_metrics: tuple[tuple[str, float], ...]
    fast_replay: FastReplayStats
    batch_producer: BatchProducerStats
    node_id: str = ""
    accelerator_ids: tuple[str, ...] = ()


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
        checkpoint_state: Mapping[str, Any] | bytes | None = None,
        allow_replay_ahead_on_restore: bool = False,
        pbt_state: PBTModelState | None = None,
        learning_starts_satisfied: bool = False,
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
        if config.n_step != 1:
            raise ValueError(
                "LearnerHost requires n_step=1 until n-step targets are constructed"
            )
        if not isinstance(allow_replay_ahead_on_restore, bool):
            raise TypeError("allow_replay_ahead_on_restore must be a bool")
        if checkpoint_state is not None and pbt_state is not None:
            raise ValueError("learner host accepts only one restore source")

        self._replay_actor = replay_actor
        self._codec = codec
        self._member_id = member_id
        self._sync_max_bytes = replay_sync_max_bytes
        self._batch_size = batch_size
        self._allow_replay_ahead_on_restore = allow_replay_ahead_on_restore
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
                learning_starts_satisfied=learning_starts_satisfied,
            )
            if pbt_state is not None:
                self._adapter.load_pbt_state(pbt_state)
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
        self._rate_base_learner_updates = 0
        self._sync_requests = 0
        self._snapshot_loads = 1
        self._delta_transactions = 0
        self._full_resyncs = 0
        self._updates_skipped_learning_start = 0
        self._weight_publications = int(pbt_state is not None)
        self._last_learner_metrics: tuple[tuple[str, float], ...] = ()
        self._update_times_ms: deque[float] = deque(maxlen=1_024)
        initial_weights = self._adapter.get_published_weights()
        self._latest_module_version = max(
            initial_weights.module_versions.values(),
            default=0,
        )
        if checkpoint_state is not None:
            try:
                if isinstance(checkpoint_state, bytes):
                    checkpoint_state = decode_learner_host_checkpoint(checkpoint_state)
                self._restore_checkpoint_state(checkpoint_state)
            except Exception:
                assert self._producer is not None
                assert self._adapter is not None
                self._producer.stop()
                self._adapter.close()
                self._fast_replay.close()
                raise

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
        max_updates: int | None = None,
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
        if max_updates is not None and (
            not isinstance(max_updates, int)
            or isinstance(max_updates, bool)
            or max_updates < 0
        ):
            raise ValueError("max_updates must be non-negative or None")
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
            queued = self._drain_queued_batches(max_batches=max_updates)
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

    def export_pbt_state(self) -> PBTModelState:
        self._require_not_stopped()
        assert self._adapter is not None
        return self._adapter.export_pbt_state()

    def get_stats(self) -> LearnerHostStats:
        self._require_not_stopped()
        assert self._adapter is not None
        assert self._producer is not None
        elapsed = (
            max(time.monotonic() - self._started_at, 0.0)
            if self._started_at is not None
            else 0.0
        )
        producer = self._producer.get_stats()
        return LearnerHostStats(
            state=self._state,
            sync_requests=self._sync_requests,
            snapshot_loads=self._snapshot_loads,
            delta_transactions=self._delta_transactions,
            full_resyncs=self._full_resyncs,
            learner_updates=self._adapter.learner_updates,
            updates_skipped_learning_start=(self._updates_skipped_learning_start),
            weight_publications=self._weight_publications,
            updates_per_s=(
                (self._adapter.learner_updates - self._rate_base_learner_updates)
                / elapsed
                if elapsed
                else 0.0
            ),
            samples_per_s=(
                producer.batches_consumed * self._batch_size / elapsed
                if elapsed
                else 0.0
            ),
            data_wait_fraction=(
                min(producer.data_wait_s / elapsed, 1.0) if elapsed else 0.0
            ),
            batch_queue_empty_fraction=(
                producer.data_wait_timeouts / producer.data_wait_calls
                if producer.data_wait_calls
                else 0.0
            ),
            update_time_ms_p50=self._percentile(self._update_times_ms, 50),
            update_time_ms_p95=self._percentile(self._update_times_ms, 95),
            latest_module_version=self._latest_module_version,
            last_learner_metrics=self._last_learner_metrics,
            fast_replay=self._fast_replay.get_stats(),
            batch_producer=producer,
        )

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return learner state without serializing the derived replay view."""

        self._require_state(LearnerHostState.PAUSED)
        assert self._adapter is not None
        assert self._producer is not None
        replay = self._fast_replay.get_stats()
        if (
            replay.cursor is None
            or replay.active_cursor != replay.cursor
            or replay.rebuild_in_progress
        ):
            raise LearnerHostError(
                "learner-local replay must be fully materialized before checkpoint"
            )
        return {
            "state_version": LEARNER_HOST_CHECKPOINT_VERSION,
            "member_id": self._member_id,
            "replay_cursor": replay.cursor,
            "adapter": self._adapter.get_state(),
            "sampler_rng_state": self._producer.get_rng_state(),
            "sync_requests": self._sync_requests,
            "snapshot_loads": self._snapshot_loads,
            "delta_transactions": self._delta_transactions,
            "full_resyncs": self._full_resyncs,
            "updates_skipped_learning_start": (self._updates_skipped_learning_start),
            "weight_publications": self._weight_publications,
            "last_learner_metrics": self._last_learner_metrics,
        }

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
            update_started = time.monotonic()
            update = self._adapter.update(
                batch,
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
            )
            if update.performed:
                self._update_times_ms.append(
                    (time.monotonic() - update_started) * 1_000
                )
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

    def _drain_queued_batches(
        self,
        *,
        max_batches: int | None = None,
    ) -> list[FlatBatch]:
        assert self._producer is not None
        if not self._producer.get_stats().prefetch_enabled:
            return []
        batches: list[FlatBatch] = []
        while max_batches is None or len(batches) < max_batches:
            try:
                batches.append(self._producer.get(timeout=0.0))
            except BatchQueueEmptyError:
                break
        self._producer.drain()
        return batches

    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("learner host checkpoint state must be a mapping")
        if state.get("state_version") != LEARNER_HOST_CHECKPOINT_VERSION:
            raise ValueError("unsupported learner host checkpoint state version")
        if state.get("member_id") != self._member_id:
            raise ValueError("learner host checkpoint member_id does not match")

        replay_cursor = state.get("replay_cursor")
        replay = self._fast_replay.get_stats()
        if not isinstance(replay_cursor, ReplayCursor) or replay.cursor is None:
            raise ValueError(
                "learner host checkpoint does not match authoritative replay"
            )
        if self._allow_replay_ahead_on_restore:
            replay_is_compatible = (
                replay.cursor.store_generation == replay_cursor.store_generation
                and replay.cursor.mutation_seq >= replay_cursor.mutation_seq
                and replay.active_cursor == replay.cursor
            )
        else:
            replay_is_compatible = (
                replay.cursor == replay_cursor and replay.active_cursor == replay_cursor
            )
        if not replay_is_compatible:
            raise ValueError(
                "learner host checkpoint does not match authoritative replay"
            )

        adapter_state = state.get("adapter")
        if not isinstance(adapter_state, Mapping):
            raise ValueError("learner host checkpoint adapter state is invalid")
        assert self._adapter is not None
        assert self._producer is not None
        self._adapter.set_state(adapter_state)
        self._rate_base_learner_updates = self._adapter.learner_updates
        self._producer.set_rng_state(state.get("sampler_rng_state"))

        self._sync_requests = self._checkpoint_counter(state, "sync_requests")
        self._snapshot_loads = self._checkpoint_counter(state, "snapshot_loads") + 1
        self._delta_transactions = self._checkpoint_counter(
            state,
            "delta_transactions",
        )
        self._full_resyncs = self._checkpoint_counter(state, "full_resyncs")
        self._updates_skipped_learning_start = self._checkpoint_counter(
            state,
            "updates_skipped_learning_start",
        )
        self._weight_publications = self._checkpoint_counter(
            state,
            "weight_publications",
        )

        metrics = state.get("last_learner_metrics")
        if not isinstance(metrics, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int | float)
            or isinstance(item[1], bool)
            or not math.isfinite(item[1])
            for item in metrics
        ):
            raise ValueError("learner host checkpoint metrics are invalid")
        self._last_learner_metrics = metrics

        weights = self._adapter.get_published_weights()
        self._latest_module_version = max(
            weights.module_versions.values(),
            default=0,
        )

    @staticmethod
    def _checkpoint_counter(state: Mapping[str, Any], name: str) -> int:
        value = state.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"learner host checkpoint {name} is invalid")
        return value

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

    @staticmethod
    def _percentile(values: tuple[float, ...] | deque[float], percentile: int) -> float:
        if not values:
            return 0.0
        return float(np.percentile(tuple(values), percentile))

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
        context = ray.get_runtime_context()
        self._node_id = context.get_node_id()
        self._accelerator_ids = tuple(
            str(value) for value in context.get_accelerator_ids().get("GPU", ())
        )
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

    def export_pbt_state(self) -> PBTModelState:
        return self._host.export_pbt_state()

    def get_stats(self) -> LearnerHostStats:
        return replace(
            self._host.get_stats(),
            node_id=self._node_id,
            accelerator_ids=self._accelerator_ids,
        )

    def get_checkpoint(self) -> LearnerHostCheckpoint:
        state = self._host.get_checkpoint_state()
        return LearnerHostCheckpoint(
            replay_cursor=state["replay_cursor"],
            payload=encode_learner_host_checkpoint(state),
        )

    def stop(self, **kwargs: Any) -> None:
        self._host.stop(**kwargs)
