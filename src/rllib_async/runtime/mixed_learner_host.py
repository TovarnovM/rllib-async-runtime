"""One replay replica and independent local learners for a mixed policy team."""

from __future__ import annotations

import copy
import math
import random
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import ray
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig

from rllib_async.learner import DQNLearnerAdapter, SACLearnerAdapter
from rllib_async.protocols import ReplayCursor, ReplaySnapshot, WeightsDescriptor
from rllib_async.protocols.episodes import SequenceEpisodeCodec
from rllib_async.replay import BatchProducerStats, FastReplay
from rllib_async.replay.batching import BatchProducerState
from rllib_async.replay.sequences import SequenceBatchCollator, SequenceConfig
from rllib_async.runtime.learner_host import (
    LearnerHost,
    LearnerHostError,
    LearnerHostState,
    LearnerHostStats,
    LearnerHostTick,
    decode_learner_host_checkpoint,
)


class MixedLearnerHost(LearnerHost):
    """Bounded round-robin updates using one authoritative replay subscription.

    Each policy has its own LearnerGroup, optimizer/target clock and random
    sampler. Groups are local: only the enclosing Ray actor reserves resources.
    The direct sampler has no pending queue to lose across a checkpoint.
    """

    def __init__(
        self,
        config: Mapping[str, SACConfig | DQNConfig],
        spaces: Mapping[str, tuple[Any, Any]],
        replay_actor: Any,
        codec: SequenceEpisodeCodec,
        *,
        member_id: str,
        publication_interval_updates: int,
        batch_size: int,
        batch_queue_capacity: int,
        batch_seed: int,
        replay_sync_max_bytes: int,
        replay_sync_interval_s: float = 0.0,
        validate_startup: bool = False,
        checkpoint_state: Mapping[str, Any] | bytes | None = None,
        allow_replay_ahead_on_restore: bool = False,
        pbt_state: object = None,
        learning_starts_satisfied: bool = False,
        trainable_modules: tuple[str, ...] | None = None,
        sequence_configs: Mapping[str, SequenceConfig] | None = None,
    ) -> None:
        if not isinstance(config, Mapping) or not config:
            raise ValueError("mixed module configurations must be a non-empty mapping")
        if any(not isinstance(mid, str) or not mid for mid in config):
            raise ValueError("mixed module IDs must be non-empty strings")
        if set(config) - set(spaces):
            raise ValueError("mixed module configurations require matching spaces")
        if not isinstance(codec, SequenceEpisodeCodec):
            raise TypeError("mixed learners require raw decision sequence episodes")
        if batch_queue_capacity != 0:
            raise ValueError(
                "mixed sequence learners require direct batching (capacity=0)"
            )
        if pbt_state is not None or learning_starts_satisfied:
            raise ValueError("mixed warm start/PBT is not part of the stage 2 contract")
        if type(batch_seed) is not int:
            raise ValueError("batch_seed must be an integer")
        if not isinstance(allow_replay_ahead_on_restore, bool):
            raise TypeError("allow_replay_ahead_on_restore must be a bool")
        for name, value in (
            ("publication_interval_updates", publication_interval_updates),
            ("batch_size", batch_size),
            ("replay_sync_max_bytes", replay_sync_max_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(replay_sync_interval_s, bool)
            or not isinstance(replay_sync_interval_s, int | float)
            or not math.isfinite(replay_sync_interval_s)
            or replay_sync_interval_s < 0
        ):
            raise ValueError("replay_sync_interval_s must be finite and non-negative")
        self._module_ids = tuple(sorted(config))
        selected = (
            self._module_ids if trainable_modules is None else tuple(trainable_modules)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected).issubset(config)
        ):
            raise ValueError("trainable_modules must be a non-empty unique team subset")
        self._trainable_modules = tuple(
            mid for mid in self._module_ids if mid in selected
        )
        sequence_configs = dict(sequence_configs or {})
        if set(sequence_configs) - set(config) or any(
            not isinstance(value, SequenceConfig) for value in sequence_configs.values()
        ):
            raise ValueError("sequence_configs must map team modules to SequenceConfig")
        self._replay_actor = replay_actor
        self._codec = codec
        self._member_id = member_id
        self._sync_max_bytes = replay_sync_max_bytes
        self._sync_interval_s = replay_sync_interval_s
        self._next_sync_at = 0.0
        self._sync_has_more = False
        self._allow_replay_ahead_on_restore = allow_replay_ahead_on_restore
        self._publication_interval = publication_interval_updates
        self._batch_size = batch_size
        self._adapters: dict[str, SACLearnerAdapter | DQNLearnerAdapter] = {}
        self._configs = dict(config)
        self._sequence_configs: dict[str, SequenceConfig] = {}
        self._collators: dict[str, SequenceBatchCollator] = {}
        self._recurrent: dict[str, bool] = {}
        self._batch_sizes: dict[str, int] = {}
        self._intensities: dict[str, float] = {}
        self._samplers = {
            mid: random.Random(batch_seed + index)
            for index, mid in enumerate(self._module_ids)
        }
        self._module_counters: dict[str, dict[str, int | float | bool]] = {
            mid: {
                "ingested_decisions": 0,
                "learned_positions": 0,
                "optimizer_updates": 0,
                "budget": 0.0,
                "ready": False,
            }
            for mid in self._module_ids
        }
        self._next_module = 0
        self._publication_epoch = 0
        self._last_published_update = 0
        self._fast_replay = FastReplay(codec)
        self._state = LearnerHostState.CREATED
        self._started_at: float | None = None
        self._rate_base_learner_updates = 0
        self._rate_base_learned_positions = 0
        self._sync_requests = 0
        self._snapshot_loads = 1
        self._delta_transactions = 0
        self._full_resyncs = 0
        self._updates_skipped_learning_start = 0
        self._weight_publications = 0
        self._last_learner_metrics: tuple[tuple[str, float], ...] = ()
        self._last_learner_results: object = None
        self._update_times_ms: deque[float] = deque(maxlen=1_024)
        self._batch_build_times_ms: deque[float] = deque(maxlen=1_024)
        self._batch_build_s = 0.0
        try:
            snapshot = ray.get(replay_actor.get_snapshot.remote())
            if not isinstance(snapshot, ReplaySnapshot):
                raise LearnerHostError("replay actor returned an invalid snapshot")
            self._fast_replay.load_snapshot(snapshot)
            for mid, module_config in self._configs.items():
                if not isinstance(module_config, SACConfig | DQNConfig):
                    raise TypeError(
                        "every mixed module requires SACConfig or DQNConfig"
                    )
                if type(module_config.n_step) is not int or module_config.n_step < 1:
                    raise ValueError("mixed module n_step must be a positive integer")
                size = module_config.train_batch_size_per_learner
                if type(size) is not int or size < 1:
                    raise ValueError(
                        "module train_batch_size_per_learner must be positive"
                    )
                intensity = module_config.training_intensity
                # Here intensity is explicitly valid learned positions/new decision.
                if isinstance(intensity, bool):
                    raise ValueError("module training_intensity must be numeric")
                intensity = 1.0 if intensity is None else float(intensity)
                if not math.isfinite(intensity) or intensity <= 0:
                    raise ValueError(
                        "module training_intensity must be finite and positive"
                    )
                self._batch_sizes[mid] = size
                self._intensities[mid] = intensity
                adapter_type = (
                    SACLearnerAdapter
                    if isinstance(module_config, SACConfig)
                    else DQNLearnerAdapter
                )
                adapter = adapter_type(
                    module_config,
                    spaces={mid: spaces[mid]},
                    member_id=member_id,
                    publication_interval_updates=publication_interval_updates,
                    validate_startup=validate_startup,
                    publish_updates=False,
                )
                self._adapters[mid] = adapter
                if set(adapter.get_published_weights().module_versions) != {mid}:
                    raise ValueError(
                        "each mixed module config must resolve to its own module ID"
                    )
                learner_module = adapter._learner_group._learner.module[mid]
                recurrent = learner_module.is_stateful()
                self._recurrent[mid] = recurrent
                self._sequence_configs[mid] = sequence_configs.get(
                    mid, SequenceConfig(burn_in=None if recurrent else 0)
                )
                if not recurrent:
                    self._sequence_configs[mid] = replace(
                        self._sequence_configs[mid], burn_in=0
                    )
                self._collators[mid] = SequenceBatchCollator(
                    gamma=module_config.gamma,
                    n_step=module_config.n_step,
                    validate_startup=validate_startup,
                    include_history=recurrent,
                )
            self._record_replay_ingestion(snapshot.episodes, initial=True)
            self._latest_weights = self._build_publication()
            if checkpoint_state is not None:
                if isinstance(checkpoint_state, bytes):
                    checkpoint_state = decode_learner_host_checkpoint(checkpoint_state)
                self._restore_checkpoint_state(checkpoint_state)
        except Exception:
            self.stop()
            raise

    @property
    def _learner_updates(self) -> int:
        return sum(adapter.learner_updates for adapter in self._adapters.values())

    @property
    def _learned_positions(self) -> int:
        return sum(
            int(value["learned_positions"]) for value in self._module_counters.values()
        )

    def _record_replay_ingestion(
        self, episodes: tuple[Any, ...], *, initial: bool = False
    ) -> None:
        for episode in episodes:
            # Immutable, validated payload metadata; no decode or size traversal.
            for mid, transitions in episode.payload.encoded_module_transitions:
                if mid not in self._module_counters:
                    continue
                clock = self._module_counters[mid]
                previous = int(clock["ingested_decisions"])
                current = previous + len(transitions)
                threshold = self._configs[mid].num_steps_sampled_before_learning_starts
                clock["ingested_decisions"] = current
                clock["ready"] = current >= threshold
                if not initial and mid in self._trainable_modules:
                    eligible = max(0, current - threshold) - max(
                        0, previous - threshold
                    )
                    clock["budget"] = (
                        float(clock["budget"]) + eligible * self._intensities[mid]
                    )

    def start(self) -> None:
        if self._state is LearnerHostState.RUNNING:
            return
        if self._state is LearnerHostState.PAUSED:
            self.resume()
            return
        self._require_state(LearnerHostState.CREATED)
        self._started_at = time.monotonic()
        self._state = LearnerHostState.RUNNING

    def pause(self, *, timeout_s: float | None = None) -> None:
        self._deadline(timeout_s)
        if self._state is LearnerHostState.PAUSED:
            return
        self._require_state(LearnerHostState.RUNNING)
        self._state = LearnerHostState.PAUSED

    def resume(self) -> None:
        if self._state is LearnerHostState.RUNNING:
            return
        self._require_state(LearnerHostState.PAUSED)
        self._state = LearnerHostState.RUNNING

    def tick(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        max_updates: int,
        sampled_module_steps: Mapping[str, int] | None = None,
    ) -> LearnerHostTick:
        self._require_state(LearnerHostState.RUNNING)
        if type(max_updates) is not int or max_updates < 1:
            raise ValueError("max_updates must be a positive integer")
        try:
            transactions, has_more = self._sync_once(respect_interval=True)
            updates, publication = self._consume_ready(max_updates)
            return LearnerHostTick(
                synced_transactions=transactions,
                sync_has_more=has_more,
                updates_performed=updates,
                updates_skipped_learning_start=0,
                published_weights=publication,
                learner_metrics=(),
                retry_after_s=max(self._next_sync_at - time.monotonic(), 0.0)
                if not updates and not has_more
                else 0.0,
            )
        except Exception:
            self._state = LearnerHostState.FAILED
            raise

    def _consume_ready(self, max_updates: int) -> tuple[int, WeightsDescriptor | None]:
        updates = 0
        missed = 0
        available = dict(self._fast_replay.module_transition_counts)
        while updates < max_updates and missed < len(self._trainable_modules):
            mid = self._trainable_modules[self._next_module]
            self._next_module = (self._next_module + 1) % len(self._trainable_modules)
            clock = self._module_counters[mid]
            budget = int(float(clock["budget"]))
            if not clock["ready"] or budget < 1 or not available.get(mid, 0):
                missed += 1
                continue
            missed = 0
            spec = self._sequence_configs[mid]
            length = min(spec.learning_length, self._batch_sizes[mid], budget)
            spec = replace(spec, learning_length=length)
            windows_count = max(1, min(self._batch_sizes[mid], budget) // length)
            started = time.monotonic()
            windows = self._fast_replay.sample_sequences(
                mid,
                windows_count,
                rng=self._samplers[mid],
                sequence_config=spec,
                n_step=self._configs[mid].n_step,
            )
            batch = self._collators[mid].collate(windows)
            batch.pop("decision_horizons", None)
            if not self._recurrent[mid]:
                batch.pop("recurrent", None)
            elapsed = time.monotonic() - started
            self._batch_build_s += elapsed
            self._batch_build_times_ms.append(elapsed * 1_000)
            positions = len(batch["rewards"])
            started = time.monotonic()
            update = self._adapters[mid].update_modules(
                {mid: batch},
                sampled_env_steps=int(clock["ingested_decisions"]),
                sampled_agent_steps=int(clock["ingested_decisions"]),
            )
            if not update.performed:
                raise LearnerHostError(
                    "ready mixed learner unexpectedly skipped its update"
                )
            clock["learned_positions"] = int(clock["learned_positions"]) + positions
            clock["optimizer_updates"] = update.learner_updates
            clock["budget"] = max(0.0, float(clock["budget"]) - positions)
            self._update_times_ms.append((time.monotonic() - started) * 1_000)
            self._last_learner_results = update.learner_results
            updates += 1
        return updates, self._maybe_publish()

    def _build_publication(self) -> WeightsDescriptor:
        state: dict[str, Any] = {}
        for adapter in self._adapters.values():
            state.update(adapter.get_inference_state())
        return WeightsDescriptor(
            member_id=self._member_id,
            module_versions={mid: self._publication_epoch for mid in self._module_ids},
            learner_updates=self._learner_updates,
            published_at_monotonic=time.monotonic(),
            state=state,
        )

    def _maybe_publish(self, *, force: bool = False) -> WeightsDescriptor | None:
        difference = self._learner_updates - self._last_published_update
        if not difference or (not force and difference < self._publication_interval):
            return None
        self._publication_epoch += 1
        publication = self._build_publication()
        self._latest_weights = publication
        self._last_published_update = self._learner_updates
        self._weight_publications += 1
        return publication

    def get_published_weights(self) -> WeightsDescriptor:
        self._require_not_stopped()
        return self._latest_weights

    def export_pbt_state(self) -> object:
        raise ValueError("mixed-team PBT is implemented in stage 3")

    def drain(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        max_updates: int | None = None,
        timeout_s: float | None = None,
        sampled_module_steps: Mapping[str, int] | None = None,
    ) -> LearnerHostTick:
        if self._state not in {LearnerHostState.RUNNING, LearnerHostState.PAUSED}:
            raise LearnerHostError("cannot drain a mixed learner that is not running")
        if max_updates is not None and (
            type(max_updates) is not int or max_updates < 0
        ):
            raise ValueError("max_updates must be non-negative or None")
        deadline = self._deadline(timeout_s)
        transactions = 0
        try:
            has_more = True
            while has_more:
                count, has_more = self._sync_once()
                transactions += count
                if has_more and self._remaining(deadline) == 0:
                    raise TimeoutError(
                        "timed out draining mixed replay synchronization"
                    )
            self._fast_replay.wait_for_idle(timeout=self._remaining(deadline))
            self._state = LearnerHostState.PAUSED
            # Direct mode has no queued selections to consume during checkpoint.
            publication = self._maybe_publish(force=True)
            return LearnerHostTick(transactions, False, 0, 0, publication, ())
        except Exception:
            self._state = LearnerHostState.FAILED
            raise

    def get_stats(self) -> LearnerHostStats:
        self._require_not_stopped()
        self._refresh_learner_metrics()
        elapsed = (
            max(time.monotonic() - self._started_at, 0.0)
            if self._started_at is not None
            else 0.0
        )
        updates = self._learner_updates
        producer_state = {
            LearnerHostState.CREATED: BatchProducerState.CREATED,
            LearnerHostState.RUNNING: BatchProducerState.RUNNING,
            LearnerHostState.PAUSED: BatchProducerState.PAUSED,
            LearnerHostState.FAILED: BatchProducerState.FAILED,
        }[self._state]
        producer = BatchProducerStats(
            state=producer_state,
            prefetch_enabled=False,
            queue_size=0,
            queue_capacity=0,
            queue_high_watermark=0,
            batches_produced=updates,
            batches_consumed=updates,
            batches_dropped=0,
            producer_failures=int(self._state is LearnerHostState.FAILED),
            queue_full_events=0,
            backpressure_s=0.0,
            data_wait_calls=0,
            data_wait_timeouts=0,
            data_wait_s=0.0,
            last_data_wait_ms=0.0,
            batch_builds=updates,
            batch_build_s=self._batch_build_s,
            batch_build_ms_p50=self._percentile(self._batch_build_times_ms, 50),
            batch_build_ms_p95=self._percentile(self._batch_build_times_ms, 95),
        )
        return LearnerHostStats(
            state=self._state,
            sync_requests=self._sync_requests,
            snapshot_loads=self._snapshot_loads,
            delta_transactions=self._delta_transactions,
            full_resyncs=self._full_resyncs,
            learner_updates=updates,
            updates_skipped_learning_start=0,
            weight_publications=self._weight_publications,
            updates_per_s=(updates - self._rate_base_learner_updates) / elapsed
            if elapsed
            else 0.0,
            samples_per_s=(self._learned_positions - self._rate_base_learned_positions)
            / elapsed
            if elapsed
            else 0.0,
            data_wait_fraction=0.0,
            batch_queue_empty_fraction=0.0,
            update_time_ms_p50=self._percentile(self._update_times_ms, 50),
            update_time_ms_p95=self._percentile(self._update_times_ms, 95),
            latest_module_version=self._publication_epoch,
            last_learner_metrics=self._last_learner_metrics,
            fast_replay=self._fast_replay.get_stats(),
            batch_producer=producer,
            module_counters=tuple(
                (mid, tuple(sorted(self._module_counters[mid].items())))
                for mid in self._module_ids
            ),
        )

    def get_checkpoint_state(self) -> dict[str, Any]:
        self._require_state(LearnerHostState.PAUSED)
        replay = self._fast_replay.get_stats()
        if (
            replay.cursor is None
            or replay.active_cursor != replay.cursor
            or replay.rebuild_in_progress
        ):
            raise LearnerHostError(
                "mixed replay must be fully materialized before checkpoint"
            )
        self._refresh_learner_metrics()
        return {
            "state_version": 1,
            "kind": "mixed_team",
            "member_id": self._member_id,
            "replay_cursor": replay.cursor,
            "trainable_modules": self._trainable_modules,
            "sequence_configs": dict(self._sequence_configs),
            "training_intensities": dict(self._intensities),
            "adapters": {
                mid: adapter.get_state() for mid, adapter in self._adapters.items()
            },
            "sampler_rng_states": {
                mid: rng.getstate() for mid, rng in self._samplers.items()
            },
            "module_counters": copy.deepcopy(self._module_counters),
            "next_module": self._next_module,
            "publication_epoch": self._publication_epoch,
            "last_published_update": self._last_published_update,
            "latest_weights": copy.deepcopy(self._latest_weights),
            "sync_requests": self._sync_requests,
            "snapshot_loads": self._snapshot_loads,
            "delta_transactions": self._delta_transactions,
            "full_resyncs": self._full_resyncs,
            "weight_publications": self._weight_publications,
            "last_learner_metrics": self._last_learner_metrics,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "torch_cuda_rng_states": tuple(
                value.clone() for value in torch.cuda.get_rng_state_all()
            )
            if torch.cuda.is_available()
            else (),
        }

    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("state_version") != 1
            or state.get("kind") != "mixed_team"
        ):
            raise ValueError("unsupported mixed learner checkpoint")
        if state.get("member_id") != self._member_id:
            raise ValueError("mixed checkpoint member_id does not match")
        if (
            state.get("trainable_modules") != self._trainable_modules
            or state.get("sequence_configs") != self._sequence_configs
            or state.get("training_intensities") != self._intensities
        ):
            raise ValueError("mixed checkpoint sampling/training contract differs")
        cursor = state.get("replay_cursor")
        replay = self._fast_replay.get_stats()
        if not isinstance(cursor, ReplayCursor) or replay.cursor is None:
            raise ValueError("mixed checkpoint replay cursor is invalid")
        same = cursor == replay.cursor
        ahead = (
            self._allow_replay_ahead_on_restore
            and cursor.store_generation == replay.cursor.store_generation
            and cursor.mutation_seq <= replay.cursor.mutation_seq
        )
        if not (same or ahead) or replay.active_cursor != replay.cursor:
            raise ValueError("mixed checkpoint does not match authoritative replay")
        adapters = state.get("adapters")
        samplers = state.get("sampler_rng_states")
        counters = state.get("module_counters")
        if any(
            not isinstance(value, Mapping) or set(value) != set(self._module_ids)
            for value in (adapters, samplers, counters)
        ):
            raise ValueError("mixed checkpoint module IDs differ")
        for mid, clock in counters.items():
            if not isinstance(clock, Mapping) or set(clock) != set(
                self._module_counters[mid]
            ):
                raise ValueError("mixed checkpoint module counters are invalid")
            for name in (
                "ingested_decisions",
                "learned_positions",
                "optimizer_updates",
            ):
                self._checkpoint_counter(clock, name)
            if (
                type(clock["ready"]) is not bool
                or isinstance(clock["budget"], bool)
                or not isinstance(clock["budget"], int | float)
                or not math.isfinite(clock["budget"])
                or clock["budget"] < 0
            ):
                raise ValueError("mixed checkpoint budget/readiness is invalid")
            if not isinstance(adapters[mid], Mapping) or clock[
                "optimizer_updates"
            ] != adapters[mid].get("learner_updates"):
                raise ValueError("mixed checkpoint optimizer clocks disagree")
            threshold = self._configs[mid].num_steps_sampled_before_learning_starts
            if clock["ready"] != (clock["ingested_decisions"] >= threshold):
                raise ValueError(
                    "mixed checkpoint readiness contradicts ingested decisions"
                )
            maximum = (
                max(0, clock["ingested_decisions"] - threshold)
                * self._intensities[mid]
            )
            if clock["budget"] + clock["learned_positions"] > maximum + 1e-7:
                raise ValueError(
                    "mixed checkpoint budget exceeds its unique-data allowance"
                )
            if mid not in self._trainable_modules and any(
                clock[name]
                for name in ("budget", "learned_positions", "optimizer_updates")
            ):
                raise ValueError("a frozen module cannot have learner progress")
            random.Random().setstate(samplers[mid])
        next_module = self._checkpoint_counter(state, "next_module")
        if next_module >= len(self._trainable_modules):
            raise ValueError("mixed checkpoint round-robin frontier is invalid")
        epoch = self._checkpoint_counter(state, "publication_epoch")
        published_update = self._checkpoint_counter(state, "last_published_update")
        weights = state.get("latest_weights")
        if (
            not isinstance(weights, WeightsDescriptor)
            or weights.member_id != self._member_id
            or dict(weights.module_versions) != {mid: epoch for mid in self._module_ids}
            or weights.learner_updates != published_update
            or not isinstance(weights.state, Mapping)
            or set(weights.state) != set(self._module_ids)
        ):
            raise ValueError("mixed checkpoint publication is invalid")
        updates = sum(clock["optimizer_updates"] for clock in counters.values())
        if published_update > updates:
            raise ValueError("mixed checkpoint publication exceeds learner progress")
        restored_counters = {
            name: self._checkpoint_counter(state, name)
            for name in (
                "sync_requests",
                "snapshot_loads",
                "delta_transactions",
                "full_resyncs",
                "weight_publications",
            )
        }
        random.Random().setstate(state.get("python_rng_state"))
        np.random.RandomState().set_state(state.get("numpy_rng_state"))
        validator = next(iter(self._adapters.values()))._validated_rng_state
        torch_rng = validator(
            state.get("torch_rng_state"),
            name="torch_rng_state",
            expected_numel=torch.get_rng_state().numel(),
        )
        current_cuda = (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
        )
        cuda_rng = state.get("torch_cuda_rng_states")
        if not isinstance(cuda_rng, tuple) or len(cuda_rng) != len(current_cuda):
            raise ValueError("mixed checkpoint CUDA RNG device count differs")
        cuda_rng = tuple(
            validator(
                value, name="torch_cuda_rng_states", expected_numel=current.numel()
            )
            for value, current in zip(cuda_rng, current_cuda)
        )
        # These adapters are still constructor-owned and cannot be observed live.
        for mid, adapter in self._adapters.items():
            adapter.set_state(adapters[mid])
            self._samplers[mid].setstate(samplers[mid])
        self._module_counters = copy.deepcopy(dict(counters))
        self._next_module = next_module
        self._publication_epoch = epoch
        self._last_published_update = published_update
        self._latest_weights = copy.deepcopy(weights)
        for name, value in restored_counters.items():
            setattr(self, f"_{name}", value)
        self._snapshot_loads += 1
        self._last_learner_metrics = tuple(state.get("last_learner_metrics", ()))
        self._rate_base_learner_updates = self._learner_updates
        self._rate_base_learned_positions = self._learned_positions
        # Restore process RNG once, after every local learner and optimizer.
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(torch_rng)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(cuda_rng))

    def stop(self, *, timeout_s: float | None = None) -> None:
        if self._state is LearnerHostState.STOPPED:
            return
        errors: list[BaseException] = []
        for adapter in self._adapters.values():
            try:
                adapter.close()
            except BaseException as error:
                errors.append(error)
        try:
            self._fast_replay.close(timeout=timeout_s)
        except BaseException as error:
            errors.append(error)
        self._state = LearnerHostState.STOPPED
        if errors:
            raise LearnerHostError("mixed learner shutdown failed") from errors[0]
