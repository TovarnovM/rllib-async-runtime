"""Asynchronous whole-episode rollout actors and bounded coordination."""

from __future__ import annotations

import copy
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import DEFAULT_MODULE_ID

from rllib_async.protocols import (
    CommitAck,
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
    WeightsDescriptor,
)
from rllib_async.rollout.episode_runner import (
    EpisodeRolloutMetrics,
    EpisodeRolloutResult,
    EpisodeRunner,
    accept_weight_publication,
    make_episode_id,
)

ROLLOUT_GROUP_CHECKPOINT_VERSION = 1


class RolloutGroupError(RuntimeError):
    """The asynchronous rollout group cannot safely make progress."""


class RolloutGroupState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RolloutCompletion:
    """One replay acknowledgement with rollout provenance and lag."""

    episode: EpisodeEnvelope
    metrics: EpisodeRolloutMetrics
    acknowledgement: CommitAck
    policy_version_lag: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_version_lag, int)
            or isinstance(self.policy_version_lag, bool)
            or self.policy_version_lag < 0
        ):
            raise ValueError("policy_version_lag must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RolloutGroupStats:
    state: RolloutGroupState
    runner_count: int
    runner_generations: tuple[tuple[str, int], ...]
    episodes_collected: int
    episodes_committed: int
    duplicate_commits: int
    env_steps: int
    agent_steps: int
    env_steps_per_s: float
    agent_steps_per_s: float
    sample_calls_started: int
    sample_failures: int
    commit_failures: int
    runner_restarts: int
    pending_sample_calls: int
    pending_episode_commits: int
    outstanding_high_watermark: int
    pending_commit_high_watermark: int
    pending_commit_low_watermark: int
    backpressured: bool
    backpressure_events: int
    backpressure_fraction: float
    episode_time_ms_p50: float
    episode_time_ms_p95: float
    policy_version_lag_p50: float
    policy_version_lag_p95: float


@ray.remote(max_concurrency=1)
class EpisodeRolloutActor:
    """Ray process boundary around one finite-call `EpisodeRunner`."""

    def __init__(
        self,
        config: AlgorithmConfig,
        codec: FlatEpisodeCodec,
        *,
        member_id: str,
        runner_id: str,
        runner_generation: int,
        max_episode_steps: int,
        initial_weights: WeightsDescriptor,
        worker_index: int,
    ) -> None:
        self._runner = EpisodeRunner(
            config,
            codec,
            member_id=member_id,
            runner_id=runner_id,
            runner_generation=runner_generation,
            max_episode_steps=max_episode_steps,
            initial_weights=initial_weights,
            worker_index=worker_index,
        )

    def collect_episode(
        self,
        weights: WeightsDescriptor | None = None,
        *,
        explore: bool = True,
    ) -> EpisodeRolloutResult:
        return self._runner.collect_episode(weights, explore=explore)

    def close(self) -> None:
        self._runner.close()


@dataclass(frozen=True, slots=True)
class _PendingCommit:
    runner_id: str
    result: EpisodeRolloutResult
    policy_version_lag: int


class AsyncRolloutGroup:
    """Keep independent episode actors busy behind a strict commit-slot bound."""

    def __init__(
        self,
        config: AlgorithmConfig,
        codec: FlatEpisodeCodec,
        replay_actor: Any,
        *,
        member_id: str,
        initial_weights: WeightsDescriptor,
        runner_count: int,
        max_episode_steps: int,
        pending_commit_high_watermark: int,
        pending_commit_low_watermark: int,
        metrics_window: int = 2_048,
        num_cpus_per_runner: float = 1.0,
        explore: bool = True,
        checkpoint_state: Mapping[str, Any] | None = None,
    ) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before AsyncRolloutGroup")
        if not isinstance(config, AlgorithmConfig):
            raise TypeError("config must be an AlgorithmConfig")
        if not isinstance(codec, FlatEpisodeCodec):
            raise TypeError("Phase 5 AsyncRolloutGroup requires FlatEpisodeCodec")
        make_episode_id(member_id, "runner-0", 0, 0)
        if (
            not isinstance(runner_count, int)
            or isinstance(runner_count, bool)
            or not 1 <= runner_count <= 16
        ):
            raise ValueError("runner_count must be between 1 and 16")
        if (
            not isinstance(pending_commit_high_watermark, int)
            or isinstance(pending_commit_high_watermark, bool)
            or pending_commit_high_watermark < 1
        ):
            raise ValueError("pending_commit_high_watermark must be positive")
        if (
            not isinstance(pending_commit_low_watermark, int)
            or isinstance(pending_commit_low_watermark, bool)
            or pending_commit_low_watermark < 0
            or pending_commit_low_watermark >= pending_commit_high_watermark
        ):
            raise ValueError(
                "pending_commit_low_watermark must be non-negative and below high"
            )
        if (
            not isinstance(max_episode_steps, int)
            or isinstance(max_episode_steps, bool)
            or max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        if (
            not isinstance(metrics_window, int)
            or isinstance(metrics_window, bool)
            or metrics_window < 1
        ):
            raise ValueError("metrics_window must be a positive integer")
        if (
            not isinstance(num_cpus_per_runner, int | float)
            or isinstance(num_cpus_per_runner, bool)
            or not math.isfinite(num_cpus_per_runner)
            or num_cpus_per_runner < 0
        ):
            raise ValueError("num_cpus_per_runner must be finite and non-negative")
        if not isinstance(explore, bool):
            raise ValueError("explore must be a boolean")
        accept_weight_publication(
            None,
            initial_weights,
            member_id=member_id,
            module_ids={DEFAULT_MODULE_ID},
        )

        self._config = config.copy(copy_frozen=False)
        self._codec = codec
        self._replay_actor = replay_actor
        self._member_id = member_id
        self._latest_weights = copy.deepcopy(initial_weights)
        self._runner_count = runner_count
        self._max_episode_steps = max_episode_steps
        self._high_watermark = pending_commit_high_watermark
        self._low_watermark = pending_commit_low_watermark
        self._metrics_window = metrics_window
        self._num_cpus_per_runner = float(num_cpus_per_runner)
        self._explore = explore
        self._state = RolloutGroupState.CREATED
        self._runners: dict[str, Any] = {}
        self._runner_weight_versions: dict[str, FrozenVersions] = {}
        self._runner_generations = {
            f"runner-{index}": 0 for index in range(runner_count)
        }
        self._worker_indices = {
            f"runner-{index}": index + 1 for index in range(runner_count)
        }
        self._idle_runners = deque(self._runner_generations)
        self._pending_samples: dict[ray.ObjectRef, str] = {}
        self._pending_commits: dict[ray.ObjectRef, _PendingCommit] = {}
        self._episodes_collected = 0
        self._episodes_committed = 0
        self._duplicate_commits = 0
        self._env_steps = 0
        self._agent_steps = 0
        self._sample_calls_started = 0
        self._sample_failures = 0
        self._commit_failures = 0
        self._runner_restarts = 0
        self._outstanding_high_watermark = 0
        self._backpressure_events = 0
        self._backpressure_s = 0.0
        self._backpressure_started: float | None = None
        self._started_at: float | None = None
        self._rate_base_env_steps = 0
        self._rate_base_agent_steps = 0
        self._rate_base_backpressure_s = 0.0
        self._episode_times_ms: deque[float] = deque(maxlen=metrics_window)
        self._policy_lags: deque[int] = deque(maxlen=metrics_window)
        if checkpoint_state is not None:
            self._restore_checkpoint_state(checkpoint_state)

        try:
            for runner_id in self._runner_generations:
                self._runners[runner_id] = self._new_actor(runner_id)
        except Exception:
            for actor in self._runners.values():
                ray.kill(actor, no_restart=True)
            raise

    def start(self) -> None:
        if self._state is RolloutGroupState.RUNNING:
            return
        if self._state is RolloutGroupState.PAUSED:
            self.resume()
            return
        if self._state is RolloutGroupState.STOPPED:
            raise RolloutGroupError("cannot start a stopped rollout group")
        self._state = RolloutGroupState.RUNNING
        self._started_at = time.monotonic()
        self._schedule_available()

    def pause(self) -> None:
        """Stop scheduling at episode boundaries while preserving pending work."""

        if self._state is RolloutGroupState.PAUSED:
            return
        self._require_running()
        self._leave_backpressure()
        self._state = RolloutGroupState.PAUSED

    def resume(self) -> None:
        if self._state is RolloutGroupState.RUNNING:
            return
        if self._state is not RolloutGroupState.PAUSED:
            raise RolloutGroupError(
                f"cannot resume rollout group in state {self._state.value!r}"
            )
        self._state = RolloutGroupState.RUNNING
        self._schedule_available()

    def drain(self, *, timeout_s: float | None = None) -> list[RolloutCompletion]:
        """Finish active episodes and commits without starting new episodes."""

        if self._state is RolloutGroupState.RUNNING:
            self.pause()
        elif self._state is not RolloutGroupState.PAUSED:
            raise RolloutGroupError(
                f"cannot drain rollout group in state {self._state.value!r}"
            )
        if timeout_s is not None and (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and non-negative or None")
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        completions: list[RolloutCompletion] = []
        while self._pending_samples or self._pending_commits:
            remaining = (
                None if deadline is None else max(deadline - time.monotonic(), 0.0)
            )
            if remaining == 0:
                raise TimeoutError("timed out draining rollout group")
            completions.extend(
                self.poll(
                    timeout_s=(min(remaining, 0.05) if remaining is not None else 0.05),
                    max_events=max(
                        len(self._pending_samples) + len(self._pending_commits),
                        1,
                    ),
                )
            )
        return completions

    def update_weights(self, weights: WeightsDescriptor) -> bool:
        """Publish weights for installation by each actor's next episode call."""

        self._require_not_stopped()
        if not accept_weight_publication(
            self._latest_weights,
            weights,
            member_id=self._member_id,
            module_ids={DEFAULT_MODULE_ID},
        ):
            return False
        self._latest_weights = copy.deepcopy(weights)
        return True

    def poll(
        self,
        *,
        timeout_s: float | None = 0.0,
        max_events: int = 1,
    ) -> list[RolloutCompletion]:
        """Advance ready sample/commit RPCs without waiting for an episode barrier."""

        self._require_pollable()
        if timeout_s is not None and (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and non-negative or None")
        if (
            not isinstance(max_events, int)
            or isinstance(max_events, bool)
            or max_events < 1
        ):
            raise ValueError("max_events must be a positive integer")

        refs = [*self._pending_samples, *self._pending_commits]
        if not refs:
            self._schedule_available()
            refs = [*self._pending_samples, *self._pending_commits]
        if not refs:
            return []

        ready, remaining = ray.wait(
            refs,
            num_returns=1,
            timeout=timeout_s,
        )
        if ready and max_events > 1 and remaining:
            additionally_ready, _ = ray.wait(
                remaining,
                num_returns=min(max_events - 1, len(remaining)),
                timeout=0,
            )
            ready.extend(additionally_ready)
        completions: list[RolloutCompletion] = []
        for ref in ready:
            if ref in self._pending_samples:
                self._finish_sample(ref)
            else:
                completions.append(self._finish_commit(ref))
            self._relieve_backpressure_if_due()
            self._schedule_available()
        return completions

    def restart_runner(self, runner_id: str) -> int:
        """Replace one actor and advance its generation before the next episode."""

        self._require_not_stopped()
        if runner_id not in self._runners:
            raise KeyError(runner_id)
        actor = self._runners[runner_id]
        for ref, active_runner_id in tuple(self._pending_samples.items()):
            if active_runner_id == runner_id:
                self._pending_samples.pop(ref)
                ray.cancel(ref)
        self._idle_runners = deque(
            value for value in self._idle_runners if value != runner_id
        )
        ray.kill(actor, no_restart=True)
        self._runner_generations[runner_id] += 1
        self._runners[runner_id] = self._new_actor(runner_id)
        self._idle_runners.append(runner_id)
        self._runner_restarts += 1
        self._relieve_backpressure_if_due()
        if self._state is RolloutGroupState.RUNNING:
            self._schedule_available()
        return self._runner_generations[runner_id]

    def get_stats(self) -> RolloutGroupStats:
        now = time.monotonic()
        elapsed = (
            max(now - self._started_at, 0.0) if self._started_at is not None else 0.0
        )
        backpressure_s = self._backpressure_s
        if self._backpressure_started is not None:
            backpressure_s += now - self._backpressure_started
        return RolloutGroupStats(
            state=self._state,
            runner_count=self._runner_count,
            runner_generations=tuple(sorted(self._runner_generations.items())),
            episodes_collected=self._episodes_collected,
            episodes_committed=self._episodes_committed,
            duplicate_commits=self._duplicate_commits,
            env_steps=self._env_steps,
            agent_steps=self._agent_steps,
            env_steps_per_s=(
                (self._env_steps - self._rate_base_env_steps) / elapsed
                if elapsed
                else 0.0
            ),
            agent_steps_per_s=(
                (self._agent_steps - self._rate_base_agent_steps) / elapsed
                if elapsed
                else 0.0
            ),
            sample_calls_started=self._sample_calls_started,
            sample_failures=self._sample_failures,
            commit_failures=self._commit_failures,
            runner_restarts=self._runner_restarts,
            pending_sample_calls=len(self._pending_samples),
            pending_episode_commits=len(self._pending_commits),
            outstanding_high_watermark=self._outstanding_high_watermark,
            pending_commit_high_watermark=self._high_watermark,
            pending_commit_low_watermark=self._low_watermark,
            backpressured=self._backpressure_started is not None,
            backpressure_events=self._backpressure_events,
            backpressure_fraction=(
                min(
                    (backpressure_s - self._rate_base_backpressure_s) / elapsed,
                    1.0,
                )
                if elapsed
                else 0.0
            ),
            episode_time_ms_p50=self._percentile(self._episode_times_ms, 50),
            episode_time_ms_p95=self._percentile(self._episode_times_ms, 95),
            policy_version_lag_p50=self._percentile(self._policy_lags, 50),
            policy_version_lag_p95=self._percentile(self._policy_lags, 95),
        )

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Capture drained rollout state for collision-free actor recreation."""

        if self._state is not RolloutGroupState.PAUSED:
            raise RolloutGroupError("rollout group must be paused before checkpoint")
        if self._pending_samples or self._pending_commits:
            raise RolloutGroupError("rollout group must be drained before checkpoint")
        if set(self._idle_runners) != set(self._runners):
            raise RolloutGroupError(
                "all rollout runners must be idle before checkpoint"
            )
        return {
            "state_version": ROLLOUT_GROUP_CHECKPOINT_VERSION,
            "member_id": self._member_id,
            "runner_count": self._runner_count,
            "latest_module_versions": dict(self._latest_weights.module_versions),
            "runner_generations": dict(self._runner_generations),
            "episodes_collected": self._episodes_collected,
            "episodes_committed": self._episodes_committed,
            "duplicate_commits": self._duplicate_commits,
            "env_steps": self._env_steps,
            "agent_steps": self._agent_steps,
            "sample_calls_started": self._sample_calls_started,
            "sample_failures": self._sample_failures,
            "commit_failures": self._commit_failures,
            "runner_restarts": self._runner_restarts,
            "outstanding_high_watermark": self._outstanding_high_watermark,
            "backpressure_events": self._backpressure_events,
            "backpressure_s": self._backpressure_s,
            "episode_times_ms": tuple(self._episode_times_ms),
            "policy_lags": tuple(self._policy_lags),
        }

    def stop(self) -> None:
        if self._state is RolloutGroupState.STOPPED:
            return
        self._leave_backpressure()
        for ref in (*self._pending_samples, *self._pending_commits):
            ray.cancel(ref)
        for actor in self._runners.values():
            ray.kill(actor, no_restart=True)
        self._pending_samples.clear()
        self._pending_commits.clear()
        self._idle_runners.clear()
        self._state = RolloutGroupState.STOPPED

    def __enter__(self) -> AsyncRolloutGroup:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _new_actor(self, runner_id: str) -> Any:
        runner_generation = self._runner_generations[runner_id]
        runner_config = self._config.copy(copy_frozen=False)
        if runner_config.seed is not None:
            runner_config.seed = int(runner_config.seed) + (
                runner_generation * self._runner_count
            )
        actor = EpisodeRolloutActor.options(
            num_cpus=self._num_cpus_per_runner,
        ).remote(
            runner_config,
            self._codec,
            member_id=self._member_id,
            runner_id=runner_id,
            runner_generation=runner_generation,
            max_episode_steps=self._max_episode_steps,
            initial_weights=self._latest_weights,
            worker_index=self._worker_indices[runner_id],
        )
        self._runner_weight_versions[runner_id] = FrozenVersions(
            self._latest_weights.module_versions
        )
        return actor

    def _restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("rollout checkpoint state must be a mapping")
        if state.get("state_version") != ROLLOUT_GROUP_CHECKPOINT_VERSION:
            raise ValueError("unsupported rollout checkpoint state version")
        if state.get("member_id") != self._member_id:
            raise ValueError("rollout checkpoint member_id does not match")
        if state.get("runner_count") != self._runner_count:
            raise ValueError("rollout checkpoint runner_count does not match")
        if state.get("latest_module_versions") != dict(
            self._latest_weights.module_versions
        ):
            raise ValueError("rollout checkpoint weights do not match learner")

        generations = state.get("runner_generations")
        if not isinstance(generations, Mapping) or set(generations) != set(
            self._runner_generations
        ):
            raise ValueError("rollout checkpoint runner generations are invalid")
        restored_generations: dict[str, int] = {}
        for runner_id, generation in generations.items():
            if (
                not isinstance(runner_id, str)
                or not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation < 0
            ):
                raise ValueError("rollout checkpoint runner generations are invalid")
            restored_generations[runner_id] = generation + 1

        self._episodes_collected = self._checkpoint_counter(
            state,
            "episodes_collected",
        )
        self._episodes_committed = self._checkpoint_counter(
            state,
            "episodes_committed",
        )
        self._duplicate_commits = self._checkpoint_counter(
            state,
            "duplicate_commits",
        )
        self._env_steps = self._checkpoint_counter(state, "env_steps")
        self._agent_steps = self._checkpoint_counter(state, "agent_steps")
        self._sample_calls_started = self._checkpoint_counter(
            state,
            "sample_calls_started",
        )
        self._sample_failures = self._checkpoint_counter(
            state,
            "sample_failures",
        )
        self._commit_failures = self._checkpoint_counter(
            state,
            "commit_failures",
        )
        self._runner_restarts = (
            self._checkpoint_counter(state, "runner_restarts") + self._runner_count
        )
        self._outstanding_high_watermark = self._checkpoint_counter(
            state,
            "outstanding_high_watermark",
        )
        self._backpressure_events = self._checkpoint_counter(
            state,
            "backpressure_events",
        )
        backpressure_s = state.get("backpressure_s")
        if (
            not isinstance(backpressure_s, int | float)
            or isinstance(backpressure_s, bool)
            or not math.isfinite(backpressure_s)
            or backpressure_s < 0
        ):
            raise ValueError("rollout checkpoint backpressure_s is invalid")
        self._backpressure_s = float(backpressure_s)
        self._rate_base_env_steps = self._env_steps
        self._rate_base_agent_steps = self._agent_steps
        self._rate_base_backpressure_s = self._backpressure_s

        episode_times = self._checkpoint_window(
            state,
            "episode_times_ms",
            integer=False,
        )
        policy_lags = self._checkpoint_window(
            state,
            "policy_lags",
            integer=True,
        )
        if self._episodes_committed > self._episodes_collected:
            raise ValueError("rollout checkpoint commits exceed collected episodes")
        if self._sample_calls_started < self._episodes_collected:
            raise ValueError("rollout checkpoint sample count is inconsistent")

        self._runner_generations = restored_generations
        self._idle_runners = deque(restored_generations)
        self._episode_times_ms = deque(episode_times, maxlen=self._metrics_window)
        self._policy_lags = deque(policy_lags, maxlen=self._metrics_window)

    @staticmethod
    def _checkpoint_counter(state: Mapping[str, Any], name: str) -> int:
        value = state.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"rollout checkpoint {name} is invalid")
        return value

    def _checkpoint_window(
        self,
        state: Mapping[str, Any],
        name: str,
        *,
        integer: bool,
    ) -> tuple[float | int, ...]:
        values = state.get(name)
        if not isinstance(values, tuple) or len(values) > self._metrics_window:
            raise ValueError(f"rollout checkpoint {name} is invalid")
        for value in values:
            if (
                not isinstance(value, int if integer else int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"rollout checkpoint {name} is invalid")
        return values

    def _schedule_available(self) -> None:
        if (
            self._state is not RolloutGroupState.RUNNING
            or self._backpressure_started is not None
        ):
            return
        while self._idle_runners and self._outstanding < self._high_watermark:
            runner_id = self._idle_runners.popleft()
            actor = self._runners[runner_id]
            weights = (
                self._latest_weights
                if self._runner_weight_versions[runner_id]
                != self._latest_weights.module_versions
                else None
            )
            ref = actor.collect_episode.remote(
                weights,
                explore=self._explore,
            )
            if weights is not None:
                self._runner_weight_versions[runner_id] = FrozenVersions(
                    weights.module_versions
                )
            self._pending_samples[ref] = runner_id
            self._sample_calls_started += 1
            self._outstanding_high_watermark = max(
                self._outstanding_high_watermark,
                self._outstanding,
            )
        if self._idle_runners and self._outstanding >= self._high_watermark:
            self._enter_backpressure()

    def _finish_sample(self, ref: ray.ObjectRef) -> None:
        runner_id = self._pending_samples.pop(ref)
        try:
            result = ray.get(ref)
        except Exception as error:
            self._sample_failures += 1
            raise RolloutGroupError(
                f"episode collection failed on {runner_id!r}"
            ) from error
        if not isinstance(result, EpisodeRolloutResult):
            self._sample_failures += 1
            raise RolloutGroupError("rollout actor returned an invalid result")
        if result.episode.runner_id != runner_id:
            self._sample_failures += 1
            raise RolloutGroupError("rollout actor returned the wrong runner_id")
        expected_generation = self._runner_generations[runner_id]
        if result.episode.runner_generation != expected_generation:
            self._sample_failures += 1
            raise RolloutGroupError("rollout actor returned a stale generation")

        lag = self._policy_lag(result.episode)
        self._episodes_collected += 1
        self._env_steps += result.episode.env_steps
        self._agent_steps += result.episode.agent_steps
        self._episode_times_ms.append(result.metrics.episode_time_s * 1_000.0)
        self._policy_lags.append(lag)
        self._idle_runners.append(runner_id)
        try:
            commit_ref = self._replay_actor.commit_episode.remote(result.episode)
        except Exception as error:
            self._commit_failures += 1
            raise RolloutGroupError("could not submit episode commit") from error
        self._pending_commits[commit_ref] = _PendingCommit(
            runner_id=runner_id,
            result=result,
            policy_version_lag=lag,
        )

    def _finish_commit(self, ref: ray.ObjectRef) -> RolloutCompletion:
        pending = self._pending_commits.pop(ref)
        try:
            acknowledgement = ray.get(ref)
        except Exception as error:
            self._commit_failures += 1
            raise RolloutGroupError(
                f"episode commit failed for {pending.result.episode.episode_id!r}"
            ) from error
        if not isinstance(acknowledgement, CommitAck):
            self._commit_failures += 1
            raise RolloutGroupError("replay actor returned an invalid acknowledgement")
        if acknowledgement.committed:
            self._episodes_committed += 1
        if acknowledgement.duplicate:
            self._duplicate_commits += 1
        return RolloutCompletion(
            episode=pending.result.episode,
            metrics=pending.result.metrics,
            acknowledgement=acknowledgement,
            policy_version_lag=pending.policy_version_lag,
        )

    def _policy_lag(self, episode: EpisodeEnvelope) -> int:
        latest = self._latest_weights.module_versions
        if set(episode.behavior_versions) != set(latest):
            raise RolloutGroupError("episode behavior module IDs do not match weights")
        lags = [
            latest[module_id] - episode.behavior_versions[module_id]
            for module_id in latest
        ]
        if any(lag < 0 for lag in lags):
            raise RolloutGroupError(
                "episode behavior version is newer than publication"
            )
        return max(lags, default=0)

    def _enter_backpressure(self) -> None:
        if self._backpressure_started is None:
            self._backpressure_started = time.monotonic()
            self._backpressure_events += 1

    def _relieve_backpressure_if_due(self) -> None:
        if (
            self._backpressure_started is not None
            and self._outstanding <= self._low_watermark
        ):
            self._leave_backpressure()

    def _leave_backpressure(self) -> None:
        if self._backpressure_started is None:
            return
        self._backpressure_s += time.monotonic() - self._backpressure_started
        self._backpressure_started = None

    @property
    def _outstanding(self) -> int:
        return len(self._pending_samples) + len(self._pending_commits)

    def _require_running(self) -> None:
        if self._state is not RolloutGroupState.RUNNING:
            raise RolloutGroupError("rollout group is not running")

    def _require_pollable(self) -> None:
        if self._state not in {
            RolloutGroupState.RUNNING,
            RolloutGroupState.PAUSED,
        }:
            raise RolloutGroupError("rollout group is not active")

    def _require_not_stopped(self) -> None:
        if self._state is RolloutGroupState.STOPPED:
            raise RolloutGroupError("rollout group is stopped")

    @staticmethod
    def _percentile(values: deque[float] | deque[int], percentile: int) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
