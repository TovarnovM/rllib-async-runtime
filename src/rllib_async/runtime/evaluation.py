"""Frozen-weight evaluation actors that never write to training replay."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from rllib_async.protocols import (
    FlatEpisodeCodec,
    FrozenVersions,
    WeightsDescriptor,
)
from rllib_async.rollout import EpisodeRolloutActor, EpisodeRolloutResult
from rllib_async.rollout.episode_runner import _accept_weight_publication


class EvaluationGroupError(RuntimeError):
    """The evaluation group cannot preserve frozen-round semantics."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One complete evaluation round under one immutable publication."""

    round_id: int
    module_versions: FrozenVersions
    learner_updates: int
    episode_returns: tuple[float, ...]
    env_steps: int

    @property
    def return_mean(self) -> float:
        return float(np.mean(self.episode_returns))

    @property
    def return_min(self) -> float:
        return min(self.episode_returns)

    @property
    def return_max(self) -> float:
        return max(self.episode_returns)


@dataclass(frozen=True, slots=True)
class EvaluationGroupStats:
    """Bounded evaluation activity and the latest completed result."""

    runner_count: int
    rounds_started: int
    rounds_completed: int
    episodes_completed: int
    failures: int
    pending_calls: int
    pending_high_watermark: int
    round_in_progress: bool
    latest_round_id: int
    latest_learner_updates: int
    latest_module_version: int
    latest_return_mean: float
    latest_return_min: float
    latest_return_max: float
    latest_env_steps: int


class AsyncEvaluationGroup:
    """Run one replay-isolated episode per actor for each frozen round."""

    def __init__(
        self,
        config: AlgorithmConfig,
        codec: FlatEpisodeCodec,
        *,
        member_id: str,
        initial_weights: WeightsDescriptor,
        episode_count: int,
        max_episode_steps: int,
        num_cpus_per_runner: float = 1.0,
    ) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before AsyncEvaluationGroup")
        if not isinstance(config, AlgorithmConfig):
            raise TypeError("config must be an AlgorithmConfig")
        if not isinstance(codec, FlatEpisodeCodec):
            raise TypeError("AsyncEvaluationGroup requires FlatEpisodeCodec")
        if (
            not isinstance(episode_count, int)
            or isinstance(episode_count, bool)
            or not 1 <= episode_count <= 16
        ):
            raise ValueError("episode_count must be between 1 and 16")
        if (
            not isinstance(max_episode_steps, int)
            or isinstance(max_episode_steps, bool)
            or max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be positive")
        if (
            not isinstance(num_cpus_per_runner, int | float)
            or isinstance(num_cpus_per_runner, bool)
            or not math.isfinite(num_cpus_per_runner)
            or num_cpus_per_runner < 0
        ):
            raise ValueError("num_cpus_per_runner must be finite and non-negative")
        _accept_weight_publication(
            None,
            initial_weights,
            member_id=member_id,
        )

        self._config = config.copy(copy_frozen=False)
        self._codec = codec
        self._member_id = member_id
        self._episode_count = episode_count
        self._max_episode_steps = max_episode_steps
        self._num_cpus_per_runner = float(num_cpus_per_runner)
        self._latest_weights = copy.deepcopy(initial_weights)
        self._actor_versions: dict[str, FrozenVersions] = {}
        self._actors: dict[str, Any] = {}
        self._pending: dict[ray.ObjectRef, str] = {}
        self._active_weights: WeightsDescriptor | None = None
        self._active_returns: list[float] = []
        self._active_env_steps = 0
        self._rounds_started = 0
        self._rounds_completed = 0
        self._episodes_completed = 0
        self._failures = 0
        self._pending_high_watermark = 0
        self._latest_result: EvaluationResult | None = None
        self._stopped = False

        try:
            for index in range(episode_count):
                runner_id = f"evaluation-{index}"
                self._actors[runner_id] = EpisodeRolloutActor.options(
                    num_cpus=self._num_cpus_per_runner,
                ).remote(
                    self._config,
                    self._codec,
                    member_id=self._member_id,
                    runner_id=runner_id,
                    runner_generation=0,
                    max_episode_steps=self._max_episode_steps,
                    initial_weights=self._latest_weights,
                    worker_index=10_000 + index,
                )
                self._actor_versions[runner_id] = FrozenVersions(
                    self._latest_weights.module_versions
                )
        except Exception:
            for actor in self._actors.values():
                ray.kill(actor, no_restart=True)
            raise

    @property
    def round_in_progress(self) -> bool:
        return bool(self._pending)

    @property
    def latest_result(self) -> EvaluationResult | None:
        return self._latest_result

    def start_round(self, weights: WeightsDescriptor) -> int:
        """Start a bounded round; every actor receives the same publication."""

        self._require_open()
        if self._pending:
            raise EvaluationGroupError("an evaluation round is already in progress")
        accepted = _accept_weight_publication(
            self._latest_weights,
            weights,
            member_id=self._member_id,
        )
        current_versions = self._latest_weights.module_versions
        candidate_versions = weights.module_versions
        if not accepted and any(
            candidate_versions[module_id] < current_versions[module_id]
            for module_id in current_versions
        ):
            raise EvaluationGroupError("cannot evaluate a stale weight publication")
        if accepted:
            self._latest_weights = copy.deepcopy(weights)
        else:
            self._latest_weights = copy.deepcopy(self._latest_weights)

        frozen = copy.deepcopy(self._latest_weights)
        self._active_weights = frozen
        self._active_returns = []
        self._active_env_steps = 0
        round_id = self._rounds_started
        self._rounds_started += 1
        for runner_id, actor in self._actors.items():
            actor_weights = (
                frozen
                if self._actor_versions[runner_id] != frozen.module_versions
                else None
            )
            ref = actor.collect_episode.remote(actor_weights, explore=False)
            if actor_weights is not None:
                self._actor_versions[runner_id] = FrozenVersions(frozen.module_versions)
            self._pending[ref] = runner_id
        self._pending_high_watermark = max(
            self._pending_high_watermark,
            len(self._pending),
        )
        return round_id

    def poll(
        self,
        *,
        timeout_s: float | None = 0.0,
        max_events: int = 1,
    ) -> EvaluationResult | None:
        """Advance ready evaluation RPCs and return a newly completed round."""

        self._require_open()
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
            raise ValueError("max_events must be positive")
        if not self._pending:
            return None

        refs = list(self._pending)
        ready, remaining = ray.wait(refs, num_returns=1, timeout=timeout_s)
        if ready and max_events > 1 and remaining:
            additionally_ready, _ = ray.wait(
                remaining,
                num_returns=min(max_events - 1, len(remaining)),
                timeout=0,
            )
            ready.extend(additionally_ready)
        for ref in ready:
            runner_id = self._pending.pop(ref)
            try:
                result = ray.get(ref)
            except Exception as error:
                self._failures += 1
                raise EvaluationGroupError(
                    f"evaluation failed on {runner_id!r}"
                ) from error
            self._accept_result(runner_id, result)

        if self._pending:
            return None
        assert self._active_weights is not None
        result = EvaluationResult(
            round_id=self._rounds_completed,
            module_versions=FrozenVersions(self._active_weights.module_versions),
            learner_updates=self._active_weights.learner_updates,
            episode_returns=tuple(self._active_returns),
            env_steps=self._active_env_steps,
        )
        self._rounds_completed += 1
        self._latest_result = result
        self._active_weights = None
        return result

    def drain(self, *, timeout_s: float | None = None) -> EvaluationResult | None:
        deadline = self._deadline(timeout_s)
        completed: EvaluationResult | None = None
        while self._pending:
            remaining = self._remaining(deadline)
            if remaining == 0:
                raise TimeoutError("timed out draining evaluation actors")
            result = self.poll(
                timeout_s=min(remaining, 0.05) if remaining is not None else 0.05,
                max_events=len(self._pending),
            )
            if result is not None:
                completed = result
        return completed

    def get_stats(self) -> EvaluationGroupStats:
        latest = self._latest_result
        return EvaluationGroupStats(
            runner_count=self._episode_count,
            rounds_started=self._rounds_started,
            rounds_completed=self._rounds_completed,
            episodes_completed=self._episodes_completed,
            failures=self._failures,
            pending_calls=len(self._pending),
            pending_high_watermark=self._pending_high_watermark,
            round_in_progress=bool(self._pending),
            latest_round_id=latest.round_id if latest is not None else -1,
            latest_learner_updates=(
                latest.learner_updates if latest is not None else 0
            ),
            latest_module_version=(
                max(latest.module_versions.values(), default=0)
                if latest is not None
                else 0
            ),
            latest_return_mean=(latest.return_mean if latest is not None else math.nan),
            latest_return_min=(latest.return_min if latest is not None else math.nan),
            latest_return_max=(latest.return_max if latest is not None else math.nan),
            latest_env_steps=latest.env_steps if latest is not None else 0,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        for ref in tuple(self._pending):
            ray.cancel(ref)
        for actor in self._actors.values():
            ray.kill(actor, no_restart=True)
        self._pending.clear()
        self._actors.clear()
        self._stopped = True

    def _accept_result(
        self,
        runner_id: str,
        result: object,
    ) -> None:
        if not isinstance(result, EpisodeRolloutResult):
            self._failures += 1
            raise EvaluationGroupError("evaluation actor returned an invalid result")
        if result.episode.runner_id != runner_id:
            self._failures += 1
            raise EvaluationGroupError("evaluation actor returned the wrong runner_id")
        if self._active_weights is None:
            self._failures += 1
            raise EvaluationGroupError("evaluation returned outside an active round")
        if result.episode.behavior_versions != self._active_weights.module_versions:
            self._failures += 1
            raise EvaluationGroupError("evaluation mixed weight publications")
        self._active_returns.append(result.metrics.episode_return)
        self._active_env_steps += result.metrics.env_steps
        self._episodes_completed += 1

    def _require_open(self) -> None:
        if self._stopped:
            raise EvaluationGroupError("evaluation group is stopped")

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
