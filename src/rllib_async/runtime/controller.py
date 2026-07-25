"""Tune-compatible controller for one end-to-end asynchronous SAC member."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from enum import Enum
from typing import Any

import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory

from rllib_async.protocols import FlatEpisodeCodec, ReplayStats, WeightsDescriptor
from rllib_async.replay import ReplayActor
from rllib_async.rollout import AsyncRolloutGroup
from rllib_async.runtime.config import AsyncSACRuntimeConfig
from rllib_async.runtime.evaluation import AsyncEvaluationGroup
from rllib_async.runtime.learner_host import (
    LearnerHostActor,
    LearnerHostStats,
    LearnerHostTick,
)


class RuntimeState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class SingleMemberAsyncSAC:
    """Compose rollout, replay, learner, evaluation, and lifecycle explicitly."""

    def __init__(
        self,
        sac_config: SACConfig,
        runtime_config: AsyncSACRuntimeConfig,
    ) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before SingleMemberAsyncSAC")
        if not isinstance(sac_config, SACConfig):
            raise TypeError("sac_config must be an SACConfig")
        if not isinstance(runtime_config, AsyncSACRuntimeConfig):
            raise TypeError("runtime_config must be an AsyncSACRuntimeConfig")
        if sac_config.num_learners != 0:
            raise ValueError("single-member runtime requires one local RLlib learner")
        if float(sac_config.num_gpus_per_learner) != (
            runtime_config.num_gpus_per_learner
        ):
            raise ValueError("SAC and runtime num_gpus_per_learner settings must match")

        self._sac_config = sac_config.copy(copy_frozen=False)
        self._config = runtime_config
        self._target_training_intensity = self._resolve_training_intensity(
            self._sac_config.training_intensity,
            batch_size=self._config.batch_size,
        )
        self._codec = FlatEpisodeCodec()
        self._state = RuntimeState.CREATED
        self._replay_actor: Any | None = None
        self._learner_actor: Any | None = None
        self._rollout_group: AsyncRolloutGroup | None = None
        self._evaluation_group: AsyncEvaluationGroup | None = None
        self._pending_learner_tick: ray.ObjectRef | None = None
        self._latest_weights: WeightsDescriptor | None = None
        self._started_at: float | None = None
        self._pump_iterations = 0
        self._reports = 0
        self._pending_rpc_high_watermark = 0
        self._learner_updates_completed = 0
        self._last_report_env_steps = 0
        self._last_report_episodes = 0
        self._next_evaluation_env_steps = 0

        spaces = self._resolve_spaces(self._sac_config)
        try:
            self._replay_actor = ReplayActor.options(
                num_cpus=self._config.num_cpus_per_replay,
            ).remote(
                self._codec,
                capacity_transitions=self._config.replay_capacity_transitions,
                capacity_bytes=self._config.replay_capacity_bytes,
                journal_capacity=self._config.replay_journal_capacity,
            )
            self._learner_actor = LearnerHostActor.options(
                num_cpus=self._config.num_cpus_per_learner,
                num_gpus=self._config.num_gpus_per_learner,
            ).remote(
                self._sac_config,
                spaces,
                self._replay_actor,
                self._codec,
                member_id=self._config.member_id,
                publication_interval_updates=(
                    self._config.publication_interval_updates
                ),
                batch_size=self._config.batch_size,
                batch_queue_capacity=self._config.batch_queue_capacity,
                batch_seed=self._config.seed,
                replay_sync_max_bytes=self._config.replay_sync_max_bytes,
            )
            initial_weights = ray.get(
                self._learner_actor.get_published_weights.remote()
            )
            if not isinstance(initial_weights, WeightsDescriptor):
                raise RuntimeError("learner host returned invalid initial weights")
            self._latest_weights = initial_weights
            self._rollout_group = AsyncRolloutGroup(
                self._sac_config,
                self._codec,
                self._replay_actor,
                member_id=self._config.member_id,
                initial_weights=initial_weights,
                runner_count=self._config.runner_count,
                max_episode_steps=self._config.max_episode_steps,
                pending_commit_high_watermark=(
                    self._config.pending_commit_high_watermark
                ),
                pending_commit_low_watermark=(
                    self._config.pending_commit_low_watermark
                ),
                num_cpus_per_runner=self._config.num_cpus_per_runner,
            )
            if self._config.evaluation_num_episodes:
                self._evaluation_group = AsyncEvaluationGroup(
                    self._sac_config,
                    self._codec,
                    member_id=self._config.member_id,
                    initial_weights=initial_weights,
                    episode_count=self._config.evaluation_num_episodes,
                    max_episode_steps=self._config.max_episode_steps,
                    num_cpus_per_runner=(self._config.num_cpus_per_evaluation_runner),
                )
        except Exception:
            self._kill_components()
            self._state = RuntimeState.FAILED
            raise

    @property
    def state(self) -> RuntimeState:
        return self._state

    def start(self) -> None:
        if self._state is RuntimeState.RUNNING:
            return
        if self._state is RuntimeState.PAUSED:
            self.resume()
            return
        if self._state is not RuntimeState.CREATED:
            raise RuntimeError(f"cannot start runtime in state {self._state.value!r}")
        assert self._learner_actor is not None
        assert self._rollout_group is not None
        assert self._latest_weights is not None
        ray.get(self._learner_actor.start.remote())
        self._rollout_group.start()
        if self._evaluation_group is not None:
            self._evaluation_group.start_round(self._latest_weights)
            self._next_evaluation_env_steps = self._config.evaluation_interval_env_steps
        self._started_at = time.monotonic()
        self._state = RuntimeState.RUNNING
        self._schedule_learner_tick()

    def pump_once(self, *, timeout_s: float | None = None) -> None:
        """Advance each asynchronous layer without introducing a barrier."""

        self._require_state(RuntimeState.RUNNING)
        timeout_s = (
            self._config.event_poll_timeout_s if timeout_s is None else timeout_s
        )
        try:
            assert self._rollout_group is not None
            self._rollout_group.poll(
                timeout_s=timeout_s,
                max_events=self._config.pending_commit_high_watermark,
            )
            self._poll_learner_tick()
            if self._evaluation_group is not None:
                self._evaluation_group.poll(
                    timeout_s=0.0,
                    max_events=self._config.evaluation_num_episodes,
                )
            self._maybe_start_evaluation()
            self._schedule_learner_tick()
            self._pump_iterations += 1
            self._record_pending_high_watermark()
        except Exception:
            self._state = RuntimeState.FAILED
            raise

    def run_for(self, duration_s: float | None = None) -> dict[str, Any]:
        """Pump until the next Tune reporting boundary and return metrics."""

        self._require_state(RuntimeState.RUNNING)
        duration_s = (
            self._config.report_interval_s if duration_s is None else duration_s
        )
        if (
            not isinstance(duration_s, int | float)
            or isinstance(duration_s, bool)
            or not math.isfinite(duration_s)
            or duration_s <= 0
        ):
            raise ValueError("duration_s must be finite and positive")
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.pump_once(
                timeout_s=min(
                    self._config.event_poll_timeout_s,
                    max(deadline - time.monotonic(), 0.0),
                )
            )
        return self.get_report()

    def get_report(self) -> dict[str, Any]:
        self._require_not_stopped()
        self._poll_learner_tick()
        assert self._rollout_group is not None
        assert self._replay_actor is not None
        assert self._learner_actor is not None

        rollout = self._rollout_group.get_stats()
        replay = ray.get(self._replay_actor.get_stats.remote())
        learner = ray.get(self._learner_actor.get_stats.remote())
        if not isinstance(replay, ReplayStats):
            raise RuntimeError("replay actor returned invalid stats")
        if not isinstance(learner, LearnerHostStats):
            raise RuntimeError("learner actor returned invalid stats")
        evaluation = (
            self._evaluation_group.get_stats()
            if self._evaluation_group is not None
            else None
        )

        env_steps_this_iter = rollout.env_steps - self._last_report_env_steps
        episodes_this_iter = rollout.episodes_collected - self._last_report_episodes
        self._last_report_env_steps = rollout.env_steps
        self._last_report_episodes = rollout.episodes_collected
        self._reports += 1

        learner_values = asdict(learner)
        fast_replay = learner_values.pop("fast_replay")
        batching = learner_values.pop("batch_producer")
        learner_values["last_learner_metrics"] = dict(learner.last_learner_metrics)
        sampled_steps = self._sampled_steps(rollout)
        learner_values["update_to_data_ratio"] = (
            learner.learner_updates / sampled_steps if sampled_steps else 0.0
        )
        learner_values["training_intensity"] = (
            learner.learner_updates * self._config.batch_size / sampled_steps
            if sampled_steps
            else 0.0
        )
        learner_values["target_training_intensity"] = self._target_training_intensity
        learner_values["effective_training_intensity"] = (
            self._effective_training_intensity()
        )
        result: dict[str, Any] = {
            "timesteps_this_iter": env_steps_this_iter,
            "episodes_this_iter": episodes_this_iter,
            "episode_reward_mean": (
                evaluation.latest_return_mean if evaluation is not None else math.nan
            ),
            "controller": self._controller_metrics(),
            "rollout": self._metric_tree(asdict(rollout)),
            "authoritative_replay": self._metric_tree(asdict(replay)),
            "fast_replay": self._metric_tree(fast_replay),
            "batching": self._metric_tree(batching),
            "learner": self._metric_tree(learner_values),
            "evaluation": (
                self._metric_tree(asdict(evaluation))
                if evaluation is not None
                else {"enabled": False}
            ),
        }
        return result

    def pause(self, *, timeout_s: float | None = None) -> None:
        if self._state is RuntimeState.PAUSED:
            return
        self._require_state(RuntimeState.RUNNING)
        timeout_s = self._config.shutdown_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        assert self._rollout_group is not None
        assert self._learner_actor is not None
        try:
            self._rollout_group.pause()
            self._finish_learner_tick(timeout_s=self._remaining(deadline))
            ray.get(
                self._learner_actor.pause.remote(timeout_s=self._remaining(deadline)),
                timeout=self._remaining(deadline),
            )
            self._state = RuntimeState.PAUSED
        except Exception:
            self._state = RuntimeState.FAILED
            raise

    def resume(self) -> None:
        if self._state is RuntimeState.RUNNING:
            return
        self._require_state(RuntimeState.PAUSED)
        assert self._learner_actor is not None
        assert self._rollout_group is not None
        ray.get(self._learner_actor.resume.remote())
        self._rollout_group.resume()
        self._state = RuntimeState.RUNNING
        self._schedule_learner_tick()

    def drain(self, *, timeout_s: float | None = None) -> None:
        """Drain rollout/evaluation RPCs and the learner's queued batches."""

        if self._state not in {RuntimeState.RUNNING, RuntimeState.PAUSED}:
            raise RuntimeError(f"cannot drain runtime in state {self._state.value!r}")
        timeout_s = self._config.shutdown_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        assert self._rollout_group is not None
        assert self._learner_actor is not None
        self._state = RuntimeState.DRAINING
        try:
            self._rollout_group.drain(timeout_s=self._remaining(deadline))
            self._finish_learner_tick(timeout_s=self._remaining(deadline))
            if self._evaluation_group is not None:
                self._evaluation_group.drain(timeout_s=self._remaining(deadline))
            rollout = self._rollout_group.get_stats()
            tick = ray.get(
                self._learner_actor.drain.remote(
                    sampled_env_steps=rollout.env_steps,
                    sampled_agent_steps=rollout.agent_steps,
                    timeout_s=self._remaining(deadline),
                ),
                timeout=self._remaining(deadline),
            )
            self._accept_learner_tick(tick)
            self._state = RuntimeState.PAUSED
        except Exception:
            self._state = RuntimeState.FAILED
            raise

    def stop(
        self,
        *,
        graceful: bool = True,
        timeout_s: float | None = None,
    ) -> None:
        if self._state is RuntimeState.STOPPED:
            return
        timeout_s = self._config.shutdown_timeout_s if timeout_s is None else timeout_s
        shutdown_error: BaseException | None = None
        try:
            if graceful and self._state in {
                RuntimeState.RUNNING,
                RuntimeState.PAUSED,
            }:
                self.drain(timeout_s=timeout_s)
        except BaseException as error:
            shutdown_error = error
        finally:
            if self._pending_learner_tick is not None:
                ray.cancel(self._pending_learner_tick)
                self._pending_learner_tick = None
            if self._evaluation_group is not None:
                self._evaluation_group.stop()
            if self._rollout_group is not None:
                self._rollout_group.stop()
            if self._learner_actor is not None:
                try:
                    ray.get(
                        self._learner_actor.stop.remote(timeout_s=timeout_s),
                        timeout=timeout_s,
                    )
                except BaseException as error:
                    if shutdown_error is None:
                        shutdown_error = error
                ray.kill(self._learner_actor, no_restart=True)
            if self._replay_actor is not None:
                ray.kill(self._replay_actor, no_restart=True)
            self._state = RuntimeState.STOPPED
        if shutdown_error is not None:
            raise RuntimeError(
                f"graceful runtime shutdown failed: {shutdown_error}"
            ) from shutdown_error

    def _schedule_learner_tick(self) -> None:
        if (
            self._state is not RuntimeState.RUNNING
            or self._pending_learner_tick is not None
        ):
            return
        assert self._learner_actor is not None
        assert self._rollout_group is not None
        rollout = self._rollout_group.get_stats()
        remaining_updates = self._remaining_learner_update_budget(rollout)
        if remaining_updates == 0:
            return
        self._pending_learner_tick = self._learner_actor.tick.remote(
            sampled_env_steps=rollout.env_steps,
            sampled_agent_steps=rollout.agent_steps,
            max_updates=min(
                self._config.learner_updates_per_tick,
                remaining_updates,
            ),
        )

    def _poll_learner_tick(self) -> None:
        if self._pending_learner_tick is None:
            return
        ready, _ = ray.wait(
            [self._pending_learner_tick],
            num_returns=1,
            timeout=0,
        )
        if ready:
            self._finish_learner_tick()

    def _finish_learner_tick(self, *, timeout_s: float | None = None) -> None:
        if self._pending_learner_tick is None:
            return
        ref = self._pending_learner_tick
        tick = ray.get(ref, timeout=timeout_s)
        self._pending_learner_tick = None
        self._accept_learner_tick(tick)

    def _accept_learner_tick(self, tick: object) -> None:
        if not isinstance(tick, LearnerHostTick):
            raise RuntimeError("learner actor returned an invalid tick result")
        if (
            not isinstance(tick.updates_performed, int)
            or isinstance(tick.updates_performed, bool)
            or tick.updates_performed < 0
        ):
            raise RuntimeError("learner actor returned an invalid update count")
        self._learner_updates_completed += tick.updates_performed
        if tick.published_weights is None:
            return
        assert self._rollout_group is not None
        self._rollout_group.update_weights(tick.published_weights)
        self._latest_weights = tick.published_weights

    def _maybe_start_evaluation(self) -> None:
        if self._evaluation_group is None or self._evaluation_group.round_in_progress:
            return
        assert self._rollout_group is not None
        assert self._latest_weights is not None
        env_steps = self._rollout_group.get_stats().env_steps
        if env_steps < self._next_evaluation_env_steps:
            return
        self._evaluation_group.start_round(self._latest_weights)
        self._next_evaluation_env_steps = (
            env_steps + self._config.evaluation_interval_env_steps
        )

    def _record_pending_high_watermark(self) -> None:
        assert self._rollout_group is not None
        rollout = self._rollout_group.get_stats()
        evaluation_pending = (
            self._evaluation_group.get_stats().pending_calls
            if self._evaluation_group is not None
            else 0
        )
        pending = (
            rollout.pending_sample_calls
            + rollout.pending_episode_commits
            + evaluation_pending
            + int(self._pending_learner_tick is not None)
        )
        self._pending_rpc_high_watermark = max(
            self._pending_rpc_high_watermark,
            pending,
        )

    def _controller_metrics(self) -> dict[str, Any]:
        assert self._rollout_group is not None
        rollout = self._rollout_group.get_stats()
        evaluation_pending = (
            self._evaluation_group.get_stats().pending_calls
            if self._evaluation_group is not None
            else 0
        )
        pending = (
            rollout.pending_sample_calls
            + rollout.pending_episode_commits
            + evaluation_pending
            + int(self._pending_learner_tick is not None)
        )
        bound = (
            self._config.pending_commit_high_watermark
            + self._config.evaluation_num_episodes
            + 1
        )
        learner_update_budget = self._learner_update_budget(rollout)
        return {
            "state": self._state.value,
            "pump_iterations": self._pump_iterations,
            "reports": self._reports,
            "pending_rpcs": pending,
            "pending_rpc_bound": bound,
            "pending_rpc_high_watermark": self._pending_rpc_high_watermark,
            "learner_rpc_pending": int(self._pending_learner_tick is not None),
            "learner_updates_completed": self._learner_updates_completed,
            "learner_update_budget": learner_update_budget,
            "learner_update_budget_remaining": max(
                learner_update_budget - self._learner_updates_completed,
                0,
            ),
            "target_training_intensity": self._target_training_intensity,
            "effective_training_intensity": (self._effective_training_intensity()),
            "uptime_s": (
                time.monotonic() - self._started_at
                if self._started_at is not None
                else 0.0
            ),
        }

    def _kill_components(self) -> None:
        if self._evaluation_group is not None:
            self._evaluation_group.stop()
        if self._rollout_group is not None:
            self._rollout_group.stop()
        if self._learner_actor is not None:
            ray.kill(self._learner_actor, no_restart=True)
        if self._replay_actor is not None:
            ray.kill(self._replay_actor, no_restart=True)

    def _require_state(self, expected: RuntimeState) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"runtime must be {expected.value!r}, not {self._state.value!r}"
            )

    def _require_not_stopped(self) -> None:
        if self._state is RuntimeState.STOPPED:
            raise RuntimeError("runtime is stopped")

    def _remaining_learner_update_budget(self, rollout: object) -> int:
        return max(
            self._learner_update_budget(rollout) - self._learner_updates_completed,
            0,
        )

    def _learner_update_budget(self, rollout: object) -> int:
        sampled_steps = self._sampled_steps(rollout)
        learning_starts = int(self._sac_config.num_steps_sampled_before_learning_starts)
        if sampled_steps < learning_starts:
            return 0

        store_steps, updates_per_round = self._training_round_robin_weights()
        completed_store_rounds = sampled_steps // store_steps
        if learning_starts == 0:
            eligible_store_rounds = completed_store_rounds
        else:
            first_training_round = math.ceil(learning_starts / store_steps)
            eligible_store_rounds = max(
                completed_store_rounds - first_training_round + 1,
                0,
            )
        return eligible_store_rounds * updates_per_round

    def _sampled_steps(self, rollout: object) -> int:
        count_steps_by = self._sac_config.count_steps_by
        if count_steps_by == "env_steps":
            sampled_steps = rollout.env_steps
        elif count_steps_by == "agent_steps":
            sampled_steps = rollout.agent_steps
        else:
            raise RuntimeError(f"unsupported count_steps_by {count_steps_by!r}")
        return sampled_steps

    def _training_round_robin_weights(self) -> tuple[int, int]:
        update_ratio = self._target_training_intensity / self._config.batch_size
        if update_ratio < 1.0:
            return max(round(1.0 / update_ratio), 1), 1
        return 1, max(round(update_ratio), 1)

    def _effective_training_intensity(self) -> float:
        store_steps, updates_per_round = self._training_round_robin_weights()
        return updates_per_round * self._config.batch_size / store_steps

    @staticmethod
    def _resolve_training_intensity(
        value: object,
        *,
        batch_size: int,
    ) -> float:
        # RLlib treats None and zero as its natural round-robin intensity. This
        # step-scheduled runtime's natural equivalent is one learner update per
        # newly eligible sampled step.
        if value is None:
            return float(batch_size)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "SAC training_intensity must be finite and non-negative or None"
            )
        if value == 0:
            return float(batch_size)
        return float(value)

    @staticmethod
    def _resolve_spaces(
        config: SACConfig,
    ) -> Mapping[str, tuple[Any, Any]]:
        probe = SingleAgentEnvRunner(config=config, worker_index=0)
        try:
            return probe.get_spaces()
        finally:
            probe.stop()

    @classmethod
    def _metric_tree(cls, value: object) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {str(key): cls._metric_tree(nested) for key, nested in value.items()}
        if isinstance(value, list | tuple):
            return [cls._metric_tree(nested) for nested in value]
        return value

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(deadline - time.monotonic(), 0.0)


class AsyncSACTrainable(Trainable):
    """Thin Tune controller around `SingleMemberAsyncSAC`."""

    @classmethod
    def default_resource_request(
        cls,
        config: dict[str, Any],
    ) -> PlacementGroupFactory:
        sac_config, runtime = cls._parse_config(config)
        del sac_config
        bundles: list[dict[str, float]] = [{"CPU": 1.0}]

        def add_bundle(*, cpu: float = 0.0, gpu: float = 0.0) -> None:
            resources: dict[str, float] = {}
            if cpu:
                resources["CPU"] = cpu
            if gpu:
                resources["GPU"] = gpu
            if resources:
                bundles.append(resources)

        add_bundle(cpu=runtime.num_cpus_per_replay)
        add_bundle(
            cpu=runtime.num_cpus_per_learner,
            gpu=runtime.num_gpus_per_learner,
        )
        for _ in range(runtime.runner_count):
            add_bundle(cpu=runtime.num_cpus_per_runner)
        for _ in range(runtime.evaluation_num_episodes):
            add_bundle(cpu=runtime.num_cpus_per_evaluation_runner)
        return PlacementGroupFactory(bundles, strategy="PACK")

    def setup(self, config: dict[str, Any]) -> None:
        sac_config, runtime_config = self._parse_config(config)
        self._runtime = SingleMemberAsyncSAC(sac_config, runtime_config)
        self._runtime.start()

    def step(self) -> dict[str, Any]:
        return self._runtime.run_for()

    def cleanup(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.stop()

    @staticmethod
    def _parse_config(
        config: Mapping[str, Any],
    ) -> tuple[SACConfig, AsyncSACRuntimeConfig]:
        if not isinstance(config, Mapping):
            raise TypeError("Trainable config must be a mapping")
        unknown = set(config) - {"sac_config", "runtime"}
        if unknown:
            raise ValueError(f"unknown Trainable settings {sorted(unknown)!r}")
        sac_config = config.get("sac_config")
        if not isinstance(sac_config, SACConfig):
            raise TypeError("config['sac_config'] must be an SACConfig")
        runtime = AsyncSACRuntimeConfig.from_mapping(
            config.get("runtime"),
            sac_config=sac_config,
        )
        return sac_config, runtime
