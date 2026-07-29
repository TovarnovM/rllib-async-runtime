"""Tune-compatible controller for one end-to-end asynchronous SAC member."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.tune import Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory

from rllib_async.protocols import (
    FlatEpisodeCodec,
    ReplayCursor,
    ReplayStats,
    WeightsDescriptor,
)
from rllib_async.replay import ReplayActor
from rllib_async.replay.checkpoint import write_replay_checkpoint
from rllib_async.rollout import AsyncRolloutGroup
from rllib_async.runtime.checkpoint import (
    RUNTIME_CHECKPOINT_FILENAME,
    RUNTIME_CHECKPOINT_STATE_VERSION,
    RUNTIME_REPLAY_FILENAME,
    RuntimeCheckpoint,
    RuntimeCheckpointState,
    read_runtime_checkpoint_bundle,
    read_runtime_member_checkpoint,
    write_runtime_checkpoint,
    write_runtime_member_checkpoint,
)
from rllib_async.runtime.config import AsyncSACRuntimeConfig, SharedReplayDescriptor
from rllib_async.runtime.evaluation import AsyncEvaluationGroup
from rllib_async.runtime.learner_host import (
    LearnerHostActor,
    LearnerHostCheckpoint,
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
        *,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        replay_actor: Any | None = None,
        member_checkpoint_state: RuntimeCheckpointState | None = None,
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
        if checkpoint_dir is not None and replay_actor is not None:
            raise ValueError(
                "standalone checkpoint restore cannot replace an external replay"
            )
        if checkpoint_dir is not None and member_checkpoint_state is not None:
            raise ValueError("runtime restore accepts only one checkpoint source")
        if member_checkpoint_state is not None and replay_actor is None:
            raise ValueError("population member restore requires an external replay")

        self._sac_config = sac_config.copy(copy_frozen=False)
        self._config = runtime_config
        self._owns_replay_actor = replay_actor is None
        restore_started = time.monotonic()
        checkpoint_state: RuntimeCheckpointState | None = None
        replay_checkpoint_state = None
        if checkpoint_dir is not None:
            checkpoint_state, replay_checkpoint_state = read_runtime_checkpoint_bundle(
                checkpoint_dir
            )
            if checkpoint_state.member_id != runtime_config.member_id:
                raise ValueError("runtime checkpoint member_id does not match")
            if checkpoint_state.runtime_config != asdict(runtime_config):
                raise ValueError("runtime checkpoint configuration does not match")
        elif member_checkpoint_state is not None:
            checkpoint_state = member_checkpoint_state
            if checkpoint_state.member_id != runtime_config.member_id:
                raise ValueError("runtime checkpoint member_id does not match")
            if checkpoint_state.runtime_config != asdict(runtime_config):
                raise ValueError("runtime checkpoint configuration does not match")
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
        self._checkpoint_sequence = 0
        self._restore_count = 0
        self._last_checkpoint_duration_s = 0.0
        self._last_restore_duration_s = 0.0
        self._restored = checkpoint_state is not None
        if checkpoint_state is not None:
            self._restore_controller_checkpoint_state(checkpoint_state)

        spaces = self._resolve_spaces(self._sac_config)
        try:
            if replay_actor is None:
                self._replay_actor = ReplayActor.options(
                    num_cpus=self._config.num_cpus_per_replay,
                ).remote(
                    self._codec,
                    capacity_transitions=self._config.replay_capacity_transitions,
                    capacity_bytes=self._config.replay_capacity_bytes,
                    journal_capacity=self._config.replay_journal_capacity,
                )
            else:
                self._replay_actor = replay_actor
            if replay_checkpoint_state is not None:
                restored_replay = ray.get(
                    self._replay_actor.load_checkpoint_state.remote(
                        replay_checkpoint_state
                    )
                )
                if (
                    not isinstance(restored_replay, ReplayStats)
                    or restored_replay.cursor != checkpoint_state.replay_cursor
                ):
                    raise RuntimeError(
                        "replay actor did not restore the checkpoint cursor"
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
                checkpoint_state=(
                    checkpoint_state.learner if checkpoint_state is not None else None
                ),
                allow_replay_ahead_on_restore=(
                    checkpoint_state is not None and not self._owns_replay_actor
                ),
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
                checkpoint_state=(
                    checkpoint_state.rollout if checkpoint_state is not None else None
                ),
            )
            if self._config.evaluation_num_episodes:
                if checkpoint_state is not None and checkpoint_state.evaluation is None:
                    raise ValueError("runtime checkpoint is missing evaluation state")
                self._evaluation_group = AsyncEvaluationGroup(
                    self._sac_config,
                    self._codec,
                    member_id=self._config.member_id,
                    initial_weights=initial_weights,
                    episode_count=self._config.evaluation_num_episodes,
                    max_episode_steps=self._config.max_episode_steps,
                    num_cpus_per_runner=(self._config.num_cpus_per_evaluation_runner),
                    checkpoint_state=(
                        checkpoint_state.evaluation
                        if checkpoint_state is not None
                        else None
                    ),
                )
            elif (
                checkpoint_state is not None and checkpoint_state.evaluation is not None
            ):
                raise ValueError(
                    "runtime checkpoint unexpectedly contains evaluation state"
                )
            if checkpoint_state is not None:
                self._validate_restored_runtime_state(checkpoint_state)
                self._last_restore_duration_s = time.monotonic() - restore_started
        except Exception:
            self._kill_components()
            self._state = RuntimeState.FAILED
            raise

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def owns_replay_actor(self) -> bool:
        return self._owns_replay_actor

    @classmethod
    def from_checkpoint(
        cls,
        sac_config: SACConfig,
        runtime_config: AsyncSACRuntimeConfig,
        checkpoint_dir: str | os.PathLike[str],
    ) -> SingleMemberAsyncSAC:
        """Recreate every owned actor from one relocatable Tune checkpoint."""

        return cls(
            sac_config,
            runtime_config,
            checkpoint_dir=checkpoint_dir,
        )

    @classmethod
    def from_member_checkpoint(
        cls,
        sac_config: SACConfig,
        runtime_config: AsyncSACRuntimeConfig,
        checkpoint_dir: str | os.PathLike[str],
        *,
        replay_actor: Any,
    ) -> SingleMemberAsyncSAC:
        """Restore one population member against an existing shared replay."""

        return cls.from_member_checkpoint_state(
            sac_config,
            runtime_config,
            read_runtime_member_checkpoint(checkpoint_dir),
            replay_actor=replay_actor,
        )

    @classmethod
    def from_member_checkpoint_state(
        cls,
        sac_config: SACConfig,
        runtime_config: AsyncSACRuntimeConfig,
        checkpoint_state: RuntimeCheckpointState,
        *,
        replay_actor: Any,
    ) -> SingleMemberAsyncSAC:
        """Restore one member state transferred independently of replay."""

        if not isinstance(checkpoint_state, RuntimeCheckpointState):
            raise TypeError("checkpoint_state must be RuntimeCheckpointState")
        return cls(
            sac_config,
            runtime_config,
            replay_actor=replay_actor,
            member_checkpoint_state=checkpoint_state,
        )

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
        self._started_at = time.monotonic()
        self._state = RuntimeState.RUNNING
        if self._evaluation_group is not None:
            if self._restored:
                self._maybe_start_evaluation()
            else:
                self._evaluation_group.start_round(self._latest_weights)
                self._next_evaluation_env_steps = (
                    self._config.evaluation_interval_env_steps
                )
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

    def get_report(
        self,
        *,
        include_authoritative_replay: bool = True,
    ) -> dict[str, Any]:
        self._require_not_stopped()
        self._poll_learner_tick()
        assert self._rollout_group is not None
        assert self._replay_actor is not None
        assert self._learner_actor is not None

        rollout = self._rollout_group.get_stats()
        replay = (
            ray.get(self._replay_actor.get_stats.remote())
            if include_authoritative_replay
            else None
        )
        learner = ray.get(self._learner_actor.get_stats.remote())
        if replay is not None and not isinstance(replay, ReplayStats):
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
        replay_values = asdict(replay) if replay is not None else None
        if replay_values is not None:
            for name in (
                "producer_episode_counts",
                "producer_transition_counts",
            ):
                replay_values[name] = dict(replay_values[name])
        for name in (
            "active_module_transition_counts",
            "active_producer_episode_counts",
            "active_producer_transition_counts",
        ):
            fast_replay[name] = dict(fast_replay[name])
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
        rollout_values = asdict(rollout)
        train_values = {
            name: rollout_values.pop(name)
            for name in (
                "episode_reward_mean",
                "episode_reward_min",
                "episode_reward_max",
                "episodes_in_window",
                "episodes_since_metric_reset",
            )
        }
        result: dict[str, Any] = {
            "timesteps_this_iter": env_steps_this_iter,
            "episodes_this_iter": episodes_this_iter,
            "episode_reward_mean": (
                evaluation.latest_return_mean if evaluation is not None else math.nan
            ),
            "controller": self._controller_metrics(),
            "train": self._metric_tree(train_values),
            "rollout": self._metric_tree(rollout_values),
            "fast_replay": self._metric_tree(fast_replay),
            "batching": self._metric_tree(batching),
            "learner": self._metric_tree(learner_values),
            "evaluation": (
                self._metric_tree(asdict(evaluation))
                if evaluation is not None
                else {"enabled": False}
            ),
        }
        if replay_values is not None:
            result["authoritative_replay"] = self._metric_tree(replay_values)
        return result

    def save_checkpoint(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        timeout_s: float | None = None,
    ) -> RuntimeCheckpoint:
        """Drain to an episode boundary and publish one coordinated checkpoint."""

        if not self._owns_replay_actor:
            raise RuntimeError(
                "externally owned replay must be checkpointed once by the "
                "population launcher"
            )
        if self._state not in {RuntimeState.RUNNING, RuntimeState.PAUSED}:
            raise RuntimeError(
                f"cannot checkpoint runtime in state {self._state.value!r}"
            )
        timeout_s = self._config.shutdown_timeout_s if timeout_s is None else timeout_s
        if (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        directory = Path(checkpoint_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
        published_files = (
            directory / RUNTIME_CHECKPOINT_FILENAME,
            directory / RUNTIME_REPLAY_FILENAME,
        )
        if any(path.exists() for path in published_files):
            raise FileExistsError(
                "checkpoint directory already contains a runtime checkpoint"
            )

        started = time.monotonic()
        deadline = started + timeout_s
        was_running = self._state is RuntimeState.RUNNING
        try:
            self.drain(timeout_s=self._remaining(deadline))
            assert self._replay_actor is not None
            assert self._learner_actor is not None
            assert self._rollout_group is not None

            replay_state = ray.get(
                self._replay_actor.get_checkpoint_state.remote(),
                timeout=self._remaining(deadline),
            )
            replay_checkpoint = write_replay_checkpoint(
                directory / RUNTIME_REPLAY_FILENAME,
                replay_state,
            )
            learner_checkpoint = ray.get(
                self._learner_actor.get_checkpoint.remote(),
                timeout=self._remaining(deadline),
            )
            if not isinstance(learner_checkpoint, LearnerHostCheckpoint):
                raise RuntimeError("learner actor returned invalid checkpoint")
            if learner_checkpoint.replay_cursor != replay_checkpoint.cursor:
                raise RuntimeError(
                    "learner and authoritative replay checkpoint cursors differ"
                )

            next_sequence = self._checkpoint_sequence + 1
            state = self._build_checkpoint_state(
                learner_checkpoint,
                replay_cursor=replay_checkpoint.cursor,
                checkpoint_sequence=next_sequence,
            )
            checkpoint = write_runtime_checkpoint(directory, state)
            self._checkpoint_sequence = next_sequence
            self._last_checkpoint_duration_s = time.monotonic() - started
            return checkpoint
        finally:
            if was_running and self._state is RuntimeState.PAUSED:
                self.resume()

    def save_member_checkpoint(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        timeout_s: float | None = None,
    ) -> RuntimeCheckpoint:
        """Publish one member cut while leaving shared replay externally owned."""

        if self._owns_replay_actor:
            raise RuntimeError(
                "standalone runtime checkpoints must include authoritative replay"
            )
        if self._state not in {RuntimeState.RUNNING, RuntimeState.PAUSED}:
            raise RuntimeError(
                f"cannot checkpoint runtime in state {self._state.value!r}"
            )
        timeout_s = self._config.shutdown_timeout_s if timeout_s is None else timeout_s
        if (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        directory = Path(checkpoint_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
        if (directory / RUNTIME_CHECKPOINT_FILENAME).exists():
            raise FileExistsError(
                "checkpoint directory already contains a member checkpoint"
            )

        started = time.monotonic()
        deadline = started + timeout_s
        was_running = self._state is RuntimeState.RUNNING
        try:
            self.drain(timeout_s=self._remaining(deadline))
            assert self._learner_actor is not None
            learner_checkpoint = ray.get(
                self._learner_actor.get_checkpoint.remote(),
                timeout=self._remaining(deadline),
            )
            if not isinstance(learner_checkpoint, LearnerHostCheckpoint):
                raise RuntimeError("learner actor returned invalid checkpoint")

            next_sequence = self._checkpoint_sequence + 1
            state = self._build_checkpoint_state(
                learner_checkpoint,
                replay_cursor=learner_checkpoint.replay_cursor,
                checkpoint_sequence=next_sequence,
            )
            checkpoint = write_runtime_member_checkpoint(directory, state)
            self._checkpoint_sequence = next_sequence
            self._last_checkpoint_duration_s = time.monotonic() - started
            return checkpoint
        finally:
            if was_running and self._state is RuntimeState.PAUSED:
                self.resume()

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
            remaining_updates = self._remaining_learner_update_budget(rollout)
            tick = ray.get(
                self._learner_actor.drain.remote(
                    sampled_env_steps=rollout.env_steps,
                    sampled_agent_steps=rollout.agent_steps,
                    max_updates=remaining_updates,
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
                if graceful:
                    try:
                        ray.get(
                            self._learner_actor.stop.remote(timeout_s=timeout_s),
                            timeout=timeout_s,
                        )
                    except BaseException as error:
                        if shutdown_error is None:
                            shutdown_error = error
                ray.kill(self._learner_actor, no_restart=True)
            if self._replay_actor is not None and self._owns_replay_actor:
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
            "member_id": self._config.member_id,
            "state": self._state.value,
            "owns_replay_actor": self._owns_replay_actor,
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
            "checkpoint_sequence": self._checkpoint_sequence,
            "restore_count": self._restore_count,
            "last_checkpoint_duration_s": self._last_checkpoint_duration_s,
            "last_restore_duration_s": self._last_restore_duration_s,
            "started_at_monotonic": self._started_at or 0.0,
            "reported_at_monotonic": time.monotonic(),
            "uptime_s": (
                time.monotonic() - self._started_at
                if self._started_at is not None
                else 0.0
            ),
        }

    def _controller_checkpoint_state(
        self,
        *,
        checkpoint_sequence: int,
    ) -> dict[str, Any]:
        return {
            "state_version": RUNTIME_CHECKPOINT_STATE_VERSION,
            "pump_iterations": self._pump_iterations,
            "reports": self._reports,
            "pending_rpc_high_watermark": self._pending_rpc_high_watermark,
            "learner_updates_completed": self._learner_updates_completed,
            "last_report_env_steps": self._last_report_env_steps,
            "last_report_episodes": self._last_report_episodes,
            "next_evaluation_env_steps": self._next_evaluation_env_steps,
            "checkpoint_sequence": checkpoint_sequence,
            "restore_count": self._restore_count,
        }

    def _build_checkpoint_state(
        self,
        learner_checkpoint: LearnerHostCheckpoint,
        *,
        replay_cursor: ReplayCursor,
        checkpoint_sequence: int,
    ) -> RuntimeCheckpointState:
        assert self._rollout_group is not None
        if learner_checkpoint.replay_cursor != replay_cursor:
            raise RuntimeError("learner checkpoint cursor changed during publication")
        return RuntimeCheckpointState(
            state_version=RUNTIME_CHECKPOINT_STATE_VERSION,
            member_id=self._config.member_id,
            runtime_config=asdict(self._config),
            replay_file=RUNTIME_REPLAY_FILENAME,
            replay_cursor=learner_checkpoint.replay_cursor,
            learner=learner_checkpoint.payload,
            rollout=self._rollout_group.get_checkpoint_state(),
            evaluation=(
                self._evaluation_group.get_checkpoint_state()
                if self._evaluation_group is not None
                else None
            ),
            controller=self._controller_checkpoint_state(
                checkpoint_sequence=checkpoint_sequence
            ),
        )

    def _restore_controller_checkpoint_state(
        self,
        checkpoint: RuntimeCheckpointState,
    ) -> None:
        state = checkpoint.controller
        if state.get("state_version") != RUNTIME_CHECKPOINT_STATE_VERSION:
            raise ValueError("unsupported controller checkpoint state version")
        self._pump_iterations = self._checkpoint_counter(
            state,
            "pump_iterations",
        )
        self._reports = self._checkpoint_counter(state, "reports")
        self._pending_rpc_high_watermark = self._checkpoint_counter(
            state,
            "pending_rpc_high_watermark",
        )
        self._learner_updates_completed = self._checkpoint_counter(
            state,
            "learner_updates_completed",
        )
        self._last_report_env_steps = self._checkpoint_counter(
            state,
            "last_report_env_steps",
        )
        self._last_report_episodes = self._checkpoint_counter(
            state,
            "last_report_episodes",
        )
        self._next_evaluation_env_steps = self._checkpoint_counter(
            state,
            "next_evaluation_env_steps",
        )
        self._checkpoint_sequence = self._checkpoint_counter(
            state,
            "checkpoint_sequence",
        )
        self._restore_count = self._checkpoint_counter(state, "restore_count") + 1

    def _validate_restored_runtime_state(
        self,
        checkpoint: RuntimeCheckpointState,
    ) -> None:
        assert self._learner_actor is not None
        assert self._rollout_group is not None
        assert self._replay_actor is not None
        assert self._latest_weights is not None
        learner = ray.get(self._learner_actor.get_stats.remote())
        replay = ray.get(self._replay_actor.get_stats.remote())
        rollout = self._rollout_group.get_stats()
        if not isinstance(learner, LearnerHostStats):
            raise RuntimeError("restored learner returned invalid stats")
        if not isinstance(replay, ReplayStats):
            raise RuntimeError("restored replay returned invalid stats")
        if self._owns_replay_actor:
            expected_replay_cursor = checkpoint.replay_cursor
        else:
            expected_replay_cursor = replay.cursor
            if (
                replay.cursor.store_generation
                != checkpoint.replay_cursor.store_generation
                or replay.cursor.mutation_seq < checkpoint.replay_cursor.mutation_seq
            ):
                raise RuntimeError(
                    "shared replay is older than or foreign to member checkpoint"
                )
        if learner.fast_replay.cursor != expected_replay_cursor:
            raise RuntimeError("restored FastReplay cursor does not match replay")
        if learner.fast_replay.active_cursor != expected_replay_cursor:
            raise RuntimeError("restored FastReplay index is not fully materialized")
        if learner.learner_updates != self._learner_updates_completed:
            raise RuntimeError("restored learner update counters do not match")
        if self._latest_weights.learner_updates > learner.learner_updates:
            raise RuntimeError("restored publication is newer than learner state")
        if self._last_report_env_steps > rollout.env_steps:
            raise ValueError("checkpoint report environment steps exceed rollout")
        if self._last_report_episodes > rollout.episodes_collected:
            raise ValueError("checkpoint report episodes exceed rollout")

    @staticmethod
    def _checkpoint_counter(state: Mapping[str, Any], name: str) -> int:
        value = state.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"controller checkpoint {name} is invalid")
        return value

    def _kill_components(self) -> None:
        if self._evaluation_group is not None:
            self._evaluation_group.stop()
        if self._rollout_group is not None:
            self._rollout_group.stop()
        if self._learner_actor is not None:
            ray.kill(self._learner_actor, no_restart=True)
        if self._replay_actor is not None and self._owns_replay_actor:
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
        sac_config, runtime, shared_replay, _ = cls._parse_config(config)
        del sac_config
        bundles = [
            {"CPU": 1.0},
            *cls._child_resource_bundles(
                runtime,
                include_replay=shared_replay is None,
            ),
        ]
        return PlacementGroupFactory(bundles, strategy="PACK")

    @staticmethod
    def _child_resource_bundles(
        runtime: AsyncSACRuntimeConfig,
        *,
        include_replay: bool,
    ) -> list[dict[str, float]]:
        bundles: list[dict[str, float]] = []

        def add_bundle(*, cpu: float = 0.0, gpu: float = 0.0) -> None:
            resources: dict[str, float] = {}
            if cpu:
                resources["CPU"] = cpu
            if gpu:
                resources["GPU"] = gpu
            if resources:
                bundles.append(resources)

        if include_replay:
            add_bundle(cpu=runtime.num_cpus_per_replay)
        add_bundle(
            cpu=runtime.num_cpus_per_learner,
            gpu=runtime.num_gpus_per_learner,
        )
        for _ in range(runtime.runner_count):
            add_bundle(cpu=runtime.num_cpus_per_runner)
        for _ in range(runtime.evaluation_num_episodes):
            add_bundle(cpu=runtime.num_cpus_per_evaluation_runner)
        return bundles

    def setup(self, config: dict[str, Any]) -> None:
        (
            sac_config,
            runtime_config,
            shared_replay,
            member_checkpoint_state,
        ) = self._parse_config(config)
        self._sac_config = sac_config.copy(copy_frozen=False)
        self._runtime_config = runtime_config
        self._shared_replay = shared_replay
        replay_actor = (
            ray.get_actor(
                shared_replay.actor_name,
                namespace=shared_replay.namespace,
            )
            if shared_replay is not None
            else None
        )
        if member_checkpoint_state is not None:
            if replay_actor is None:
                raise ValueError("population member checkpoint requires shared replay")
            self._runtime = SingleMemberAsyncSAC.from_member_checkpoint_state(
                self._sac_config,
                self._runtime_config,
                member_checkpoint_state,
                replay_actor=replay_actor,
            )
        else:
            self._runtime = SingleMemberAsyncSAC(
                self._sac_config,
                self._runtime_config,
                replay_actor=replay_actor,
            )
        self._runtime.start()

    def step(self) -> dict[str, Any]:
        return self._runtime.run_for()

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        if self._runtime.owns_replay_actor:
            self._runtime.save_checkpoint(checkpoint_dir)
        else:
            self._runtime.save_member_checkpoint(checkpoint_dir)
        return None

    def load_checkpoint(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, str | os.PathLike):
            raise TypeError("AsyncSACTrainable requires a directory checkpoint")
        previous = self._runtime
        previous.stop(graceful=False)
        if previous.owns_replay_actor:
            restored = SingleMemberAsyncSAC.from_checkpoint(
                self._sac_config,
                self._runtime_config,
                checkpoint,
            )
        else:
            if self._shared_replay is None:
                raise RuntimeError("shared replay descriptor was lost")
            replay_actor = ray.get_actor(
                self._shared_replay.actor_name,
                namespace=self._shared_replay.namespace,
            )
            restored = SingleMemberAsyncSAC.from_member_checkpoint(
                self._sac_config,
                self._runtime_config,
                checkpoint,
                replay_actor=replay_actor,
            )
        self._runtime = restored
        restored.start()

    def cleanup(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.stop()

    @staticmethod
    def _parse_config(
        config: Mapping[str, Any],
    ) -> tuple[
        SACConfig,
        AsyncSACRuntimeConfig,
        SharedReplayDescriptor | None,
        RuntimeCheckpointState | None,
    ]:
        if not isinstance(config, Mapping):
            raise TypeError("Trainable config must be a mapping")
        unknown = set(config) - {
            "member",
            "sac_config",
            "runtime",
            "shared_replay",
            "member_checkpoint_state",
            "member_checkpoint_states",
        }
        if unknown:
            raise ValueError(f"unknown Trainable settings {sorted(unknown)!r}")
        member = config.get("member")
        if member is not None:
            if not isinstance(member, Mapping):
                raise TypeError("config['member'] must be a mapping")
            if "sac_config" in config or "runtime" in config:
                raise ValueError(
                    "population member config cannot mix nested and top-level values"
                )
            member_unknown = set(member) - {
                "sac_config",
                "runtime",
                "member_checkpoint_state",
            }
            if member_unknown:
                raise ValueError(
                    f"unknown population member settings {sorted(member_unknown)!r}"
                )
            member_values = member
        else:
            member_values = config

        sac_config = member_values.get("sac_config")
        if not isinstance(sac_config, SACConfig):
            raise TypeError("config['sac_config'] must be an SACConfig")
        runtime = AsyncSACRuntimeConfig.from_mapping(
            member_values.get("runtime"),
            sac_config=sac_config,
        )
        shared_replay = SharedReplayDescriptor.from_mapping(config.get("shared_replay"))
        member_checkpoint_state = member_values.get("member_checkpoint_state")
        member_checkpoint_states = config.get("member_checkpoint_states")
        if member_checkpoint_states is not None:
            if not isinstance(member_checkpoint_states, Mapping):
                raise TypeError("config['member_checkpoint_states'] must be a mapping")
            unknown_member_ids = set(member_checkpoint_states) - {
                runtime.member_id,
            }
            if unknown_member_ids and member is None:
                raise ValueError(
                    "standalone Trainable config contains foreign member checkpoints"
                )
            selected_state = member_checkpoint_states.get(runtime.member_id)
            if member_checkpoint_state is not None and selected_state is not None:
                raise ValueError("member checkpoint state was provided twice")
            if selected_state is not None:
                member_checkpoint_state = selected_state
        if member_checkpoint_state is not None and not isinstance(
            member_checkpoint_state,
            RuntimeCheckpointState,
        ):
            raise TypeError(
                "config['member_checkpoint_state'] must be RuntimeCheckpointState"
            )
        return sac_config, runtime, shared_replay, member_checkpoint_state
