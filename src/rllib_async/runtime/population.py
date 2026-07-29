"""Legacy and single-trial fixed SAC population runtimes."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune import ResultGrid, Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory

from rllib_async.learner import SACLearnerAdapter
from rllib_async.protocols import FlatEpisodeCodec, ReplayCursor, ReplayStats
from rllib_async.replay import ReplayActor
from rllib_async.replay.reference import EpisodeStoreState
from rllib_async.runtime.checkpoint import (
    PopulationCheckpoint,
    RuntimeCheckpointState,
    read_population_checkpoint_bundle,
    read_runtime_member_checkpoint,
    write_population_checkpoint,
)
from rllib_async.runtime.config import (
    AsyncSACRuntimeConfig,
    SharedReplayDescriptor,
)
from rllib_async.runtime.controller import (
    AsyncSACTrainable,
    RuntimeState,
    SingleMemberAsyncSAC,
)


class PopulationError(RuntimeError):
    """A population cannot satisfy its lifecycle contract."""


_SHARED_REPLAY_SETTINGS = (
    "replay_capacity_transitions",
    "replay_capacity_bytes",
    "replay_journal_capacity",
    "num_cpus_per_replay",
)
_SAC_STRUCTURAL_SETTINGS = (
    "env",
    "env_config",
    "framework_str",
    "num_learners",
    "num_gpus_per_learner",
    "count_steps_by",
    "train_batch_size_per_learner",
    "num_steps_sampled_before_learning_starts",
    "n_step",
    "twin_q",
    "gamma",
    "tau",
    "target_entropy",
    "initial_alpha",
    "target_network_update_freq",
    "grad_clip",
    "grad_clip_by",
    "replay_buffer_config",
    "policy_model_config",
    "q_model_config",
    "batch_mode",
    "num_envs_per_env_runner",
    "enable_rl_module_and_learner",
    "enable_env_runner_and_connector_v2",
)


def _member_id_from_trial(trial: Any) -> str:
    return str(trial.config["member"]["runtime"]["member_id"])


def _member_trial_name(trial: Any) -> str:
    return f"population-{_member_id_from_trial(trial)}"


def _member_trial_directory(trial: Any) -> str:
    return _member_id_from_trial(trial)


@dataclass(frozen=True, slots=True)
class PopulationMemberSpec:
    """One fixed Tune member configuration."""

    sac_config: SACConfig
    runtime_config: AsyncSACRuntimeConfig

    def __post_init__(self) -> None:
        if not isinstance(self.sac_config, SACConfig):
            raise TypeError("sac_config must be an SACConfig")
        if not isinstance(self.runtime_config, AsyncSACRuntimeConfig):
            raise TypeError("runtime_config must be AsyncSACRuntimeConfig")


def _validate_population_members(
    members: Sequence[PopulationMemberSpec],
    *,
    validate_spaces: bool,
    validate_structure: bool,
) -> tuple[PopulationMemberSpec, ...]:
    resolved = tuple(members)
    if any(not isinstance(member, PopulationMemberSpec) for member in resolved):
        raise TypeError("members must contain PopulationMemberSpec values")
    member_ids = tuple(member.runtime_config.member_id for member in resolved)
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("population member IDs must be unique")
    if not resolved:
        return resolved

    first_runtime = resolved[0].runtime_config
    first_runtime_settings: dict[str, Any] | None = None
    first_sac_settings: tuple[Any, ...] | None = None
    first_training_intensity: float | None = None
    if validate_structure:
        first_runtime_settings = asdict(first_runtime)
        first_runtime_settings.pop("member_id")
        first_runtime_settings.pop("seed")
        first_sac_settings = tuple(
            getattr(resolved[0].sac_config, name) for name in _SAC_STRUCTURAL_SETTINGS
        )
        first_training_intensity = SingleMemberAsyncSAC._resolve_training_intensity(
            resolved[0].sac_config.training_intensity,
            batch_size=first_runtime.batch_size,
        )
    for member in resolved[1:]:
        runtime = member.runtime_config
        for name in _SHARED_REPLAY_SETTINGS:
            if getattr(runtime, name) != getattr(first_runtime, name):
                raise ValueError(
                    f"population members must share replay setting {name!r}"
                )
        if validate_structure:
            assert first_runtime_settings is not None
            assert first_sac_settings is not None
            assert first_training_intensity is not None
            runtime_settings = asdict(runtime)
            runtime_settings.pop("member_id")
            runtime_settings.pop("seed")
            if runtime_settings != first_runtime_settings:
                raise ValueError("population members must share runtime topology")
            sac_settings = tuple(
                getattr(member.sac_config, name) for name in _SAC_STRUCTURAL_SETTINGS
            )
            if sac_settings != first_sac_settings:
                raise ValueError(
                    "population members must share structural SAC settings"
                )
            training_intensity = SingleMemberAsyncSAC._resolve_training_intensity(
                member.sac_config.training_intensity,
                batch_size=runtime.batch_size,
            )
            if training_intensity != first_training_intensity:
                raise ValueError("population members must share training_intensity")
    if validate_spaces:
        space_fingerprints = {
            SACLearnerAdapter._fingerprint_spaces(
                SingleMemberAsyncSAC._resolve_spaces(member.sac_config)
            )
            for member in resolved
        }
        if len(space_fingerprints) != 1:
            raise ValueError("population members must share observation/action spaces")
    return resolved


def _validate_id_segment(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{name} must be a non-empty path segment")
    return value


def make_runtime_member_id(
    run_id: str,
    slot_id: str,
    generation: int,
) -> str:
    """Build the stable generation-specific identity used by owned actors."""

    run_id = _validate_id_segment(run_id, name="run_id")
    slot_id = _validate_id_segment(slot_id, name="slot_id")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise ValueError("generation must be a non-negative integer")
    return f"{run_id}-{slot_id}-g{generation:04d}"


class PopulationLauncher:
    """Launch exactly two fixed members against one named detached replay."""

    def __init__(
        self,
        members: Sequence[PopulationMemberSpec],
        *,
        replay_actor_name: str | None = None,
        namespace: str | None = None,
        replay_state: EpisodeStoreState | None = None,
        member_checkpoint_states: Mapping[
            str,
            RuntimeCheckpointState,
        ]
        | None = None,
    ) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before PopulationLauncher")
        if len(members) != 2:
            raise ValueError("Phase 8 population requires exactly two members")
        self._members = _validate_population_members(
            members,
            validate_spaces=True,
            validate_structure=False,
        )
        member_ids = tuple(member.runtime_config.member_id for member in self._members)

        actor_name = (
            f"rllib-async-replay-{uuid.uuid4().hex}"
            if replay_actor_name is None
            else replay_actor_name
        )
        if not isinstance(actor_name, str) or not actor_name:
            raise ValueError("replay_actor_name must be a non-empty string")
        resolved_namespace = (
            ray.get_runtime_context().namespace if namespace is None else namespace
        )
        if not isinstance(resolved_namespace, str) or not resolved_namespace:
            raise ValueError("population namespace must be a non-empty string")

        checkpoint_states = dict(member_checkpoint_states or {})
        if replay_state is not None and not isinstance(replay_state, EpisodeStoreState):
            raise TypeError("replay_state must be EpisodeStoreState")
        if checkpoint_states and set(checkpoint_states) != set(member_ids):
            raise ValueError(
                "population checkpoint member IDs do not match launch members"
            )
        for member in self._members:
            member_id = member.runtime_config.member_id
            state = checkpoint_states.get(member_id)
            if state is None:
                continue
            if not isinstance(state, RuntimeCheckpointState):
                raise TypeError(
                    "member_checkpoint_states must contain RuntimeCheckpointState"
                )
            if state.runtime_config != asdict(member.runtime_config):
                raise ValueError(
                    f"checkpoint configuration for {member_id!r} does not match"
                )

        self._descriptor = SharedReplayDescriptor(
            actor_name=actor_name,
            namespace=resolved_namespace,
        )
        self._replay_state = replay_state
        self._member_checkpoint_states = checkpoint_states
        self._replay_actor: Any | None = None
        self._results: ResultGrid | None = None

    @property
    def descriptor(self) -> SharedReplayDescriptor:
        return self._descriptor

    @property
    def replay_actor(self) -> Any:
        if self._replay_actor is None:
            raise PopulationError("population replay has not been started")
        return self._replay_actor

    def start(self) -> None:
        if self._replay_actor is not None:
            return
        runtime = self._members[0].runtime_config
        actor = ReplayActor.options(
            name=self._descriptor.actor_name,
            namespace=self._descriptor.namespace,
            lifetime="detached",
            num_cpus=runtime.num_cpus_per_replay,
        ).remote(
            FlatEpisodeCodec(),
            capacity_transitions=runtime.replay_capacity_transitions,
            capacity_bytes=runtime.replay_capacity_bytes,
            journal_capacity=runtime.replay_journal_capacity,
        )
        try:
            if self._replay_state is not None:
                restored = ray.get(
                    actor.load_checkpoint_state.remote(self._replay_state)
                )
                expected_cursor = ReplayCursor(
                    self._replay_state.store_generation,
                    self._replay_state.mutation_seq,
                )
                if (
                    not isinstance(restored, ReplayStats)
                    or restored.cursor != expected_cursor
                ):
                    raise PopulationError(
                        "shared replay returned invalid restore statistics"
                    )
        except Exception:
            ray.kill(actor, no_restart=True)
            raise
        self._replay_actor = actor

    def fit(self, *, run_config: RunConfig) -> ResultGrid:
        """Run two fixed Tune trials concurrently without a PBT scheduler."""

        if not isinstance(run_config, RunConfig):
            raise TypeError("run_config must be a RunConfig")
        if run_config.checkpoint_config.checkpoint_at_end is not True:
            raise ValueError(
                "population runs require checkpoint_at_end=True so every member "
                "can participate in a population checkpoint"
            )
        if self._results is not None:
            raise PopulationError("PopulationLauncher.fit() may only be called once")
        self.start()

        choices: list[dict[str, Any]] = []
        for member in self._members:
            choice: dict[str, Any] = {
                "sac_config": member.sac_config.copy(copy_frozen=False),
                "runtime": asdict(member.runtime_config),
            }
            choices.append(choice)

        param_space: dict[str, Any] = {
            "member": tune.grid_search(choices),
            "shared_replay": asdict(self._descriptor),
        }
        if self._member_checkpoint_states:
            param_space["member_checkpoint_states"] = self._member_checkpoint_states
        results = tune.Tuner(
            AsyncSACTrainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=1,
                max_concurrent_trials=2,
                reuse_actors=False,
                trial_name_creator=_member_trial_name,
                trial_dirname_creator=_member_trial_directory,
            ),
            run_config=run_config,
        ).fit()
        result_list = list(results)
        if len(result_list) != 2:
            raise PopulationError("Tune did not produce exactly two member trials")
        errors = [result.error for result in result_list if result.error is not None]
        if errors:
            raise PopulationError(f"population member failed: {errors[0]}") from errors[
                0
            ]
        observed_ids = {
            result.config["member"]["runtime"]["member_id"] for result in result_list
        }
        expected_ids = {member.runtime_config.member_id for member in self._members}
        if observed_ids != expected_ids:
            raise PopulationError("Tune result member IDs do not match the population")
        self._results = results
        return results

    def get_replay_stats(self) -> ReplayStats:
        stats = ray.get(self.replay_actor.get_stats.remote())
        if not isinstance(stats, ReplayStats):
            raise PopulationError("shared replay returned invalid statistics")
        return stats

    def save_checkpoint(
        self,
        directory: str | Path,
        *,
        results: ResultGrid | None = None,
    ) -> PopulationCheckpoint:
        """Publish shared replay once after both Tune trials have checkpointed."""

        selected_results = results if results is not None else self._results
        if selected_results is None:
            raise PopulationError("fit the population before checkpointing it")

        member_states: dict[str, RuntimeCheckpointState] = {}
        for result in selected_results:
            if result.error is not None:
                raise PopulationError(
                    f"cannot checkpoint failed member: {result.error}"
                ) from result.error
            if result.checkpoint is None:
                raise PopulationError(
                    "every population member must have a Tune checkpoint"
                )
            member_id = result.config["member"]["runtime"]["member_id"]
            with result.checkpoint.as_directory() as checkpoint_dir:
                state = read_runtime_member_checkpoint(checkpoint_dir)
            if state.member_id != member_id:
                raise PopulationError("Tune checkpoint member ID does not match result")
            if member_id in member_states:
                raise PopulationError("duplicate member checkpoint in Tune results")
            member_states[member_id] = state

        expected_ids = {member.runtime_config.member_id for member in self._members}
        if set(member_states) != expected_ids:
            raise PopulationError("Tune results do not cover every population member")
        replay_state = ray.get(self.replay_actor.get_checkpoint_state.remote())
        if not isinstance(replay_state, EpisodeStoreState):
            raise PopulationError("shared replay returned invalid checkpoint state")
        return write_population_checkpoint(
            directory,
            replay_state=replay_state,
            members=member_states,
        )

    @classmethod
    def from_checkpoint(
        cls,
        members: Sequence[PopulationMemberSpec],
        directory: str | Path,
        *,
        replay_actor_name: str | None = None,
        namespace: str | None = None,
    ) -> PopulationLauncher:
        """Restore shared replay once and carry both member cuts into Tune."""

        _, replay_state, member_states = read_population_checkpoint_bundle(directory)
        launcher = cls(
            members,
            replay_actor_name=replay_actor_name,
            namespace=namespace,
            replay_state=replay_state,
            member_checkpoint_states=member_states,
        )
        launcher.start()
        return launcher

    def close(self) -> None:
        actor = self._replay_actor
        self._replay_actor = None
        if actor is not None:
            ray.kill(actor, no_restart=True)

    def __enter__(self) -> PopulationLauncher:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class PopulationAsyncSAC:
    """Own N active SAC members and one authoritative replay in one process."""

    _POLL_SLEEP_S = 0.001
    _TRAIN_METRIC_KEY = "train/episode_reward_mean"

    def __init__(
        self,
        members: Sequence[PopulationMemberSpec],
        *,
        run_id: str | None = None,
        report_interval_s: float | None = None,
    ) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before PopulationAsyncSAC")
        if len(members) < 2:
            raise ValueError("population requires at least two members")
        templates = _validate_population_members(
            members,
            validate_spaces=True,
            validate_structure=True,
        )
        self._run_id = _validate_id_segment(
            run_id or f"run-{uuid.uuid4().hex[:12]}",
            name="run_id",
        )
        self._report_interval_s = self._resolve_report_interval(
            templates,
            report_interval_s,
        )
        self._slot_ids = tuple(member.runtime_config.member_id for member in templates)
        self._member_specs: dict[str, PopulationMemberSpec] = {}
        for slot_id, member in zip(self._slot_ids, templates, strict=True):
            runtime_member_id = make_runtime_member_id(
                self._run_id,
                slot_id,
                0,
            )
            self._member_specs[slot_id] = PopulationMemberSpec(
                member.sac_config.copy(copy_frozen=False),
                replace(
                    member.runtime_config,
                    member_id=runtime_member_id,
                ),
            )

        self._state = RuntimeState.CREATED
        self._replay_actor: Any | None = None
        self._members: dict[str, SingleMemberAsyncSAC] = {}
        self._next_pump_index = 0
        self._report_index = 0

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def report_interval_s(self) -> float:
        return self._report_interval_s

    @property
    def slot_ids(self) -> tuple[str, ...]:
        return self._slot_ids

    @property
    def runtime_member_ids(self) -> dict[str, str]:
        return {
            slot_id: member.runtime_config.member_id
            for slot_id, member in self._member_specs.items()
        }

    @property
    def replay_actor(self) -> Any:
        if self._replay_actor is None:
            raise PopulationError("population replay has not been started")
        return self._replay_actor

    @property
    def members(self) -> dict[str, SingleMemberAsyncSAC]:
        return dict(self._members)

    def start(self) -> None:
        if self._state is RuntimeState.RUNNING:
            return
        if self._state is not RuntimeState.CREATED:
            raise PopulationError(
                f"cannot start population in state {self._state.value!r}"
            )
        replay_config = self._member_specs[self._slot_ids[0]].runtime_config
        try:
            self._replay_actor = ReplayActor.options(
                num_cpus=replay_config.num_cpus_per_replay,
            ).remote(
                FlatEpisodeCodec(),
                capacity_transitions=replay_config.replay_capacity_transitions,
                capacity_bytes=replay_config.replay_capacity_bytes,
                journal_capacity=replay_config.replay_journal_capacity,
            )
            for slot_id in self._slot_ids:
                member = self._member_specs[slot_id]
                self._members[slot_id] = SingleMemberAsyncSAC(
                    member.sac_config,
                    member.runtime_config,
                    replay_actor=self._replay_actor,
                )
            for member in self._members.values():
                member.start()
        except Exception:
            self._shutdown_components(graceful=False)
            self._state = RuntimeState.FAILED
            raise
        self._state = RuntimeState.RUNNING

    def pump_once(self) -> None:
        self._require_running()
        try:
            for offset in range(len(self._slot_ids)):
                index = (self._next_pump_index + offset) % len(self._slot_ids)
                self._members[self._slot_ids[index]].pump_once(timeout_s=0.0)
            self._next_pump_index = (self._next_pump_index + 1) % len(self._slot_ids)
        except Exception:
            self._state = RuntimeState.FAILED
            raise

    def run_for_report_interval(
        self,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        self._require_running()
        duration_s = self._report_interval_s if duration_s is None else duration_s
        if (
            not isinstance(duration_s, int | float)
            or isinstance(duration_s, bool)
            or not math.isfinite(duration_s)
            or duration_s <= 0
        ):
            raise ValueError("duration_s must be finite and positive")
        deadline = time.monotonic() + duration_s
        while True:
            self.pump_once()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self._POLL_SLEEP_S, remaining))
        return self.get_report()

    def get_report(self) -> dict[str, Any]:
        self._require_running()
        try:
            member_reports = {
                slot_id: self._members[slot_id].get_report(
                    include_authoritative_replay=False,
                )
                for slot_id in self._slot_ids
            }
            replay = ray.get(self.replay_actor.get_stats.remote())
            if not isinstance(replay, ReplayStats):
                raise PopulationError("shared replay returned invalid statistics")
        except Exception:
            self._state = RuntimeState.FAILED
            raise
        self._report_index += 1
        return self._format_report(member_reports, replay)

    def stop(self, *, graceful: bool = True) -> None:
        if self._state is RuntimeState.STOPPED:
            return
        errors = self._shutdown_components(graceful=graceful)
        self._state = RuntimeState.STOPPED
        if errors:
            raise PopulationError(
                f"population shutdown failed: {errors[0]}"
            ) from errors[0]

    def _shutdown_components(self, *, graceful: bool) -> list[BaseException]:
        errors: list[BaseException] = []
        for member in reversed(tuple(self._members.values())):
            try:
                member.stop(graceful=graceful)
            except BaseException as error:
                errors.append(error)
        if self._replay_actor is not None:
            try:
                ray.kill(self._replay_actor, no_restart=True)
            except BaseException as error:
                errors.append(error)
            self._replay_actor = None
        return errors

    def _format_report(
        self,
        member_reports: Mapping[str, Mapping[str, Any]],
        replay: ReplayStats,
    ) -> dict[str, Any]:
        formatted_members: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for slot_id in self._slot_ids:
            report = member_reports[slot_id]
            score = self._finite_metric(
                self._extract_metric(report, self._TRAIN_METRIC_KEY)
            )
            train = dict(report.get("train", {}))
            episodes_since_reset = train.get("episodes_since_metric_reset", 0)
            eligible = (
                score is not None
                and isinstance(episodes_since_reset, int)
                and not isinstance(episodes_since_reset, bool)
                and episodes_since_reset > 0
            )
            if eligible:
                assert score is not None
                scores[slot_id] = score

            learner = dict(report.get("learner", {}))
            learner.setdefault("updates", learner.get("learner_updates", 0))
            rollout = dict(report.get("rollout", {}))
            rollout.setdefault(
                "episodes",
                rollout.get("episodes_collected", 0),
            )
            sac_config = self._member_specs[slot_id].sac_config
            formatted_members[slot_id] = {
                "runtime_member_id": self.runtime_member_ids[slot_id],
                "timesteps_this_iter": report.get("timesteps_this_iter", 0),
                "episodes_this_iter": report.get("episodes_this_iter", 0),
                "train": train,
                "pbt": {
                    "generation": 0,
                    "eligible": int(eligible),
                    "current_score": score if score is not None else math.nan,
                    "exploit_count_as_target": 0,
                },
                "hparams": {
                    "actor_lr": sac_config.actor_lr,
                    "critic_lr": sac_config.critic_lr,
                    "alpha_lr": sac_config.alpha_lr,
                },
                "controller": dict(report.get("controller", {})),
                "rollout": rollout,
                "fast_replay": dict(report.get("fast_replay", {})),
                "batching": dict(report.get("batching", {})),
                "learner": learner,
                "evaluation": dict(report.get("evaluation", {})),
            }

        score_values = tuple(scores.values())
        population_metrics: dict[str, Any] = {
            "run_id": self._run_id,
            "report_index": self._report_index,
            "size": len(self._slot_ids),
            "eligible_members": len(score_values),
            "best_score": max(score_values) if score_values else math.nan,
            "mean_score": (
                math.fsum(score_values) / len(score_values)
                if score_values
                else math.nan
            ),
            "worst_score": min(score_values) if score_values else math.nan,
            "exploit_count": 0,
            "reports_since_perturbation": self._report_index,
            "last_exploit_duration_s": 0.0,
        }
        replay_metrics = {
            "store_generation": replay.cursor.store_generation,
            "cursor": replay.cursor.mutation_seq,
            "episodes": replay.episode_count,
            "transitions": replay.total_transitions,
            "bytes": replay.total_estimated_bytes,
            "insert_rate": math.fsum(
                float(report.get("rollout", {}).get("env_steps_per_s", 0.0))
                for report in member_reports.values()
            ),
            "sample_rate": math.fsum(
                float(report.get("learner", {}).get("samples_per_s", 0.0))
                for report in member_reports.values()
            ),
            "committed_episodes": replay.committed_episodes,
            "duplicate_commits": replay.duplicate_commits,
        }
        return {
            "timesteps_this_iter": sum(
                int(report.get("timesteps_this_iter", 0))
                for report in member_reports.values()
            ),
            "episodes_this_iter": sum(
                int(report.get("episodes_this_iter", 0))
                for report in member_reports.values()
            ),
            "population": population_metrics,
            "members": formatted_members,
            "replay": replay_metrics,
        }

    @staticmethod
    def _extract_metric(
        report: Mapping[str, Any],
        metric_key: str,
    ) -> object | None:
        value: object = report
        for component in metric_key.split("/"):
            if not component or not isinstance(value, Mapping):
                return None
            if component not in value:
                return None
            value = value[component]
        return value

    @staticmethod
    def _finite_metric(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return None
        return resolved if math.isfinite(resolved) else None

    @staticmethod
    def _resolve_report_interval(
        members: Sequence[PopulationMemberSpec],
        value: float | None,
    ) -> float:
        if value is None:
            intervals = {member.runtime_config.report_interval_s for member in members}
            if len(intervals) != 1:
                raise ValueError("population members need one shared report_interval_s")
            value = intervals.pop()
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("report_interval_s must be finite and positive")
        return float(value)

    def _require_running(self) -> None:
        if self._state is not RuntimeState.RUNNING:
            raise PopulationError(
                f"population must be running, not {self._state.value!r}"
            )


class PopulationTrainable(Trainable):
    """One Tune trial owning a complete fixed SAC population."""

    @classmethod
    def default_resource_request(
        cls,
        config: dict[str, Any],
    ) -> PlacementGroupFactory:
        members, _, _ = cls._parse_config(config)
        bundles: list[dict[str, float]] = [{"CPU": 1.0}]
        replay_cpu = members[0].runtime_config.num_cpus_per_replay
        if replay_cpu:
            bundles.append({"CPU": replay_cpu})
        for member in members:
            bundles.extend(
                AsyncSACTrainable._child_resource_bundles(
                    member.runtime_config,
                    include_replay=False,
                )
            )
        return PlacementGroupFactory(bundles, strategy="PACK")

    def setup(self, config: dict[str, Any]) -> None:
        members, run_id, report_interval_s = self._parse_config(config)
        self._population = PopulationAsyncSAC(
            members,
            run_id=run_id,
            report_interval_s=report_interval_s,
        )
        self._population.start()

    def step(self) -> dict[str, Any]:
        return self._population.run_for_report_interval()

    def cleanup(self) -> None:
        population = getattr(self, "_population", None)
        if population is not None:
            population.stop()

    @staticmethod
    def _parse_config(
        config: Mapping[str, Any],
    ) -> tuple[tuple[PopulationMemberSpec, ...], str | None, float]:
        if not isinstance(config, Mapping):
            raise TypeError("PopulationTrainable config must be a mapping")
        unknown = set(config) - {
            "members",
            "run_id",
            "report_interval_s",
        }
        if unknown:
            raise ValueError(
                f"unknown population Trainable settings {sorted(unknown)!r}"
            )
        raw_members = config.get("members")
        if not isinstance(raw_members, Sequence) or isinstance(
            raw_members,
            str | bytes,
        ):
            raise TypeError("config['members'] must be a sequence")

        members: list[PopulationMemberSpec] = []
        for value in raw_members:
            if isinstance(value, PopulationMemberSpec):
                members.append(value)
                continue
            if not isinstance(value, Mapping):
                raise TypeError("population members must be specs or mappings")
            member_unknown = set(value) - {"sac_config", "runtime"}
            if member_unknown:
                raise ValueError(
                    f"unknown population member settings {sorted(member_unknown)!r}"
                )
            sac_config = value.get("sac_config")
            if not isinstance(sac_config, SACConfig):
                raise TypeError("member sac_config must be an SACConfig")
            members.append(
                PopulationMemberSpec(
                    sac_config,
                    AsyncSACRuntimeConfig.from_mapping(
                        value.get("runtime"),
                        sac_config=sac_config,
                    ),
                )
            )
        if len(members) < 2:
            raise ValueError("population requires at least two members")
        resolved_members = _validate_population_members(
            members,
            validate_spaces=False,
            validate_structure=True,
        )

        run_id = config.get("run_id")
        if run_id is not None:
            run_id = _validate_id_segment(run_id, name="run_id")
        report_interval_s = PopulationAsyncSAC._resolve_report_interval(
            resolved_members,
            config.get("report_interval_s"),
        )
        return resolved_members, run_id, report_interval_s
