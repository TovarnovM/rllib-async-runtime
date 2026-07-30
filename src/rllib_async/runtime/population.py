"""Legacy fixed and single-trial PBT SAC population runtimes."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import ray
from ray import cloudpickle, tune
from ray.air import RunConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune import ResultGrid, Trainable
from ray.tune.execution.placement_groups import PlacementGroupFactory

from rllib_async.learner import PBTModelState, SACLearnerAdapter
from rllib_async.protocols import FlatEpisodeCodec, ReplayCursor, ReplayStats
from rllib_async.replay import ReplayActor
from rllib_async.replay.reference import EpisodeStoreState
from rllib_async.runtime.checkpoint import (
    InvalidPopulationCheckpointError,
    PopulationCheckpoint,
    RuntimeCheckpointState,
    read_pbt_checkpoint_metadata,
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


_LOGGER = logging.getLogger(__name__)
_MUTABLE_HPARAMS = ("actor_lr", "critic_lr", "alpha_lr")
_PBT_CHECKPOINT_SCHEMA_VERSION = 2
_POPULATION_RUN_MODES = frozenset({"new", "resume", "warm_start"})
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


def _config_value_fingerprint(value: object) -> str:
    """Return a stable digest for a JSON-like or cloudpickle-able config value."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = cloudpickle.dumps(value)
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FloatMutation:
    """Bounded multiplicative mutation for one learning rate."""

    low: float
    high: float
    factors: tuple[float, ...] = (0.8, 1.2)

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("high", self.high)):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"mutation {name} must be finite")
        if self.low <= 0:
            raise ValueError("mutation low must be positive")
        if self.high <= self.low:
            raise ValueError("mutation high must exceed low")
        if not isinstance(self.factors, Sequence) or isinstance(
            self.factors,
            str | bytes,
        ):
            raise TypeError("mutation factors must be a sequence")
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("mutation factors must not be empty")
        if any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in factors
        ):
            raise ValueError("mutation factors must be finite and positive")
        if all(float(value) == 1.0 for value in factors):
            raise ValueError("mutation factors must be able to change a value")
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(
            self,
            "factors",
            tuple(float(value) for value in factors),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | FloatMutation,
    ) -> FloatMutation:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("mutation must be FloatMutation or a mapping")
        unknown = set(value) - {"low", "high", "factors"}
        if unknown:
            raise ValueError(f"unknown mutation settings {sorted(unknown)!r}")
        try:
            return cls(
                low=value["low"],
                high=value["high"],
                factors=value.get("factors", (0.8, 1.2)),
            )
        except KeyError as error:
            raise ValueError("mutation requires low and high") from error


@dataclass(frozen=True, slots=True)
class SimplePBTConfig:
    """Fixed report-cadence configuration for best-to-worst PBT."""

    perturbation_interval_reports: int
    metric_key: str = "train/episode_reward_mean"
    mode: Literal["max", "min"] = "max"
    reward_window_episodes: int = 100
    min_episodes_after_restart: int = 20
    seed: int = 0
    mutations: Mapping[str, FloatMutation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("perturbation_interval_reports", self.perturbation_interval_reports),
            ("reward_window_episodes", self.reward_window_episodes),
            ("min_episodes_after_restart", self.min_episodes_after_restart),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.metric_key, str)
            or not self.metric_key
            or any(not part for part in self.metric_key.split("/"))
        ):
            raise ValueError("metric_key must be a slash-delimited non-empty key")
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("PBT seed must be an integer")
        mutations = self.mutations
        if not isinstance(mutations, Mapping):
            raise TypeError("mutations must be a mapping")
        unknown = set(mutations) - set(_MUTABLE_HPARAMS)
        if unknown:
            raise ValueError(f"unknown PBT mutation keys {sorted(unknown)!r}")
        resolved: dict[str, FloatMutation] = {}
        for name, value in mutations.items():
            resolved[name] = FloatMutation.from_mapping(value)
        object.__setattr__(self, "mutations", resolved)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | SimplePBTConfig | None,
    ) -> SimplePBTConfig | None:
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("PBT config must be SimplePBTConfig or a mapping")
        unknown = set(value) - {
            "perturbation_interval_reports",
            "metric_key",
            "mode",
            "reward_window_episodes",
            "min_episodes_after_restart",
            "seed",
            "mutations",
        }
        if unknown:
            raise ValueError(f"unknown PBT settings {sorted(unknown)!r}")
        if "perturbation_interval_reports" not in value:
            raise ValueError("PBT config requires perturbation_interval_reports")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class _PBTMemberCheckpointMetadata:
    generation: int
    runtime_member_id: str
    hparams: Mapping[str, Any]
    exploit_count_as_target: int
    sac_seed: int
    env_fingerprint: str
    env_config_fingerprint: str

    @classmethod
    def from_mapping(
        cls,
        slot_id: str,
        value: object,
    ) -> _PBTMemberCheckpointMetadata:
        if not isinstance(value, Mapping):
            raise ValueError(f"PBT member {slot_id!r} metadata must be a mapping")
        expected = {
            "generation",
            "runtime_member_id",
            "exploit_count_as_target",
            *_MUTABLE_HPARAMS,
            "sac_seed",
            "env_fingerprint",
            "env_config_fingerprint",
        }
        if set(value) != expected:
            raise ValueError(f"PBT member {slot_id!r} metadata fields do not match")
        generation = value["generation"]
        exploit_count = value["exploit_count_as_target"]
        for name, counter in (
            ("generation", generation),
            ("exploit_count_as_target", exploit_count),
        ):
            if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
                raise ValueError(f"PBT member {slot_id!r} {name} is invalid")
        if exploit_count != generation:
            raise ValueError(
                f"PBT member {slot_id!r} exploit count does not match generation"
            )
        runtime_member_id = _validate_id_segment(
            value["runtime_member_id"],
            name=f"PBT member {slot_id!r} runtime_member_id",
        )
        sac_seed = value["sac_seed"]
        if not isinstance(sac_seed, int) or isinstance(sac_seed, bool):
            raise ValueError(f"PBT member {slot_id!r} SAC seed is invalid")
        fingerprints = {}
        for name in ("env_fingerprint", "env_config_fingerprint"):
            fingerprint = value[name]
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError(f"PBT member {slot_id!r} {name} is invalid")
            fingerprints[name] = fingerprint
        hparams = {name: value[name] for name in _MUTABLE_HPARAMS}
        return cls(
            generation=generation,
            runtime_member_id=runtime_member_id,
            hparams=hparams,
            exploit_count_as_target=exploit_count,
            sac_seed=sac_seed,
            env_fingerprint=fingerprints["env_fingerprint"],
            env_config_fingerprint=fingerprints["env_config_fingerprint"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "runtime_member_id": self.runtime_member_id,
            **dict(self.hparams),
            "exploit_count_as_target": self.exploit_count_as_target,
            "sac_seed": self.sac_seed,
            "env_fingerprint": self.env_fingerprint,
            "env_config_fingerprint": self.env_config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class _PBTCheckpointMetadata:
    run_id: str
    population_report_index: int
    reports_since_perturbation: int
    exploit_count: int
    population_size: int
    report_interval_s: float
    pbt_config: SimplePBTConfig | None
    members: Mapping[str, _PBTMemberCheckpointMetadata]

    @classmethod
    def from_mapping(cls, value: object) -> _PBTCheckpointMetadata:
        try:
            if not isinstance(value, Mapping):
                raise ValueError("PBT checkpoint metadata must be a mapping")
            expected = {
                "schema_version",
                "run_id",
                "population_report_index",
                "reports_since_perturbation",
                "exploit_count",
                "population_size",
                "report_interval_s",
                "pbt_config",
                "members",
            }
            if set(value) != expected:
                raise ValueError("PBT checkpoint metadata fields do not match")
            if value["schema_version"] != _PBT_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("unsupported PBT checkpoint schema version")
            run_id = _validate_id_segment(value["run_id"], name="PBT run_id")
            counters: dict[str, int] = {}
            for name in (
                "population_report_index",
                "reports_since_perturbation",
                "exploit_count",
                "population_size",
            ):
                item = value[name]
                if (
                    not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < (2 if name == "population_size" else 0)
                ):
                    raise ValueError(f"PBT checkpoint {name} is invalid")
                counters[name] = item
            if (
                counters["reports_since_perturbation"]
                > counters["population_report_index"]
            ):
                raise ValueError("PBT report counters are inconsistent")
            report_interval_s = value["report_interval_s"]
            if (
                not isinstance(report_interval_s, int | float)
                or isinstance(report_interval_s, bool)
                or not math.isfinite(report_interval_s)
                or report_interval_s <= 0
            ):
                raise ValueError("PBT checkpoint report interval is invalid")
            pbt_config = SimplePBTConfig.from_mapping(value["pbt_config"])
            raw_members = value["members"]
            if not isinstance(raw_members, Mapping):
                raise ValueError("PBT checkpoint members must be a mapping")
            members = {
                _validate_id_segment(slot_id, name="PBT slot_id"): (
                    _PBTMemberCheckpointMetadata.from_mapping(slot_id, item)
                )
                for slot_id, item in raw_members.items()
            }
            if len(members) != counters["population_size"]:
                raise ValueError(
                    "PBT checkpoint population size does not match members"
                )
            if pbt_config is not None:
                for slot_id, member in members.items():
                    for name, item in member.hparams.items():
                        if (
                            not isinstance(item, int | float)
                            or isinstance(item, bool)
                            or not math.isfinite(item)
                            or item <= 0
                        ):
                            raise ValueError(
                                f"PBT member {slot_id!r} {name} is invalid"
                            )
                    for name, mutation in pbt_config.mutations.items():
                        item = float(member.hparams[name])
                        if not mutation.low <= item <= mutation.high:
                            raise ValueError(
                                f"PBT member {slot_id!r} {name}={item} is outside "
                                f"[{mutation.low}, {mutation.high}]"
                            )
            if (
                sum(member.exploit_count_as_target for member in members.values())
                != counters["exploit_count"]
            ):
                raise ValueError("PBT checkpoint exploit counters are inconsistent")
            for slot_id, member in members.items():
                if member.runtime_member_id != make_runtime_member_id(
                    run_id,
                    slot_id,
                    member.generation,
                ):
                    raise ValueError(
                        f"PBT member {slot_id!r} runtime ID does not match generation"
                    )
            return cls(
                run_id=run_id,
                population_report_index=counters["population_report_index"],
                reports_since_perturbation=counters["reports_since_perturbation"],
                exploit_count=counters["exploit_count"],
                population_size=counters["population_size"],
                report_interval_s=float(report_interval_s),
                pbt_config=pbt_config,
                members=members,
            )
        except InvalidPopulationCheckpointError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidPopulationCheckpointError(
                f"invalid PBT checkpoint metadata: {error}"
            ) from error

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": _PBT_CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "population_report_index": self.population_report_index,
            "reports_since_perturbation": self.reports_since_perturbation,
            "exploit_count": self.exploit_count,
            "population_size": self.population_size,
            "report_interval_s": self.report_interval_s,
            "pbt_config": (
                None
                if self.pbt_config is None
                else {
                    "perturbation_interval_reports": (
                        self.pbt_config.perturbation_interval_reports
                    ),
                    "metric_key": self.pbt_config.metric_key,
                    "mode": self.pbt_config.mode,
                    "reward_window_episodes": (self.pbt_config.reward_window_episodes),
                    "min_episodes_after_restart": (
                        self.pbt_config.min_episodes_after_restart
                    ),
                    "seed": self.pbt_config.seed,
                    "mutations": {
                        name: {
                            "low": mutation.low,
                            "high": mutation.high,
                            "factors": list(mutation.factors),
                        }
                        for name, mutation in self.pbt_config.mutations.items()
                    },
                }
            ),
            "members": {
                slot_id: member.to_mapping() for slot_id, member in self.members.items()
            },
        }


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
        pbt_config: SimplePBTConfig | None = None,
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
        if pbt_config is not None and not isinstance(
            pbt_config,
            SimplePBTConfig,
        ):
            raise TypeError("pbt_config must be SimplePBTConfig or None")
        if pbt_config is not None and not pbt_config.mutations:
            raise ValueError("enabled PBT requires at least one mutation")
        self._pbt_config = pbt_config
        self._slot_ids = tuple(member.runtime_config.member_id for member in templates)
        self._member_specs: dict[str, PopulationMemberSpec] = {}
        self._base_runtime_seeds: dict[str, int] = {}
        self._base_sac_seeds: dict[str, int] = {}
        for slot_id, member in zip(self._slot_ids, templates, strict=True):
            runtime_member_id = make_runtime_member_id(
                self._run_id,
                slot_id,
                0,
            )
            self._base_runtime_seeds[slot_id] = member.runtime_config.seed
            self._base_sac_seeds[slot_id] = (
                member.runtime_config.seed
                if member.sac_config.seed is None
                else int(member.sac_config.seed)
            )
            self._member_specs[slot_id] = PopulationMemberSpec(
                member.sac_config.copy(copy_frozen=False),
                replace(
                    member.runtime_config,
                    member_id=runtime_member_id,
                ),
            )

        self._generations = dict.fromkeys(self._slot_ids, 0)
        self._exploit_count_as_target = dict.fromkeys(self._slot_ids, 0)
        self._current_hparams = {
            slot_id: {
                name: getattr(self._member_specs[slot_id].sac_config, name)
                for name in _MUTABLE_HPARAMS
            }
            for slot_id in self._slot_ids
        }
        if self._pbt_config is not None:
            for slot_id, hparams in self._current_hparams.items():
                for name, value in hparams.items():
                    if (
                        not isinstance(value, int | float)
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError(
                            f"PBT member {slot_id!r} {name} must be "
                            "a finite positive scalar"
                        )
                    hparams[name] = float(value)
        self._state = RuntimeState.CREATED
        self._replay_actor: Any | None = None
        self._members: dict[str, SingleMemberAsyncSAC] = {}
        self._next_pump_index = 0
        self._report_index = 0
        self._reports_since_perturbation = 0
        self._exploit_count = 0
        self._last_exploit_duration_s = 0.0
        self._last_pbt_event = self._empty_pbt_event("not_started")
        self._restarting_slot: str | None = None
        self._checkpoint_replay_state: EpisodeStoreState | None = None
        self._checkpoint_member_states: (
            dict[
                str,
                RuntimeCheckpointState,
            ]
            | None
        ) = None
        self._warm_start_source_states: (
            dict[
                str,
                RuntimeCheckpointState,
            ]
            | None
        ) = None
        self._warm_start_source_specs: (
            dict[
                str,
                PopulationMemberSpec,
            ]
            | None
        ) = None

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

    @classmethod
    def from_checkpoint(
        cls,
        members: Sequence[PopulationMemberSpec],
        directory: str | Path,
        *,
        run_id: str | None = None,
        report_interval_s: float | None = None,
        pbt_config: SimplePBTConfig | None = None,
    ) -> PopulationAsyncSAC:
        """Build an exact population continuation from one coordinated snapshot."""

        metadata = _PBTCheckpointMetadata.from_mapping(
            read_pbt_checkpoint_metadata(directory)
        )
        _, replay_state, member_states = read_population_checkpoint_bundle(directory)
        if run_id is not None and run_id != metadata.run_id:
            raise ValueError("exact resume run_id does not match checkpoint")
        if pbt_config is not None and pbt_config != metadata.pbt_config:
            raise ValueError("exact resume PBT config does not match checkpoint")
        population = cls(
            members,
            run_id=metadata.run_id,
            report_interval_s=(
                metadata.report_interval_s
                if report_interval_s is None
                else report_interval_s
            ),
            pbt_config=metadata.pbt_config,
        )
        population._prepare_exact_restore(
            metadata,
            replay_state,
            member_states,
        )
        return population

    @classmethod
    def from_warm_start_checkpoint(
        cls,
        members: Sequence[PopulationMemberSpec],
        directory: str | Path,
        *,
        run_id: str | None = None,
        report_interval_s: float | None = None,
        pbt_config: SimplePBTConfig | None = None,
    ) -> PopulationAsyncSAC:
        """Build a fresh PBT run from checkpoint models and shared replay."""

        metadata = _PBTCheckpointMetadata.from_mapping(
            read_pbt_checkpoint_metadata(directory)
        )
        _, replay_state, member_states = read_population_checkpoint_bundle(directory)
        population = cls(
            members,
            run_id=run_id,
            report_interval_s=report_interval_s,
            pbt_config=pbt_config,
        )
        if population.run_id == metadata.run_id:
            raise ValueError("warm start requires a new run_id")
        population._prepare_warm_start(
            metadata,
            replay_state,
            member_states,
        )
        return population

    def _prepare_exact_restore(
        self,
        metadata: _PBTCheckpointMetadata,
        replay_state: EpisodeStoreState,
        member_states: Mapping[str, RuntimeCheckpointState],
    ) -> None:
        if metadata.run_id != self._run_id:
            raise InvalidPopulationCheckpointError(
                "PBT checkpoint run_id does not match population"
            )
        if metadata.report_interval_s != self._report_interval_s:
            raise ValueError("exact resume report interval does not match checkpoint")
        if metadata.pbt_config != self._pbt_config:
            raise ValueError("exact resume PBT config does not match checkpoint")
        if set(metadata.members) != set(self._slot_ids):
            raise ValueError("exact resume population slots do not match checkpoint")
        self._validate_replay_checkpoint_config(replay_state)

        restored_specs: dict[str, PopulationMemberSpec] = {}
        restored_states: dict[str, RuntimeCheckpointState] = {}
        expected_runtime_ids: set[str] = set()
        for slot_id in self._slot_ids:
            member_metadata = metadata.members[slot_id]
            template = self._member_specs[slot_id]
            if self._base_sac_seeds[slot_id] != member_metadata.sac_seed:
                raise ValueError(
                    f"exact resume SAC seed for {slot_id!r} does not match checkpoint"
                )
            for name, expected in (
                ("env", member_metadata.env_fingerprint),
                ("env_config", member_metadata.env_config_fingerprint),
            ):
                if (
                    _config_value_fingerprint(getattr(template.sac_config, name))
                    != expected
                ):
                    raise ValueError(
                        f"exact resume SAC {name} for {slot_id!r} "
                        "does not match checkpoint"
                    )
            sac_config = template.sac_config.copy(copy_frozen=False)
            sac_config.training(**dict(member_metadata.hparams))
            sac_config.debugging(
                seed=self._base_sac_seeds[slot_id] + member_metadata.generation
            )
            runtime_config = replace(
                template.runtime_config,
                member_id=member_metadata.runtime_member_id,
                seed=(self._base_runtime_seeds[slot_id] + member_metadata.generation),
            )
            checkpoint_state = member_states.get(member_metadata.runtime_member_id)
            if checkpoint_state is None:
                raise InvalidPopulationCheckpointError(
                    f"PBT checkpoint is missing member {slot_id!r}"
                )
            if checkpoint_state.runtime_config != asdict(runtime_config):
                raise ValueError(
                    f"exact resume runtime config for {slot_id!r} does not match"
                )
            restored_specs[slot_id] = PopulationMemberSpec(
                sac_config,
                runtime_config,
            )
            restored_states[slot_id] = checkpoint_state
            expected_runtime_ids.add(member_metadata.runtime_member_id)
        if set(member_states) != expected_runtime_ids:
            raise InvalidPopulationCheckpointError(
                "PBT checkpoint contains foreign member states"
            )

        self._member_specs = restored_specs
        self._generations = {
            slot_id: metadata.members[slot_id].generation for slot_id in self._slot_ids
        }
        self._exploit_count_as_target = {
            slot_id: metadata.members[slot_id].exploit_count_as_target
            for slot_id in self._slot_ids
        }
        self._current_hparams = {
            slot_id: dict(metadata.members[slot_id].hparams)
            for slot_id in self._slot_ids
        }
        self._report_index = metadata.population_report_index
        self._reports_since_perturbation = metadata.reports_since_perturbation
        self._exploit_count = metadata.exploit_count
        self._last_pbt_event = self._empty_pbt_event("resumed")
        self._checkpoint_replay_state = replay_state
        self._checkpoint_member_states = restored_states

    def _prepare_warm_start(
        self,
        metadata: _PBTCheckpointMetadata,
        replay_state: EpisodeStoreState,
        member_states: Mapping[str, RuntimeCheckpointState],
    ) -> None:
        if set(metadata.members) != set(self._slot_ids):
            raise ValueError("warm start population slots do not match checkpoint")
        self._validate_replay_checkpoint_config(replay_state)
        self._validate_hparams_in_mutation_bounds()

        source_states: dict[str, RuntimeCheckpointState] = {}
        source_specs: dict[str, PopulationMemberSpec] = {}
        expected_runtime_ids: set[str] = set()
        for slot_id in self._slot_ids:
            member_metadata = metadata.members[slot_id]
            checkpoint_state = member_states.get(member_metadata.runtime_member_id)
            if checkpoint_state is None:
                raise InvalidPopulationCheckpointError(
                    f"PBT checkpoint is missing member {slot_id!r}"
                )
            target_template = self._member_specs[slot_id]
            source_sac_config = target_template.sac_config.copy(copy_frozen=False)
            source_sac_config.training(**dict(member_metadata.hparams))
            source_runtime_config = AsyncSACRuntimeConfig.from_mapping(
                checkpoint_state.runtime_config,
                sac_config=source_sac_config,
            )
            requested_runtime = asdict(target_template.runtime_config)
            source_runtime = asdict(source_runtime_config)
            for name in ("member_id", "seed"):
                requested_runtime.pop(name)
                source_runtime.pop(name)
            if requested_runtime != source_runtime:
                raise ValueError(
                    f"warm start runtime topology for {slot_id!r} does not match"
                )
            source_states[slot_id] = checkpoint_state
            source_specs[slot_id] = PopulationMemberSpec(
                source_sac_config,
                source_runtime_config,
            )
            expected_runtime_ids.add(member_metadata.runtime_member_id)
        if set(member_states) != expected_runtime_ids:
            raise InvalidPopulationCheckpointError(
                "PBT checkpoint contains foreign member states"
            )

        self._checkpoint_replay_state = replay_state
        self._warm_start_source_states = source_states
        self._warm_start_source_specs = source_specs
        self._last_pbt_event = self._empty_pbt_event("warm_started")

    def _validate_replay_checkpoint_config(
        self,
        replay_state: EpisodeStoreState,
    ) -> None:
        runtime = self._member_specs[self._slot_ids[0]].runtime_config
        expected = (
            runtime.replay_capacity_transitions,
            runtime.replay_capacity_bytes,
            runtime.replay_journal_capacity,
        )
        observed = (
            replay_state.capacity_transitions,
            replay_state.capacity_bytes,
            replay_state.journal_capacity,
        )
        if observed != expected:
            raise ValueError("population replay config does not match checkpoint")

    def _validate_hparams_in_mutation_bounds(self) -> None:
        if self._pbt_config is None:
            return
        for slot_id, hparams in self._current_hparams.items():
            for name, mutation in self._pbt_config.mutations.items():
                value = hparams[name]
                if not mutation.low <= value <= mutation.high:
                    raise ValueError(
                        f"warm start {slot_id!r} {name}={value} is outside "
                        f"[{mutation.low}, {mutation.high}]"
                    )

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
            if self._checkpoint_replay_state is not None:
                expected_cursor = ReplayCursor(
                    self._checkpoint_replay_state.store_generation,
                    self._checkpoint_replay_state.mutation_seq,
                )
                restored_replay = ray.get(
                    self._replay_actor.load_checkpoint_state.remote(
                        self._checkpoint_replay_state
                    )
                )
                if (
                    not isinstance(restored_replay, ReplayStats)
                    or restored_replay.cursor != expected_cursor
                ):
                    raise PopulationError(
                        "shared replay returned invalid restore statistics"
                    )
                self._checkpoint_replay_state = None
            for slot_id in self._slot_ids:
                member = self._member_specs[slot_id]
                reward_window_episodes = (
                    self._pbt_config.reward_window_episodes
                    if self._pbt_config is not None
                    else 100
                )
                if self._checkpoint_member_states is not None:
                    self._members[slot_id] = (
                        SingleMemberAsyncSAC.from_member_checkpoint_state(
                            member.sac_config,
                            member.runtime_config,
                            self._checkpoint_member_states[slot_id],
                            replay_actor=self._replay_actor,
                            reward_window_episodes=reward_window_episodes,
                        )
                    )
                elif (
                    self._warm_start_source_states is not None
                    and self._warm_start_source_specs is not None
                ):
                    source = self._warm_start_source_specs[slot_id]
                    pbt_state = (
                        SingleMemberAsyncSAC.pbt_state_from_member_checkpoint_state(
                            source.sac_config,
                            source.runtime_config,
                            self._warm_start_source_states[slot_id],
                            replay_actor=self._replay_actor,
                        )
                    )
                    self._members[slot_id] = SingleMemberAsyncSAC.from_pbt_state(
                        member.sac_config,
                        member.runtime_config,
                        pbt_state,
                        replay_actor=self._replay_actor,
                        reward_window_episodes=reward_window_episodes,
                    )
                else:
                    self._members[slot_id] = SingleMemberAsyncSAC(
                        member.sac_config,
                        member.runtime_config,
                        replay_actor=self._replay_actor,
                        reward_window_episodes=reward_window_episodes,
                    )
            self._checkpoint_member_states = None
            self._warm_start_source_states = None
            self._warm_start_source_specs = None
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
            self._report_index += 1
            if self._pbt_config is not None:
                self._reports_since_perturbation += 1
            else:
                self._reports_since_perturbation = self._report_index
            report = self._format_report(member_reports, replay)
            event = self._maybe_run_pbt_step(member_reports)
            report["pbt"] = event
            report["population"].update(
                {
                    "exploit_count": self._exploit_count,
                    "reports_since_perturbation": (self._reports_since_perturbation),
                    "last_exploit_duration_s": self._last_exploit_duration_s,
                }
            )
            return report
        except Exception:
            self._state = RuntimeState.FAILED
            raise

    def save_checkpoint(
        self,
        directory: str | Path,
        *,
        timeout_s: float | None = None,
    ) -> PopulationCheckpoint:
        """Pause every member and publish one coordinated population snapshot."""

        self._require_running()
        runtime = self._member_specs[self._slot_ids[0]].runtime_config
        timeout_s = runtime.shutdown_timeout_s if timeout_s is None else timeout_s
        if (
            not isinstance(timeout_s, int | float)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        destination = Path(directory)
        if not destination.is_dir():
            raise FileNotFoundError(
                f"checkpoint directory does not exist: {destination}"
            )

        deadline = time.monotonic() + timeout_s
        checkpoint_error: BaseException | None = None
        try:
            for slot_id in self._slot_ids:
                self._members[slot_id].pause(
                    timeout_s=self._checkpoint_remaining(deadline)
                )
            for slot_id in self._slot_ids:
                self._members[slot_id].drain(
                    timeout_s=self._checkpoint_remaining(deadline)
                )

            member_states: dict[str, RuntimeCheckpointState] = {}
            for slot_id in self._slot_ids:
                state = self._members[slot_id].get_member_checkpoint_state(
                    timeout_s=self._checkpoint_remaining(deadline)
                )
                member_states[state.member_id] = state
            replay_state = ray.get(
                self.replay_actor.get_checkpoint_state.remote(),
                timeout=self._checkpoint_remaining(deadline),
            )
            if not isinstance(replay_state, EpisodeStoreState):
                raise PopulationError("shared replay returned invalid checkpoint state")
            return write_population_checkpoint(
                destination,
                replay_state=replay_state,
                members=member_states,
                pbt_metadata=self._checkpoint_metadata().to_mapping(),
            )
        except BaseException as error:
            checkpoint_error = error
            raise
        finally:
            resume_errors: list[BaseException] = []
            for slot_id in self._slot_ids:
                member = self._members.get(slot_id)
                if member is None or member.state is not RuntimeState.PAUSED:
                    continue
                try:
                    member.resume()
                except BaseException as error:
                    resume_errors.append(error)
            if resume_errors:
                self._state = RuntimeState.FAILED
                if checkpoint_error is None:
                    raise PopulationError(
                        f"population checkpoint resume failed: {resume_errors[0]}"
                    ) from resume_errors[0]

    def _checkpoint_metadata(self) -> _PBTCheckpointMetadata:
        members = {
            slot_id: _PBTMemberCheckpointMetadata(
                generation=self._generations[slot_id],
                runtime_member_id=self.runtime_member_ids[slot_id],
                hparams=dict(self._current_hparams[slot_id]),
                exploit_count_as_target=(self._exploit_count_as_target[slot_id]),
                sac_seed=self._base_sac_seeds[slot_id],
                env_fingerprint=_config_value_fingerprint(
                    self._member_specs[slot_id].sac_config.env
                ),
                env_config_fingerprint=_config_value_fingerprint(
                    self._member_specs[slot_id].sac_config.env_config
                ),
            )
            for slot_id in self._slot_ids
        }
        metadata = _PBTCheckpointMetadata(
            run_id=self._run_id,
            population_report_index=self._report_index,
            reports_since_perturbation=self._reports_since_perturbation,
            exploit_count=self._exploit_count,
            population_size=len(self._slot_ids),
            report_interval_s=self._report_interval_s,
            pbt_config=self._pbt_config,
            members=members,
        )
        return _PBTCheckpointMetadata.from_mapping(metadata.to_mapping())

    @staticmethod
    def _checkpoint_remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("population checkpoint timed out")
        return remaining

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
        scores = self._eligible_member_scores(member_reports)
        metric_key = (
            self._pbt_config.metric_key
            if self._pbt_config is not None
            else self._TRAIN_METRIC_KEY
        )
        for slot_id in self._slot_ids:
            report = member_reports[slot_id]
            score = self._finite_metric(self._extract_metric(report, metric_key))
            train = dict(report.get("train", {}))
            eligible = slot_id in scores

            learner = dict(report.get("learner", {}))
            learner.setdefault("updates", learner.get("learner_updates", 0))
            rollout = dict(report.get("rollout", {}))
            rollout.setdefault(
                "episodes",
                rollout.get("episodes_collected", 0),
            )
            formatted_members[slot_id] = {
                "runtime_member_id": self.runtime_member_ids[slot_id],
                "timesteps_this_iter": report.get("timesteps_this_iter", 0),
                "episodes_this_iter": report.get("episodes_this_iter", 0),
                "train": train,
                "pbt": {
                    "generation": self._generations[slot_id],
                    "eligible": int(eligible),
                    "current_score": score if score is not None else math.nan,
                    "exploit_count_as_target": (self._exploit_count_as_target[slot_id]),
                },
                "hparams": dict(self._current_hparams[slot_id]),
                "controller": dict(report.get("controller", {})),
                "rollout": rollout,
                "fast_replay": dict(report.get("fast_replay", {})),
                "batching": dict(report.get("batching", {})),
                "learner": learner,
                "evaluation": dict(report.get("evaluation", {})),
            }

        score_values = tuple(scores.values())
        minimize = self._pbt_config is not None and self._pbt_config.mode == "min"
        population_metrics: dict[str, Any] = {
            "run_id": self._run_id,
            "report_index": self._report_index,
            "size": len(self._slot_ids),
            "eligible_members": len(score_values),
            "best_score": (
                (min(score_values) if minimize else max(score_values))
                if score_values
                else math.nan
            ),
            "mean_score": (
                math.fsum(score_values) / len(score_values)
                if score_values
                else math.nan
            ),
            "worst_score": (
                (max(score_values) if minimize else min(score_values))
                if score_values
                else math.nan
            ),
            "exploit_count": self._exploit_count,
            "reports_since_perturbation": self._reports_since_perturbation,
            "last_exploit_duration_s": self._last_exploit_duration_s,
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

    def _eligible_member_scores(
        self,
        member_reports: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, float]:
        metric_key = (
            self._pbt_config.metric_key
            if self._pbt_config is not None
            else self._TRAIN_METRIC_KEY
        )
        minimum_episodes = (
            self._pbt_config.min_episodes_after_restart
            if self._pbt_config is not None
            else 1
        )
        scores: dict[str, float] = {}
        for slot_id in self._slot_ids:
            member = self._members.get(slot_id)
            if (
                member is not None and member.state is not RuntimeState.RUNNING
            ) or slot_id == self._restarting_slot:
                continue
            report = member_reports[slot_id]
            score = self._finite_metric(self._extract_metric(report, metric_key))
            episodes_since_reset = report.get("train", {}).get(
                "episodes_since_metric_reset",
                0,
            )
            if (
                score is not None
                and isinstance(episodes_since_reset, int)
                and not isinstance(episodes_since_reset, bool)
                and episodes_since_reset >= minimum_episodes
            ):
                scores[slot_id] = score
        return scores

    def _maybe_run_pbt_step(
        self,
        member_reports: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        config = self._pbt_config
        if config is None:
            return self._record_skipped_pbt_event("disabled")
        if self._reports_since_perturbation < config.perturbation_interval_reports:
            return self._record_skipped_pbt_event("interval_not_reached")

        scores = self._eligible_member_scores(member_reports)
        if len(scores) < 2:
            return self._record_skipped_pbt_event("not_enough_eligible_members")
        donor_slot, target_slot = self._select_donor_and_target(
            scores,
            mode=config.mode,
        )
        donor_score = scores[donor_slot]
        target_score = scores[target_slot]
        if donor_score == target_score:
            return self._record_skipped_pbt_event("equal_scores")

        started = time.monotonic()
        donor_state = self._members[donor_slot].export_pbt_state()
        (
            new_hparams,
            mutated_parameter,
            parameter_index,
            factor,
            old_value,
            new_value,
        ) = self._mutate_hparams(
            self._current_hparams[donor_slot],
            config=config,
            exploit_count=self._exploit_count,
        )
        runtime_member_id = self._replace_target(
            target_slot,
            new_hparams=new_hparams,
            donor_state=donor_state,
        )
        self._exploit_count += 1
        self._reports_since_perturbation = 0
        self._last_exploit_duration_s = time.monotonic() - started
        event = {
            "exploit_count": self._exploit_count,
            "event_happened": 1,
            "event_reason": "exploit",
            "mutated_parameter_index": parameter_index,
            "mutation_factor": factor,
            "old_value": old_value,
            "new_value": new_value,
            "donor_score": donor_score,
            "target_score": target_score,
            "donor_slot": donor_slot,
            "target_slot": target_slot,
            "mutated_parameter": mutated_parameter,
            "new_runtime_member_id": runtime_member_id,
            "duration_s": self._last_exploit_duration_s,
        }
        self._last_pbt_event = event
        _LOGGER.info(
            "PBT exploit donor=%s target=%s parameter=%s old=%s new=%s "
            "generation=%s runtime_member_id=%s",
            donor_slot,
            target_slot,
            mutated_parameter,
            old_value,
            new_value,
            self._generations[target_slot],
            runtime_member_id,
        )
        return dict(event)

    @staticmethod
    def _select_donor_and_target(
        scores: Mapping[str, float],
        *,
        mode: Literal["max", "min"],
    ) -> tuple[str, str]:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        if len(scores) < 2:
            raise ValueError("selection requires at least two scores")
        for slot_id, score in scores.items():
            if not isinstance(slot_id, str) or not slot_id:
                raise ValueError("selection slot IDs must be non-empty strings")
            if not math.isfinite(score):
                raise ValueError("selection scores must be finite")
        if mode == "max":
            donor_slot = min(scores, key=lambda slot_id: (-scores[slot_id], slot_id))
            target_slot = min(scores, key=lambda slot_id: (scores[slot_id], slot_id))
        else:
            donor_slot = min(scores, key=lambda slot_id: (scores[slot_id], slot_id))
            target_slot = min(scores, key=lambda slot_id: (-scores[slot_id], slot_id))
        return donor_slot, target_slot

    @staticmethod
    def _mutate_hparams(
        hparams: Mapping[str, Any],
        *,
        config: SimplePBTConfig,
        exploit_count: int,
    ) -> tuple[dict[str, float], str, int, float, float, float]:
        if (
            not isinstance(exploit_count, int)
            or isinstance(exploit_count, bool)
            or exploit_count < 0
        ):
            raise ValueError("exploit_count must be a non-negative integer")
        missing = set(_MUTABLE_HPARAMS) - set(hparams)
        if missing:
            raise ValueError(f"mutable hparams are missing {sorted(missing)!r}")

        candidates: dict[str, tuple[tuple[float, float], ...]] = {}
        for name in sorted(config.mutations):
            value = hparams[name]
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"mutable hparam {name!r} must be positive")
            old_value = float(value)
            mutation = config.mutations[name]
            changed: list[tuple[float, float]] = []
            for factor in mutation.factors:
                new_value = min(
                    max(old_value * factor, mutation.low),
                    mutation.high,
                )
                if new_value == old_value and factor != 1.0:
                    # A one-sided mutation eventually reaches its bound. Reflect
                    # its direction there so a healthy long-running population
                    # cannot fail merely because every donor is saturated.
                    new_value = min(
                        max(old_value / factor, mutation.low),
                        mutation.high,
                    )
                if new_value == old_value and factor != 1.0:
                    # Extremely small floating-point steps may round back to the
                    # same value. The opposite bound is a deterministic last
                    # resort that preserves the mutation's bounded contract.
                    new_value = mutation.high if factor < 1.0 else mutation.low
                if new_value != old_value:
                    changed.append((new_value / old_value, new_value))
            if changed:
                candidates[name] = tuple(changed)
        if not candidates:
            raise PopulationError("no configured PBT mutation can change donor hparams")

        rng = random.Random(config.seed + exploit_count)
        parameter = rng.choice(tuple(sorted(candidates)))
        factor, new_value = rng.choice(candidates[parameter])
        old_value = float(hparams[parameter])
        mutated = {name: float(hparams[name]) for name in _MUTABLE_HPARAMS}
        mutated[parameter] = new_value
        parameter_index = tuple(sorted(config.mutations)).index(parameter)
        return (
            mutated,
            parameter,
            parameter_index,
            factor,
            old_value,
            new_value,
        )

    def _replace_target(
        self,
        target_slot: str,
        *,
        new_hparams: Mapping[str, float],
        donor_state: PBTModelState,
    ) -> str:
        if self._pbt_config is None:
            raise PopulationError("target replacement requires enabled PBT")
        old_target = self._members[target_slot]
        old_spec = self._member_specs[target_slot]
        generation = self._generations[target_slot] + 1
        runtime_member_id = make_runtime_member_id(
            self._run_id,
            target_slot,
            generation,
        )
        runtime_seed = self._base_runtime_seeds[target_slot] + generation
        sac_seed = self._base_sac_seeds[target_slot] + generation
        sac_config = old_spec.sac_config.copy(copy_frozen=False)
        sac_config.training(**dict(new_hparams))
        sac_config.debugging(seed=sac_seed)
        runtime_config = replace(
            old_spec.runtime_config,
            member_id=runtime_member_id,
            seed=runtime_seed,
        )

        self._restarting_slot = target_slot
        replacement: SingleMemberAsyncSAC | None = None
        try:
            old_target.stop()
            replacement = SingleMemberAsyncSAC.from_pbt_state(
                sac_config,
                runtime_config,
                donor_state,
                replay_actor=self.replay_actor,
                reward_window_episodes=self._pbt_config.reward_window_episodes,
            )
            replacement.start()
        except Exception as error:
            if replacement is not None:
                with suppress(BaseException):
                    replacement.stop(graceful=False)
            raise PopulationError(
                f"failed to restart PBT target {target_slot!r}"
            ) from error
        finally:
            self._restarting_slot = None

        self._members[target_slot] = replacement
        self._member_specs[target_slot] = PopulationMemberSpec(
            sac_config,
            runtime_config,
        )
        self._generations[target_slot] = generation
        self._exploit_count_as_target[target_slot] += 1
        self._current_hparams[target_slot] = dict(new_hparams)
        return runtime_member_id

    def _record_skipped_pbt_event(self, reason: str) -> dict[str, Any]:
        event = self._empty_pbt_event(reason)
        self._last_pbt_event = event
        return dict(event)

    def _empty_pbt_event(self, reason: str) -> dict[str, Any]:
        return {
            "exploit_count": self._exploit_count,
            "event_happened": 0,
            "event_reason": reason,
            "mutated_parameter_index": -1,
            "mutation_factor": math.nan,
            "old_value": math.nan,
            "new_value": math.nan,
            "donor_score": math.nan,
            "target_score": math.nan,
            "donor_slot": "",
            "target_slot": "",
            "mutated_parameter": "",
            "new_runtime_member_id": "",
            "duration_s": 0.0,
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
    """One Tune trial owning a complete SAC population."""

    @classmethod
    def default_resource_request(
        cls,
        config: dict[str, Any],
    ) -> PlacementGroupFactory:
        members, _, _, _, _, _ = cls._parse_config(config)
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
        (
            members,
            run_id,
            report_interval_s,
            pbt_config,
            run_mode,
            checkpoint_path,
        ) = self._parse_config(config)
        self._members = members
        self._run_id = run_id
        self._report_interval_s = report_interval_s
        self._pbt_config = pbt_config
        if run_mode == "new":
            self._population = PopulationAsyncSAC(
                members,
                run_id=run_id,
                report_interval_s=report_interval_s,
                pbt_config=pbt_config,
            )
        elif run_mode == "resume":
            assert checkpoint_path is not None
            self._population = PopulationAsyncSAC.from_checkpoint(
                members,
                checkpoint_path,
                run_id=run_id,
                report_interval_s=report_interval_s,
                pbt_config=pbt_config,
            )
        else:
            assert run_mode == "warm_start"
            assert checkpoint_path is not None
            self._population = PopulationAsyncSAC.from_warm_start_checkpoint(
                members,
                checkpoint_path,
                run_id=run_id,
                report_interval_s=report_interval_s,
                pbt_config=pbt_config,
            )
        self._population.start()

    def step(self) -> dict[str, Any]:
        return self._population.run_for_report_interval()

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        self._population.save_checkpoint(checkpoint_dir)
        return None

    def load_checkpoint(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, str | os.PathLike):
            raise TypeError("PopulationTrainable requires a directory checkpoint")
        previous = self._population
        previous.stop(graceful=False)
        restored = PopulationAsyncSAC.from_checkpoint(
            self._members,
            checkpoint,
            run_id=self._run_id,
            report_interval_s=self._report_interval_s,
            pbt_config=self._pbt_config,
        )
        self._population = restored
        restored.start()

    def cleanup(self) -> None:
        population = getattr(self, "_population", None)
        if population is not None:
            population.stop()

    @staticmethod
    def _parse_config(
        config: Mapping[str, Any],
    ) -> tuple[
        tuple[PopulationMemberSpec, ...],
        str | None,
        float,
        SimplePBTConfig | None,
        Literal["new", "resume", "warm_start"],
        str | None,
    ]:
        if not isinstance(config, Mapping):
            raise TypeError("PopulationTrainable config must be a mapping")
        unknown = set(config) - {
            "members",
            "pbt",
            "run_id",
            "report_interval_s",
            "run_mode",
            "checkpoint_path",
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
        pbt_config = SimplePBTConfig.from_mapping(config.get("pbt"))
        run_mode = config.get("run_mode", "new")
        if run_mode not in _POPULATION_RUN_MODES:
            raise ValueError("run_mode must be 'new', 'resume', or 'warm_start'")
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path is not None:
            if not isinstance(checkpoint_path, str | os.PathLike):
                raise TypeError("checkpoint_path must be a filesystem path")
            checkpoint_path = os.fspath(checkpoint_path)
            if not checkpoint_path:
                raise ValueError("checkpoint_path must not be empty")
        if run_mode == "new" and checkpoint_path is not None:
            raise ValueError("new population mode does not accept checkpoint_path")
        if run_mode != "new" and checkpoint_path is None:
            raise ValueError(f"{run_mode} population mode requires checkpoint_path")
        return (
            resolved_members,
            run_id,
            report_interval_s,
            pbt_config,
            run_mode,
            checkpoint_path,
        )
