"""Two-member Tune launcher with one externally owned replay actor."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune import ResultGrid

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
from rllib_async.runtime.controller import AsyncSACTrainable


class PopulationError(RuntimeError):
    """A two-member population cannot satisfy its lifecycle contract."""


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


class PopulationLauncher:
    """Launch exactly two fixed members against one named detached replay."""

    _SHARED_REPLAY_SETTINGS = (
        "replay_capacity_transitions",
        "replay_capacity_bytes",
        "replay_journal_capacity",
        "num_cpus_per_replay",
    )

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
        self._members = tuple(members)
        if any(
            not isinstance(member, PopulationMemberSpec) for member in self._members
        ):
            raise TypeError("members must contain PopulationMemberSpec values")

        member_ids = tuple(member.runtime_config.member_id for member in self._members)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("population member IDs must be unique")
        first_runtime = self._members[0].runtime_config
        for member in self._members[1:]:
            runtime = member.runtime_config
            for name in self._SHARED_REPLAY_SETTINGS:
                if getattr(runtime, name) != getattr(first_runtime, name):
                    raise ValueError(
                        f"population members must share replay setting {name!r}"
                    )

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
