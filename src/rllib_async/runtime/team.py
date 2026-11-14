"""Mixed DQN/SAC teams using the existing asynchronous member lifecycle."""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import gymnasium as gym
import ray
from ray import cloudpickle
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.policy.policy import PolicySpec

from rllib_async.learner.dqn_adapter import prepare_dqn_config, validate_dqn_config
from rllib_async.protocols.episodes import RawMultiModuleEpisodeCodec
from rllib_async.replay.prioritized import PrioritizedReplayConfig
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime.checkpoint import RuntimeCheckpointState
from rllib_async.runtime.config import AsyncRuntimeConfig
from rllib_async.runtime.controller import SingleMemberAsyncSAC
from rllib_async.runtime.learner_host import LearnerHostActor
from rllib_async.runtime.mixed_learner_host import MixedPBTModelState


def _rollout_contract(config: AlgorithmConfig) -> str:
    """Fingerprint data semantics once, including closures in policy routing."""

    return hashlib.sha256(
        cloudpickle.dumps(
            (
                RawMultiModuleEpisodeCodec.schema_version,
                config.env,
                config.env_config,
                config.policy_mapping_fn,
                config.normalize_actions,
                config.clip_actions,
            )
        )
    ).hexdigest()


def prepare_team_config(
    rollout_config: AlgorithmConfig,
    module_configs: Mapping[str, SACConfig | DQNConfig],
    runtime_config: AsyncRuntimeConfig,
    *,
    trainable_modules: Sequence[str] | None = None,
    sequence_configs: Mapping[str, SequenceConfig] | None = None,
) -> tuple[
    AlgorithmConfig,
    dict[str, SACConfig | DQNConfig],
    dict[str, tuple[gym.Space, gym.Space]],
    tuple[str, ...],
    dict[str, SequenceConfig],
]:
    """Resolve one module spec per algorithm before creating any Ray actors.

    Rollout policies declare the observation/action spaces and agent mapping.
    Each module config owns its learner settings, model and sampling horizon.
    """

    if not isinstance(rollout_config, SACConfig | DQNConfig):
        raise TypeError("rollout_config must be an SACConfig or DQNConfig")
    if not isinstance(runtime_config, AsyncRuntimeConfig):
        raise TypeError("runtime_config must be an AsyncRuntimeConfig")
    if not isinstance(module_configs, Mapping) or not module_configs:
        raise ValueError("module_configs must be a non-empty mapping")
    if any(not isinstance(mid, str) or not mid for mid in module_configs):
        raise ValueError("module IDs must be non-empty strings")
    if set(rollout_config.policies) != set(module_configs):
        raise ValueError("rollout policies and module_configs must have the same IDs")
    if runtime_config.batch_queue_capacity:
        raise ValueError("mixed teams currently require batch_queue_capacity=0")
    if rollout_config.num_envs_per_env_runner != 1:
        raise ValueError("mixed teams require one environment per runner")
    if not (
        rollout_config.enable_rl_module_and_learner
        and rollout_config.enable_env_runner_and_connector_v2
    ):
        raise ValueError("mixed teams require the RLModule and ConnectorV2 API stack")
    trainable = (
        tuple(sorted(module_configs))
        if trainable_modules is None
        else tuple(trainable_modules)
    )
    if (
        not trainable
        or len(set(trainable)) != len(trainable)
        or not set(trainable) <= set(module_configs)
    ):
        raise ValueError("trainable_modules must be a non-empty unique subset")
    sequences = dict(sequence_configs or {})
    if not set(sequences) <= set(module_configs):
        raise ValueError("sequence_configs contains unknown module IDs")
    if any(not isinstance(value, SequenceConfig) for value in sequences.values()):
        raise TypeError("sequence_configs values must be SequenceConfig instances")
    sequences = {
        mid: sequences.get(mid, SequenceConfig()) for mid in sorted(module_configs)
    }

    rollout = rollout_config.copy(copy_frozen=False)
    spaces = {}
    modules = {}
    specs = {}
    for mid in sorted(module_configs):
        policy = rollout.policies[mid]
        if not isinstance(policy, PolicySpec):
            policy = PolicySpec(*policy)
        if policy.observation_space is None or policy.action_space is None:
            raise ValueError(f"rollout policy {mid!r} must declare both spaces")
        spaces[mid] = (policy.observation_space, policy.action_space)
        source = module_configs[mid]
        if not isinstance(source, SACConfig | DQNConfig):
            raise TypeError(f"module {mid!r} must use SACConfig or DQNConfig")
        config = source.copy(copy_frozen=False)
        if set(config.algorithm_config_overrides_per_module) - {mid}:
            raise ValueError(f"module {mid!r} has overrides for other module IDs")
        if any(
            "model_config" in overrides
            for overrides in config.algorithm_config_overrides_per_module.values()
        ):
            raise ValueError("use RLModuleSpec.model_config for module model settings")
        if config.algorithm_config_overrides_per_module:
            config = config.get_config_for_module(mid).copy(copy_frozen=False)
            config.rl_module(algorithm_config_overrides_per_module={})
        if config.num_learners != 0:
            raise ValueError("mixed teams require local learners (num_learners=0)")
        if config.num_gpus_per_learner != runtime_config.num_gpus_per_learner:
            raise ValueError("module and host num_gpus_per_learner must match")
        spec = copy.deepcopy(config.rl_module_spec)
        if isinstance(spec, MultiRLModuleSpec):
            keys = set(spec.rl_module_specs)
            if keys == {DEFAULT_MODULE_ID}:
                spec = spec.rl_module_specs[DEFAULT_MODULE_ID]
            elif keys == {mid}:
                spec = spec.rl_module_specs[mid]
            else:
                raise ValueError(f"module {mid!r} config must define only its own spec")
        config.rl_module(rl_module_spec=spec)
        config.multi_agent(
            policies={mid: copy.deepcopy(policy)},
            policy_mapping_fn=lambda agent_id,
            *args,
            module_id=mid,
            **kwargs: module_id,
            count_steps_by="agent_steps",
            policies_to_train=[mid],
        )
        if isinstance(config, DQNConfig):
            validate_dqn_config(config, spaces={mid: spaces[mid]})
            config = prepare_dqn_config(config)
        module_spec = config.get_multi_rl_module_spec(spaces={mid: spaces[mid]})
        if set(module_spec.rl_module_specs) != {mid}:
            raise ValueError(f"module {mid!r} resolved additional learner modules")
        config.rl_module(rl_module_spec=module_spec)
        modules[mid] = config
        specs[mid] = copy.deepcopy(module_spec.rl_module_specs[mid])
    rollout.rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs=specs))
    rollout.learners(
        num_learners=0,
        num_gpus_per_learner=runtime_config.num_gpus_per_learner,
    )
    rollout.multi_agent(policies_to_train=list(trainable), count_steps_by="agent_steps")
    return rollout, modules, spaces, tuple(sorted(trainable)), sequences


class SingleMemberAsyncTeam(SingleMemberAsyncSAC):
    """One replay and one learner host for a heterogeneous policy team.

    Rollout uses complete atomic inference bundles. The host owns independent
    per-module readiness and budgets; the controller only bounds each call.
    """

    _algorithm = "team"
    _algorithm_config_type = AlgorithmConfig
    _pbt_state_type = MixedPBTModelState

    def __init__(
        self,
        rollout_config: AlgorithmConfig,
        module_configs: Mapping[str, SACConfig | DQNConfig],
        runtime_config: AsyncRuntimeConfig,
        *,
        trainable_modules: Sequence[str] | None = None,
        sequence_configs: Mapping[str, SequenceConfig] | None = None,
        per_configs: Mapping[str, PrioritizedReplayConfig] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        replay_actor: Any | None = None,
        member_checkpoint_state: RuntimeCheckpointState | None = None,
        pbt_state: MixedPBTModelState | None = None,
        reward_window_episodes: int = 100,
        worker_bundle_start: int = 2,
    ) -> None:
        (
            rollout,
            self._module_configs,
            self._team_spaces,
            self._trainable_modules,
            self._sequence_configs,
        ) = prepare_team_config(
            rollout_config,
            module_configs,
            runtime_config,
            trainable_modules=trainable_modules,
            sequence_configs=sequence_configs,
        )
        self._team_rollout_contract = _rollout_contract(rollout)
        self._per_configs = dict(per_configs or {})
        if set(self._per_configs) - set(self._trainable_modules) or any(
            not isinstance(value, PrioritizedReplayConfig)
            for value in self._per_configs.values()
        ):
            raise ValueError(
                "per_configs must map trainable team modules to PER configs"
            )
        if pbt_state is not None:
            self._validate_transfer_contract(pbt_state)
        super().__init__(
            rollout,
            runtime_config,
            checkpoint_dir=checkpoint_dir,
            replay_actor=replay_actor,
            member_checkpoint_state=member_checkpoint_state,
            pbt_state=pbt_state,
            reward_window_episodes=reward_window_episodes,
            worker_bundle_start=worker_bundle_start,
        )

    def _make_codec(self) -> RawMultiModuleEpisodeCodec:
        return RawMultiModuleEpisodeCodec(
            validate_startup=self._config.validate_startup
        )

    def _resolve_spaces(self, config: AlgorithmConfig) -> Mapping[str, tuple[Any, Any]]:
        return self._team_spaces

    def _new_learner_actor(
        self,
        sac_config: AlgorithmConfig,
        spaces: Mapping[str, tuple[Any, Any]],
        replay_actor: Any,
        codec: RawMultiModuleEpisodeCodec,
        *,
        runtime_config: AsyncRuntimeConfig,
        checkpoint_state: bytes | None,
        allow_replay_ahead_on_restore: bool,
        pbt_state: Any,
        learning_starts_satisfied: bool,
    ) -> Any:
        return LearnerHostActor.options(
            num_cpus=runtime_config.num_cpus_per_learner,
            num_gpus=runtime_config.num_gpus_per_learner,
            **runtime_config.core_actor_options(),
        ).remote(
            self._module_configs,
            spaces,
            replay_actor,
            codec,
            trainable_modules=self._trainable_modules,
            sequence_configs=self._sequence_configs,
            per_configs=self._per_configs,
            member_id=runtime_config.member_id,
            publication_interval_updates=runtime_config.publication_interval_updates,
            batch_size=runtime_config.batch_size,
            batch_queue_capacity=runtime_config.batch_queue_capacity,
            batch_seed=runtime_config.seed,
            replay_sync_max_bytes=runtime_config.replay_sync_max_bytes,
            replay_sync_interval_s=runtime_config.replay_sync_interval_s,
            validate_startup=runtime_config.validate_startup,
            checkpoint_state=checkpoint_state,
            allow_replay_ahead_on_restore=allow_replay_ahead_on_restore,
            pbt_state=pbt_state,
            learning_starts_satisfied=False,
        )

    @staticmethod
    def _sampled_step_kwargs(rollout: Any) -> dict[str, Any]:
        return {
            **SingleMemberAsyncSAC._sampled_step_kwargs(rollout),
            "sampled_module_steps": rollout.module_steps,
        }

    def _remaining_learner_update_budget(self, rollout: object) -> int:
        # Only the host knows which unique decisions are committed and indexed.
        # It enforces its finite per-module budgets, including during drain.
        return self._config.learner_updates_per_tick

    def _controller_metrics(self) -> dict[str, Any]:
        metrics = super()._controller_metrics()
        for name in (
            "learner_update_budget",
            "learner_update_budget_remaining",
            "target_training_intensity",
            "effective_training_intensity",
        ):
            metrics.pop(name, None)
        metrics["training_budget_owner"] = "learner_host_per_module"
        return metrics

    def get_report(self, **kwargs: Any) -> dict[str, Any]:
        report = super().get_report(**kwargs)
        # Module batch sizes/intensities differ; the old scalar would misreport them.
        for name in (
            "training_intensity",
            "target_training_intensity",
            "effective_training_intensity",
        ):
            report["learner"].pop(name, None)
        return report

    def _validate_transfer_contract(self, state: MixedPBTModelState) -> None:
        if not isinstance(state, MixedPBTModelState):
            raise TypeError("pbt_state must be MixedPBTModelState")
        if state.rollout_contract != self._team_rollout_contract:
            raise ValueError(
                "PBT environment, routing or action semantics do not match"
            )

    def validate_pbt_state(self, state: MixedPBTModelState) -> None:
        """Preflight a complete donor before the population stops this member."""

        self._require_not_stopped()
        self._validate_transfer_contract(state)
        ray.get(self._learner_actor.validate_pbt_state.remote(state))

    def export_pbt_state(self) -> MixedPBTModelState:
        state = super().export_pbt_state()
        return replace(state, rollout_contract=self._team_rollout_contract)

    @classmethod
    def from_member_checkpoint_state(
        cls,
        rollout_config: AlgorithmConfig,
        module_configs: Mapping[str, SACConfig | DQNConfig],
        runtime_config: AsyncRuntimeConfig,
        checkpoint_state: RuntimeCheckpointState,
        *,
        replay_actor: Any,
        **kwargs: Any,
    ) -> SingleMemberAsyncTeam:
        return cls(
            rollout_config,
            module_configs,
            runtime_config,
            replay_actor=replay_actor,
            member_checkpoint_state=checkpoint_state,
            **kwargs,
        )

    @classmethod
    def from_pbt_state(
        cls,
        rollout_config: AlgorithmConfig,
        module_configs: Mapping[str, SACConfig | DQNConfig],
        runtime_config: AsyncRuntimeConfig,
        pbt_state: MixedPBTModelState,
        *,
        replay_actor: Any,
        **kwargs: Any,
    ) -> SingleMemberAsyncTeam:
        return cls(
            rollout_config,
            module_configs,
            runtime_config,
            replay_actor=replay_actor,
            pbt_state=pbt_state,
            **kwargs,
        )

    @classmethod
    def pbt_state_from_member_checkpoint_state(
        cls,
        rollout_config: AlgorithmConfig,
        module_configs: Mapping[str, SACConfig | DQNConfig],
        runtime_config: AsyncRuntimeConfig,
        checkpoint_state: RuntimeCheckpointState,
        *,
        replay_actor: Any,
        **kwargs: Any,
    ) -> MixedPBTModelState:
        """Extract models on the learner device, releasing it before warm start."""

        source = cls.from_member_checkpoint_state(
            rollout_config,
            module_configs,
            runtime_config,
            checkpoint_state,
            replay_actor=replay_actor,
            **kwargs,
        )
        try:
            return source.export_pbt_state()
        finally:
            source.stop(graceful=False)

    def _controller_checkpoint_state(
        self, *, checkpoint_sequence: int
    ) -> dict[str, Any]:
        state = super()._controller_checkpoint_state(
            checkpoint_sequence=checkpoint_sequence
        )
        state["rollout_contract"] = self._team_rollout_contract
        return state

    def _restore_controller_checkpoint_state(
        self, checkpoint: RuntimeCheckpointState
    ) -> None:
        if checkpoint.controller.get("rollout_contract") != self._team_rollout_contract:
            raise ValueError(
                "mixed checkpoint environment, policy routing "
                "or action transforms differ"
            )
        super()._restore_controller_checkpoint_state(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        rollout_config: AlgorithmConfig,
        module_configs: Mapping[str, SACConfig | DQNConfig],
        runtime_config: AsyncRuntimeConfig,
        checkpoint_dir: str | os.PathLike[str],
        **kwargs: Any,
    ) -> SingleMemberAsyncTeam:
        return cls(
            rollout_config,
            module_configs,
            runtime_config,
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )
