"""DQN learner adapter consuming replay-prepared targets with RLlib losses."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
    DefaultDQNTorchRLModule,
)
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.env import INPUT_ENV_SINGLE_SPACES, INPUT_ENV_SPACES
from ray.rllib.policy.sample_batch import MultiAgentBatch

from rllib_async.learner.action_masks import ActionMaskedDQNTorchRLModule
from rllib_async.learner.sac_adapter import (
    SAC_TARGET_UPDATE_STATE,
    LearnerUpdateResult,
    SACBatchError,
    _LocalOffPolicyLearnerAdapter,
    _SelectedModuleUpdates,
    build_rllib_multi_module_sac_batch,
    build_rllib_sac_batch,
)

DQN_ADAPTER_STATE_VERSION = 1
DQN_TARGET_UPDATE_STATE = SAC_TARGET_UPDATE_STATE
DQNUpdateResult = LearnerUpdateResult
C51_DUELING_HEAD_VERSION = 2


class DQNBatchError(ValueError):
    """A batch does not satisfy the prepared discrete DQN contract."""


class DQNLearnerAdapterError(RuntimeError):
    """The DQN adapter cannot satisfy an update or lifecycle request."""


def validate_dqn_config(
    config: DQNConfig,
    *,
    spaces: Mapping[str, tuple[gym.Space, gym.Space]] | None = None,
) -> None:
    """Reject unsupported modes before constructing learner or rollout actors."""

    if not isinstance(config, DQNConfig):
        raise TypeError("config must be a DQNConfig")
    if any(
        "model_config" in overrides
        for overrides in config.algorithm_config_overrides_per_module.values()
    ):
        raise ValueError(
            "RLlib 2.56.1 cannot apply model_config through algorithm overrides; "
            "use RLModuleSpec.model_config for each module"
        )
    configs = [
        config,
        *(
            config.get_config_for_module(module_id)
            for module_id in config.algorithm_config_overrides_per_module
        ),
    ]
    for module_config in configs:
        if (
            not isinstance(module_config.n_step, int)
            or isinstance(module_config.n_step, bool)
            or module_config.n_step < 1
        ):
            raise ValueError("DQN n_step must be a positive integer")
        if not isinstance(module_config.noisy, bool):
            raise ValueError("DQN noisy must be a bool")
        if module_config.noisy and (
            not isinstance(module_config.sigma0, int | float)
            or isinstance(module_config.sigma0, bool)
            or not np.isfinite(module_config.sigma0)
            or module_config.sigma0 <= 0
        ):
            raise ValueError("noisy DQN requires finite positive sigma0")
        if module_config.burn_in_len:
            raise ValueError(
                "recurrent DQN requires sequence replay and burn-in support"
            )
        if (
            not isinstance(module_config.num_atoms, int)
            or isinstance(module_config.num_atoms, bool)
            or module_config.num_atoms < 1
        ):
            raise ValueError("DQN num_atoms must be a positive integer")
        if module_config.num_atoms > 1 and not (
            np.isfinite(module_config.v_min)
            and np.isfinite(module_config.v_max)
            and module_config.v_min < module_config.v_max
        ):
            raise ValueError("C51 requires finite v_min < v_max")
    spec = config.rl_module_spec
    specs = (
        spec.rl_module_specs.values()
        if isinstance(spec, MultiRLModuleSpec)
        else (spec,)
    )
    models = [
        *(item.model_config for item in configs),
        *(item.model_config for item in specs),
    ]
    for model in models:
        if model is None:
            continue
        for name in ("use_lstm", "use_attention"):
            enabled = (
                model.get(name, False)
                if isinstance(model, Mapping)
                else getattr(model, name, False)
            )
            if enabled:
                raise ValueError(
                    "recurrent DQN requires sequence replay and burn-in support"
                )
    if spaces is not None:
        for module_id, (_, action_space) in spaces.items():
            if module_id in (INPUT_ENV_SPACES, INPUT_ENV_SINGLE_SPACES):
                continue
            if (
                not isinstance(action_space, gym.spaces.Discrete)
                or action_space.start != 0
            ):
                raise ValueError(
                    f"DQN module {module_id!r} requires "
                    "a zero-based Discrete action space"
                )


def prepare_dqn_config(config: DQNConfig) -> DQNConfig:
    """Apply the same safe default exploration module to learner and EnvRunner."""

    validate_dqn_config(config)
    prepared = config.copy(copy_frozen=False)
    spec = prepared.rl_module_spec
    specs = (
        spec.rl_module_specs.items()
        if isinstance(spec, MultiRLModuleSpec)
        else ((DEFAULT_MODULE_ID, spec),)
    )
    for module_id, item in specs:
        if item.module_class is DefaultDQNTorchRLModule:
            item.module_class = ActionMaskedDQNTorchRLModule
        module_config = prepared.get_config_for_module(module_id)
        model = item.model_config or {}
        if is_dataclass(model):
            model = asdict(model)
        if "noisy" in model and model["noisy"] != module_config.noisy:
            raise ValueError("configure noisy through DQNConfig.training(noisy=...)")
        if module_config.noisy:
            if not isinstance(item.module_class, type) or not issubclass(
                item.module_class, ActionMaskedDQNTorchRLModule
            ):
                raise ValueError("noisy DQN requires a supported DQN module class")
            item.model_config = {
                **model,
                "noisy": True,
                "std_init": module_config.sigma0,
            }
    prepared.rl_module(rl_module_spec=spec)
    return prepared


class CheckpointableDQNTorchLearner(_SelectedModuleUpdates, DQNTorchLearner):
    """Serialize the target-update cursor omitted by RLlib 2.56.1."""

    def get_state(
        self,
        components: str | Collection[str] | None = None,
        *,
        not_components: str | Collection[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        state = super().get_state(
            components=components, not_components=not_components, **kwargs
        )
        if components is None:
            state[DQN_TARGET_UPDATE_STATE] = {
                module_id: self.last_update_ts_by_mid[module_id]
                for module_id in tuple(self.module.keys())
            }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        target_update_ts = state.get(DQN_TARGET_UPDATE_STATE)
        if target_update_ts is not None:
            if not isinstance(target_update_ts, Mapping) or set(
                target_update_ts
            ) != set(self.module.keys()):
                raise ValueError(
                    "DQN target update state must match the learner module IDs"
                )
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in target_update_ts.values()
            ):
                raise ValueError(
                    "DQN target update timesteps must be non-negative integers"
                )
        super().set_state(state)
        if target_update_ts is not None:
            self.last_update_ts_by_mid = defaultdict(int, target_update_ts)


def _validate_dqn_columns(
    batch: MultiAgentBatch,
    *,
    validate: bool,
    action_spaces: Mapping[str, gym.spaces.Discrete] | None = None,
) -> MultiAgentBatch:
    for module_id, sample in batch.policy_batches.items():
        actions = np.asarray(sample[Columns.ACTIONS])
        if actions.ndim != 1 or actions.dtype.kind not in "iu":
            raise DQNBatchError("DQN actions must be a one-dimensional integer array")
        sample[Columns.ACTIONS] = np.ascontiguousarray(actions, dtype=np.int64)
        if validate and (
            np.any(actions < 0)
            or (
                action_spaces is not None
                and np.any(actions >= action_spaces[module_id].n)
            )
        ):
            raise DQNBatchError("DQN actions must be within the Discrete action space")
    return batch


def build_rllib_dqn_batch(
    batch: Mapping[str, Any], *, validate: bool = True
) -> MultiAgentBatch:
    """Build RLlib's prepared-target input with the common column conversion."""

    try:
        converted = build_rllib_sac_batch(batch, validate=validate)
    except SACBatchError as error:
        raise DQNBatchError(str(error).replace("SAC", "DQN")) from error
    return _validate_dqn_columns(converted, validate=validate)


def build_rllib_multi_module_dqn_batch(
    batches: Mapping[str, Mapping[str, Any]], *, validate: bool = True
) -> MultiAgentBatch:
    try:
        converted = build_rllib_multi_module_sac_batch(batches, validate=validate)
    except SACBatchError as error:
        raise DQNBatchError(str(error).replace("SAC", "DQN")) from error
    return _validate_dqn_columns(converted, validate=validate)


class DQNLearnerAdapter(_LocalOffPolicyLearnerAdapter):
    """Local DQN lifecycle, with stock RLlib loss/optimizer/target updates."""

    _algorithm = "DQN"
    _config_type = DQNConfig
    _stock_learner_class = DQNTorchLearner
    _checkpointable_learner_class = CheckpointableDQNTorchLearner
    _batch_error = DQNBatchError
    _adapter_error = DQNLearnerAdapterError
    _config_contract_fields = (
        "adam_epsilon",
        "count_steps_by",
        "double_q",
        "dueling",
        "gamma",
        "grad_clip",
        "grad_clip_by",
        "learner_config_dict",
        "lr",
        "n_step",
        "noisy",
        "num_atoms",
        "sigma0",
        "target_network_update_freq",
        "tau",
        "td_error_loss_fn",
        "torch_compile_learner",
        "torch_compile_learner_what_to_compile",
        "train_batch_size_per_learner",
        "v_min",
        "v_max",
    )

    def __init__(
        self,
        config: DQNConfig,
        *,
        spaces: Mapping[str, tuple[gym.Space, gym.Space]],
        member_id: str,
        publication_interval_updates: int,
        learning_starts_satisfied: bool = False,
        validate_startup: bool = False,
        publish_updates: bool = True,
    ) -> None:
        validate_dqn_config(config, spaces=spaces)
        self._action_spaces = {
            module_id: pair[1]
            for module_id, pair in spaces.items()
            if module_id not in (INPUT_ENV_SPACES, INPUT_ENV_SINGLE_SPACES)
        }
        super().__init__(
            prepare_dqn_config(config),
            spaces=spaces,
            member_id=member_id,
            publication_interval_updates=publication_interval_updates,
            learning_starts_satisfied=learning_starts_satisfied,
            validate_startup=validate_startup,
            publish_updates=publish_updates,
        )
        configs = (
            config,
            *(
                config.get_config_for_module(module_id)
                for module_id in config.algorithm_config_overrides_per_module
            ),
        )
        self._c51_dueling = any(
            item.dueling and item.num_atoms > 1 for item in configs
        )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        if self._c51_dueling:
            state["c51_dueling_head_version"] = C51_DUELING_HEAD_VERSION
        return state

    def set_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("adapter state must be a mapping")
        if (
            self._c51_dueling
            and state.get("c51_dueling_head_version") != C51_DUELING_HEAD_VERSION
        ):
            raise ValueError(
                "legacy C51 dueling checkpoint has a scalar value head; "
                "resume requires the per-atom value-head checkpoint version"
            )
        # Non-noisy stage-1/2 checkpoints predate these two config fields. Their
        # network shape and next-update semantics are unchanged.
        contract = state.get("config_contract")
        if (
            isinstance(contract, Mapping)
            and "noisy" not in contract
            and "sigma0" not in contract
        ):
            state = copy.deepcopy(dict(state))
            state["config_contract"] = {
                **contract,
                "noisy": False,
                "sigma0": self._config.sigma0,
            }
        super().set_state(state)

    def _build_batch(
        self, batch: Mapping[str, Any], *, validate: bool
    ) -> MultiAgentBatch:
        converted = build_rllib_dqn_batch(batch, validate=validate)
        if validate:
            return _validate_dqn_columns(
                converted, validate=True, action_spaces=self._action_spaces
            )
        return converted

    def _build_multi_module_batch(
        self, batches: Mapping[str, Mapping[str, Any]], *, validate: bool
    ) -> MultiAgentBatch:
        converted = build_rllib_multi_module_dqn_batch(batches, validate=validate)
        if validate:
            return _validate_dqn_columns(
                converted, validate=True, action_spaces=self._action_spaces
            )
        return converted
