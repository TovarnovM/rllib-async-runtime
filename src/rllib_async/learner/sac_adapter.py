"""A thin learner-local adapter around RLlib's SAC implementation."""

from __future__ import annotations

import copy
import hashlib
import pickle
import time
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_METRICS_LOGGER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import LR_KEY
from ray.rllib.core.learner.utils import update_target_network
from ray.rllib.core.rl_module.apis import TargetNetworkAPI
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_TARGET_UPDATES,
    TD_ERROR_KEY,
    WEIGHTS_SEQ_NO,
)
from ray.rllib.utils.numpy import convert_to_numpy

from rllib_async.gnn.graph_batch import validate_packed_graph_batch
from rllib_async.protocols.weights import WeightsDescriptor

SAC_ADAPTER_STATE_VERSION = 1
PBT_MODEL_STATE_VERSION = 1
SAC_TEMPERATURE_STATE = "_rllib_async_sac_temperature"
SAC_TARGET_UPDATE_STATE = "_rllib_async_target_update_ts"
_LEARNER_GROUP_STATE = "learner_group"
_REQUIRED_BATCH_COLUMNS = (
    Columns.OBS,
    Columns.NEXT_OBS,
    Columns.ACTIONS,
    Columns.REWARDS,
    Columns.TERMINATEDS,
    Columns.TRUNCATEDS,
)
_OPTIONAL_BATCH_COLUMNS = ("n_step", "weights", "recurrent", "decision_horizons")
_ALLOWED_BATCH_COLUMNS = frozenset((*_REQUIRED_BATCH_COLUMNS, *_OPTIONAL_BATCH_COLUMNS))
_CONFIG_CONTRACT_FIELDS = (
    "actor_lr",
    "alpha_lr",
    "count_steps_by",
    "critic_lr",
    "gamma",
    "grad_clip",
    "grad_clip_by",
    "initial_alpha",
    "learner_config_dict",
    "n_step",
    "policy_model_config",
    "q_model_config",
    "target_entropy",
    "target_network_update_freq",
    "tau",
    "torch_compile_learner",
    "torch_compile_learner_what_to_compile",
    "train_batch_size_per_learner",
    "twin_q",
)


class SACBatchError(ValueError):
    """A collated batch cannot be consumed safely by RLlib SAC."""


class SACLearnerAdapterError(RuntimeError):
    """The SAC adapter cannot satisfy an update or lifecycle request."""


@dataclass(frozen=True, slots=True)
class PBTModelState:
    """Model and temperature state transferred between PBT generations."""

    state_version: int
    spaces_fingerprint: str
    model_fingerprint: str
    module_state: Mapping[str, Any]
    temperature_state: Mapping[str, Any]
    algorithm: str = "SAC"
    config_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"SAC", "DQN"}:
            raise ValueError("unsupported PBT algorithm")
        if (
            not isinstance(self.state_version, int)
            or isinstance(self.state_version, bool)
            or self.state_version != PBT_MODEL_STATE_VERSION
        ):
            raise ValueError("unsupported PBT model state version")
        for name, value in (
            ("spaces_fingerprint", self.spaces_fingerprint),
            ("model_fingerprint", self.model_fingerprint),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name, value in (
            ("module_state", self.module_state),
            ("temperature_state", self.temperature_state),
        ):
            if not isinstance(value, Mapping) or (
                not value and (name == "module_state" or self.algorithm == "SAC")
            ):
                raise ValueError(f"{name} must be a non-empty mapping")
            object.__setattr__(self, name, copy.deepcopy(dict(value)))
        if self.algorithm == "DQN" and self.temperature_state:
            raise ValueError("DQN PBT state cannot contain SAC temperature")
        if self.config_contract is not None:
            if not isinstance(self.config_contract, Mapping):
                raise ValueError("PBT configuration contract must be a mapping")
            object.__setattr__(
                self, "config_contract", copy.deepcopy(dict(self.config_contract))
            )

    def __getstate__(self) -> list[Any]:
        return [
            self.state_version,
            self.spaces_fingerprint,
            self.model_fingerprint,
            self.module_state,
            self.temperature_state,
            self.algorithm,
            self.config_contract,
        ]

    def __setstate__(self, state: list[Any]) -> None:
        if len(state) == 5:
            state = [*state, "SAC", None]
        if len(state) != 7:
            raise ValueError("invalid PBT model state")
        names = (
            "state_version",
            "spaces_fingerprint",
            "model_fingerprint",
            "module_state",
            "temperature_state",
            "algorithm",
            "config_contract",
        )
        for name, value in zip(names, state, strict=True):
            object.__setattr__(self, name, value)
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class LearnerUpdateResult:
    """Outcome of one adapter update request."""

    performed: bool
    learner_updates: int
    sampled_env_steps: int
    sampled_agent_steps: int
    learner_results: list[dict[str, Any]] | None
    published_weights: WeightsDescriptor | None
    priority_errors: Mapping[str, np.ndarray] | None = None


# Preserve the public SAC result type and checkpoint import path.
SACUpdateResult = LearnerUpdateResult


class _SelectedModuleUpdates:
    """Keep RLlib's LR/target hooks on the modules that actually trained.

    The pinned DQN base hook visits every module, even for a partial batch.
    Both SAC and DQN use the same supported target-network and LR APIs here.
    """

    def apply_gradients(self, gradients_dict: dict[Any, Any]) -> None:
        # TorchLearner steps every registered optimizer, including stale grads
        # left by an earlier update. Restrict its existing implementation to
        # optimizers owning gradients from this call; restore the registry even
        # when the update fails. Local learners serialize all update calls.
        original = self._module_optimizers
        if len(original) == 1 or len(gradients_dict) == len(self._params):
            super().apply_gradients(gradients_dict)
            return
        selected = {
            mid: names
            for mid, names in original.items()
            if any(
                self.filter_param_dict_for_optimizer(
                    gradients_dict, self._named_optimizers[name]
                )
                for name in names
            )
        }
        if len(selected) == len(original):
            super().apply_gradients(gradients_dict)
            return
        self._module_optimizers = selected
        try:
            super().apply_gradients(gradients_dict)
        finally:
            self._module_optimizers = original

    def after_gradient_based_update(self, *, timesteps: dict[str, Any]) -> None:
        selected = timesteps.get("rllib_async_updated_modules")
        if selected is None or set(selected) == set(self.module.keys()):
            super().after_gradient_based_update(timesteps=timesteps)
            return
        timestep = timesteps.get(NUM_ENV_STEPS_SAMPLED_LIFETIME, 0)
        for module_id in selected:
            for optimizer_name in self._module_optimizers.get(module_id, ()):
                optimizer = self._named_optimizers[optimizer_name]
                schedule = self._optimizer_lr_schedules.get(optimizer)
                if self._lr_schedulers:
                    for scheduler in self._lr_schedulers.get(module_id, {}).get(
                        optimizer_name, ()
                    ):
                        scheduler.step()
                elif schedule is not None:
                    self._set_optimizer_lr(
                        optimizer, lr=schedule.update(timestep=timestep)
                    )
                self.metrics.log_value(
                    key=(module_id, f"{optimizer_name[len(module_id) + 1 :]}_{LR_KEY}"),
                    reduce="mean",
                    value=self._get_optimizer_lr(optimizer),
                )
            module = self.module[module_id].unwrapped()
            config = self.config.get_config_for_module(module_id)
            if (
                isinstance(module, TargetNetworkAPI)
                and timestep - self.last_update_ts_by_mid[module_id]
                >= config.target_network_update_freq
            ):
                for main_net, target_net in module.get_target_network_pairs():
                    update_target_network(
                        main_net=main_net, target_net=target_net, tau=config.tau
                    )
                self.last_update_ts_by_mid[module_id] = timestep
                self.metrics.log_value(
                    (module_id, NUM_TARGET_UPDATES), 1, reduce="lifetime_sum"
                )
                self.metrics.log_value(
                    (module_id, LAST_TARGET_UPDATE_TS), timestep, reduce="max"
                )


class CheckpointableSACTorchLearner(_SelectedModuleUpdates, SACTorchLearner):
    """Persist SAC state and restrict partial updates to participating modules.

    RLlib supplies the losses, gradient computation, optimizers and target
    operations. The shared mixin limits their application for partial batches;
    this subclass adds temperature and target-clock checkpoint state.
    """

    def get_state(
        self,
        components: str | Collection[str] | None = None,
        *,
        not_components: str | Collection[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        state = super().get_state(
            components=components,
            not_components=not_components,
            **kwargs,
        )
        if components is None:
            state[SAC_TEMPERATURE_STATE] = self.get_temperature_state()
            state[SAC_TARGET_UPDATE_STATE] = {
                module_id: self.last_update_ts_by_mid[module_id]
                for module_id in tuple(self.module.keys())
            }
        return state

    def get_temperature_state(self) -> dict[str, np.ndarray]:
        return {
            module_id: value.detach().cpu().numpy().copy()
            for module_id, value in self.curr_log_alpha.items()
        }

    def set_state(self, state: dict[str, Any]) -> None:
        temperatures = self._validate_temperatures(state.get(SAC_TEMPERATURE_STATE))
        target_update_ts = self._validate_target_update_ts(
            state.get(SAC_TARGET_UPDATE_STATE)
        )

        super().set_state(state)

        if temperatures is not None:
            for module_id, value in temperatures.items():
                target = self.curr_log_alpha[module_id]
                with torch.no_grad():
                    target.copy_(value)
        if target_update_ts is not None:
            self.last_update_ts_by_mid = defaultdict(int, target_update_ts)

    def _validate_temperatures(
        self,
        raw: object,
    ) -> dict[str, torch.Tensor] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"{SAC_TEMPERATURE_STATE} must be a mapping")
        expected = set(self.curr_log_alpha)
        if set(raw) != expected:
            raise ValueError(
                f"{SAC_TEMPERATURE_STATE} module IDs must be {sorted(expected)!r}"
            )
        converted: dict[str, torch.Tensor] = {}
        for module_id, value in raw.items():
            target = self.curr_log_alpha[module_id]
            tensor = torch.as_tensor(
                value,
                dtype=target.dtype,
                device=target.device,
            )
            if tensor.shape != target.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid SAC temperature for module {module_id!r}")
            converted[module_id] = tensor
        return converted

    def _validate_target_update_ts(
        self,
        raw: object,
    ) -> dict[str, int] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"{SAC_TARGET_UPDATE_STATE} must be a mapping")
        expected = set(self.module.keys())
        if set(raw) != expected:
            raise ValueError(
                f"{SAC_TARGET_UPDATE_STATE} module IDs must be {sorted(expected)!r}"
            )
        validated: dict[str, int] = {}
        for module_id, value in raw.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"target update timestep for module {module_id!r} "
                    "must be a non-negative integer"
                )
            validated[module_id] = value
        return validated


def _build_sac_sample_batch(
    batch: Mapping[str, Any], *, validate: bool = True
) -> SampleBatch:
    if not validate:
        columns = dict(batch)
        batch_size = len(columns[Columns.REWARDS])
        columns[Columns.REWARDS] = np.ascontiguousarray(
            columns[Columns.REWARDS], dtype=np.float32
        )
        for name, dtype in (("n_step", np.int64), ("weights", np.float32)):
            columns[name] = (
                np.ascontiguousarray(columns[name], dtype=dtype)
                if name in columns
                else np.ones(batch_size, dtype=dtype)
            )
        return SampleBatch(columns)
    if not isinstance(batch, Mapping):
        raise SACBatchError("SAC batch must be a mapping")
    missing = set(_REQUIRED_BATCH_COLUMNS) - set(batch)
    if missing:
        raise SACBatchError(f"SAC batch is missing columns {sorted(missing)!r}")
    extra = set(batch) - _ALLOWED_BATCH_COLUMNS
    if extra:
        raise SACBatchError(f"SAC batch has unsupported columns {sorted(extra)!r}")

    columns: dict[str, Any] = {}
    observations = batch[Columns.OBS]
    next_observations = batch[Columns.NEXT_OBS]
    mapped_observations = isinstance(observations, Mapping)
    if mapped_observations != isinstance(next_observations, Mapping):
        raise SACBatchError("'obs' and 'new_obs' must use the same representation")
    graph_observations = mapped_observations and "node_features" in observations
    if graph_observations:
        try:
            packed_observations = validate_packed_graph_batch(observations)
            packed_next_observations = validate_packed_graph_batch(
                next_observations, terminated=batch[Columns.TERMINATEDS]
            )
        except ValueError as error:
            raise SACBatchError(str(error)) from error
        batch_size = len(packed_observations["graph_ptr"]) - 1
        _validate_graph_observation_pair(
            packed_observations,
            packed_next_observations,
        )
        columns[Columns.OBS] = packed_observations
        columns[Columns.NEXT_OBS] = packed_next_observations
    elif mapped_observations:
        current, batch_size = _validate_observation_mapping(observations)
        following, next_size = _validate_observation_mapping(next_observations)
        if batch_size != next_size or set(current) != set(following):
            raise SACBatchError(
                "'obs' and 'new_obs' mappings must have matching keys/counts"
            )
        columns[Columns.OBS] = current
        columns[Columns.NEXT_OBS] = following
    else:
        observations_array = _required_batch_array(
            observations,
            name=Columns.OBS,
        )
        next_observations_array = _required_batch_array(
            next_observations,
            name=Columns.NEXT_OBS,
        )
        if observations_array.shape != next_observations_array.shape:
            raise SACBatchError("'obs' and 'new_obs' must have identical shapes")
        batch_size = len(observations_array)
        columns[Columns.OBS] = observations_array
        columns[Columns.NEXT_OBS] = next_observations_array

    if batch_size < 1:
        raise SACBatchError("SAC batch must contain at least one transition")
    for name in (
        Columns.ACTIONS,
        Columns.REWARDS,
        Columns.TERMINATEDS,
        Columns.TRUNCATEDS,
    ):
        value = np.asarray(batch[name])
        if value.ndim == 0:
            raise SACBatchError(f"column {name!r} must have a batch dimension")
        if value.dtype.kind not in "biuf":
            raise SACBatchError(f"column {name!r} must be real numeric or boolean")
        if value.dtype.kind in "f" and not np.isfinite(value).all():
            raise SACBatchError(f"column {name!r} must contain finite values")
        if len(value) != batch_size:
            raise SACBatchError("all SAC columns must have the same leading dimension")
        columns[name] = np.ascontiguousarray(value)

    for name in (Columns.REWARDS, Columns.TERMINATEDS, Columns.TRUNCATEDS):
        if columns[name].shape != (batch_size,):
            raise SACBatchError(f"column {name!r} must be one-dimensional")

    if columns[Columns.TERMINATEDS].dtype.kind != "b":
        raise SACBatchError(f"column {Columns.TERMINATEDS!r} must be boolean")
    if columns[Columns.TRUNCATEDS].dtype.kind != "b":
        raise SACBatchError(f"column {Columns.TRUNCATEDS!r} must be boolean")
    columns[Columns.REWARDS] = np.ascontiguousarray(
        columns[Columns.REWARDS],
        dtype=np.float32,
    )

    n_step = np.asarray(batch.get("n_step", np.ones(batch_size, dtype=np.int64)))
    if (
        n_step.shape != (batch_size,)
        or n_step.dtype.kind not in "iu"
        or np.any(n_step < 1)
    ):
        raise SACBatchError("column 'n_step' must contain positive integers")
    columns["n_step"] = np.ascontiguousarray(n_step, dtype=np.int64)

    weights = np.asarray(batch.get("weights", np.ones(batch_size, dtype=np.float32)))
    if (
        weights.shape != (batch_size,)
        or weights.dtype.kind not in "biuf"
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise SACBatchError("column 'weights' must contain finite non-negative scalars")
    columns["weights"] = np.ascontiguousarray(weights, dtype=np.float32)

    if "recurrent" in batch:
        columns["recurrent"] = _validate_recurrent_metadata(
            batch["recurrent"], batch_size
        )
    if "decision_horizons" in batch:
        horizons = np.asarray(batch["decision_horizons"])
        if (
            horizons.shape != (batch_size,)
            or horizons.dtype.kind not in "iu"
            or np.any(horizons < 1)
        ):
            raise SACBatchError(
                "decision_horizons must contain one positive integer per loss row"
            )
        columns["decision_horizons"] = np.ascontiguousarray(horizons, dtype=np.int64)

    return SampleBatch(columns)


def _validate_observation_mapping(
    observations: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    if not observations:
        raise SACBatchError("observation mappings must not be empty")
    columns = {}
    lengths = set()
    for name, value in observations.items():
        if isinstance(value, Mapping):
            columns[name], count = _validate_observation_mapping(value)
        else:
            columns[name] = _required_batch_array(value, name=name)
            count = len(columns[name])
        lengths.add(count)
    if len(lengths) != 1:
        raise SACBatchError(
            "observation mapping fields must have the same batch dimension"
        )
    return columns, lengths.pop()


def _validate_recurrent_metadata(raw: object, batch_size: int) -> Mapping[str, Any]:
    required = {"obs", "seq_lens", "prefix_lens", "current_indices", "next_indices"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise SACBatchError("recurrent metadata has invalid columns")
    lengths = np.asarray(raw["seq_lens"])
    prefixes = np.asarray(raw["prefix_lens"])
    if (
        lengths.ndim != 1
        or not len(lengths)
        or lengths.dtype.kind not in "iu"
        or np.any(lengths < 1)
    ):
        raise SACBatchError("recurrent seq_lens must contain positive integers")
    if (
        prefixes.shape != lengths.shape
        or prefixes.dtype.kind not in "iu"
        or np.any(prefixes < 0)
        or np.any(prefixes >= lengths)
    ):
        raise SACBatchError("recurrent prefix_lens are outside their histories")
    observations = raw["obs"]
    if isinstance(observations, Mapping) and "graph_ptr" in observations:
        history_count = len(observations["graph_ptr"]) - 1
    elif isinstance(observations, Mapping):
        _, history_count = _validate_observation_mapping(observations)
    else:
        history_count = len(_required_batch_array(observations, name="recurrent obs"))
    if int(lengths.sum()) != history_count:
        raise SACBatchError("recurrent history lengths do not match observations")
    indices = []
    for name in ("current_indices", "next_indices"):
        value = np.asarray(raw[name])
        if (
            value.shape != (batch_size,)
            or value.dtype.kind not in "iu"
            or np.any(value < 0)
            or np.any(value >= history_count)
        ):
            raise SACBatchError(f"recurrent {name} is outside its history")
        indices.append(value)
    ends = np.cumsum(lengths)
    current_history = np.searchsorted(ends, indices[0], side="right")
    next_history = np.searchsorted(ends, indices[1], side="right")
    starts = np.concatenate(([0], ends[:-1]))
    if (
        np.any(current_history != next_history)
        or np.any(indices[1] <= indices[0])
        or np.any(indices[0] < starts[current_history] + prefixes[current_history])
    ):
        raise SACBatchError(
            "recurrent loss/target indices cross prefix or agent boundaries"
        )
    return raw


def _required_batch_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise SACBatchError(f"column {name!r} must have a batch dimension")
    if array.dtype.kind not in "biuf":
        raise SACBatchError(f"column {name!r} must be real numeric or boolean")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise SACBatchError(f"column {name!r} must contain finite values")
    return np.ascontiguousarray(array)


def _validate_graph_observation_pair(
    observations: Mapping[str, np.ndarray],
    next_observations: Mapping[str, np.ndarray],
) -> None:
    if len(observations["graph_ptr"]) != len(next_observations["graph_ptr"]):
        raise SACBatchError("'obs' and 'new_obs' graph counts must match")
    if (
        observations["node_features"].shape[1]
        != next_observations["node_features"].shape[1]
    ):
        raise SACBatchError("'obs' and 'new_obs' node feature dimensions must match")
    for optional in ("edge_features", "action_mask"):
        if (optional in observations) != (optional in next_observations):
            raise SACBatchError(
                f"'obs' and 'new_obs' must agree on optional {optional!r}"
            )
        if optional in observations and (
            observations[optional].shape[1:] != next_observations[optional].shape[1:]
        ):
            raise SACBatchError(
                f"'obs' and 'new_obs' {optional!r} feature shapes must match"
            )


def build_rllib_sac_batch(
    batch: Mapping[str, Any], *, validate: bool = True
) -> MultiAgentBatch:
    """Validate one NumPy transition batch and build RLlib's exact SAC input."""

    sample_batch = _build_sac_sample_batch(batch, validate=validate)
    return MultiAgentBatch(
        {DEFAULT_MODULE_ID: sample_batch},
        env_steps=sample_batch.count,
    )


def build_rllib_multi_module_sac_batch(
    batches: Mapping[str, Mapping[str, Any]],
    *,
    validate: bool = True,
) -> MultiAgentBatch:
    """Validate independent batches for one heterogeneous MultiRLModule."""

    if not isinstance(batches, Mapping) or not batches:
        raise SACBatchError("multi-module SAC batches must be a non-empty mapping")
    if any(not isinstance(module_id, str) or not module_id for module_id in batches):
        raise SACBatchError("module IDs must be non-empty strings")
    sample_batches = {
        module_id: _build_sac_sample_batch(batch, validate=validate)
        for module_id, batch in batches.items()
    }
    return MultiAgentBatch(
        sample_batches,
        env_steps=max(batch.count for batch in sample_batches.values()),
    )


class _LocalOffPolicyLearnerAdapter:
    """Shared local learner lifecycle; algorithm subclasses own batch/config hooks."""

    _algorithm = "SAC"
    _config_type = SACConfig
    _stock_learner_class = SACTorchLearner
    _checkpointable_learner_class = CheckpointableSACTorchLearner
    _config_contract_fields = _CONFIG_CONTRACT_FIELDS
    _batch_error = SACBatchError
    _adapter_error = SACLearnerAdapterError

    def __init__(
        self,
        config: SACConfig,
        *,
        spaces: Mapping[str, tuple[gym.Space, gym.Space]],
        member_id: str,
        publication_interval_updates: int,
        learning_starts_satisfied: bool = False,
        validate_startup: bool = False,
        publish_updates: bool = True,
    ) -> None:
        if not isinstance(validate_startup, bool):
            raise TypeError("validate_startup must be a bool")
        self._validate_next_batch = validate_startup
        if not isinstance(publish_updates, bool):
            raise TypeError("publish_updates must be a bool")
        self._publish_updates = publish_updates
        if not isinstance(config, self._config_type):
            raise TypeError(f"config must be a {self._config_type.__name__}")
        if not isinstance(member_id, str) or not member_id:
            raise ValueError("member_id must be a non-empty string")
        if (
            not isinstance(publication_interval_updates, int)
            or isinstance(publication_interval_updates, bool)
            or publication_interval_updates < 1
        ):
            raise ValueError("publication_interval_updates must be positive")
        if config.framework_str != "torch":
            raise ValueError(f"{self._algorithm}LearnerAdapter supports PyTorch only")
        if not config.enable_rl_module_and_learner:
            raise ValueError(
                f"{self._algorithm}LearnerAdapter requires "
                "the RLModule/Learner API stack"
            )
        if not config.enable_env_runner_and_connector_v2:
            raise ValueError(f"{self._algorithm}LearnerAdapter requires ConnectorV2")
        if config.num_learners != 0:
            raise ValueError(
                f"{self._algorithm}LearnerAdapter requires one local RLlib learner"
            )
        if not isinstance(learning_starts_satisfied, bool):
            raise TypeError("learning_starts_satisfied must be a bool")
        resolved_learner_class = config.learner_class
        if resolved_learner_class not in {
            None,
            self._stock_learner_class,
            self._checkpointable_learner_class,
        }:
            raise ValueError(
                f"custom {self._algorithm} learner classes are not supported in v0.1"
            )

        adapter_config = config.copy(copy_frozen=False)
        adapter_config.learners(learner_class=self._checkpointable_learner_class)

        self._config = adapter_config
        self._member_id = member_id
        self._publication_interval_updates = publication_interval_updates
        self._learning_starts = adapter_config.num_steps_sampled_before_learning_starts
        self._learning_starts_satisfied = learning_starts_satisfied
        self._config_contract = {
            name: copy.deepcopy(getattr(adapter_config, name))
            for name in self._config_contract_fields
        }
        self._module_config_contract = {
            name: copy.deepcopy(getattr(adapter_config, name))
            for name in (
                "model_config",
                "rl_module_spec",
                "algorithm_config_overrides_per_module",
            )
        }
        try:
            pickle.dumps(
                (self._config_contract, self._module_config_contract),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (pickle.PickleError, TypeError) as error:
            raise ValueError(
                f"{self._algorithm} configuration contract must be pickle-safe"
            ) from error
        self._spaces_fingerprint = self._fingerprint_spaces(spaces)
        self._learner_group = adapter_config.build_learner_group(spaces=dict(spaces))
        self._closed = False
        self._learner_updates = 0
        self._sampled_env_steps = 0
        self._sampled_agent_steps = 0
        self._last_published_update = 0

        module_state = self._get_inference_module_state()
        if not set(module_state).issubset(spaces):
            self._learner_group.shutdown()
            self._closed = True
            raise self._adapter_error(
                "RLlib learner module IDs are missing configured spaces"
            )
        self._module_versions = {module_id: 0 for module_id in sorted(module_state)}
        self._latest_weights = self._new_descriptor(
            module_state,
            module_versions=self._module_versions,
            learner_updates=0,
        )

    @property
    def member_id(self) -> str:
        return self._member_id

    @property
    def learner_updates(self) -> int:
        return self._learner_updates

    @property
    def sampled_env_steps(self) -> int:
        return self._sampled_env_steps

    @property
    def sampled_agent_steps(self) -> int:
        return self._sampled_agent_steps

    def update(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int | None = None,
        return_priority_errors: bool = False,
    ) -> LearnerUpdateResult:
        """Perform at most one RLlib update at absolute sampled-step counters."""

        self._require_open()
        if set(self._module_versions) != {DEFAULT_MODULE_ID}:
            raise self._batch_error(
                "update() requires a single default module; "
                "use update_modules() for a heterogeneous learner"
            )
        sampled_agent_steps, should_update = self._begin_update(
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
        )
        if not should_update:
            return self._skipped_update_result(
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
            )
        rllib_batch = self._build_batch(batch, validate=self._validate_next_batch)
        update = self._perform_update(
            rllib_batch,
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
            return_priority_errors=return_priority_errors,
        )
        self._validate_next_batch = False
        return update

    def update_modules(
        self,
        batches: Mapping[str, Mapping[str, Any]],
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        return_priority_errors: bool = False,
    ) -> LearnerUpdateResult:
        """Update a non-empty subset without advancing untouched module hooks."""

        self._require_open()
        if not isinstance(batches, Mapping):
            raise self._batch_error(
                f"multi-module {self._algorithm} batches must be a non-empty mapping"
            )
        if not set(batches).issubset(self._module_versions):
            raise self._batch_error(
                "multi-module batch IDs must belong to the learner module IDs"
            )
        sampled_agent_steps, should_update = self._begin_update(
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
        )
        if not should_update or not batches:
            return self._skipped_update_result(
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
            )
        rllib_batch = self._build_multi_module_batch(
            batches, validate=self._validate_next_batch
        )
        update = self._perform_update(
            rllib_batch,
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
            return_priority_errors=return_priority_errors,
        )
        self._validate_next_batch = False
        return update

    def _begin_update(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int | None,
    ) -> tuple[int, bool]:
        self._require_open()
        sampled_agent_steps = (
            sampled_env_steps if sampled_agent_steps is None else sampled_agent_steps
        )
        self._validate_counter(
            "sampled_env_steps",
            sampled_env_steps,
            previous=self._sampled_env_steps,
        )
        self._validate_counter(
            "sampled_agent_steps",
            sampled_agent_steps,
            previous=self._sampled_agent_steps,
        )
        self._sampled_env_steps = sampled_env_steps
        self._sampled_agent_steps = sampled_agent_steps

        threshold_steps = (
            sampled_agent_steps
            if self._config.count_steps_by == "agent_steps"
            else sampled_env_steps
        )
        if threshold_steps >= self._learning_starts:
            self._learning_starts_satisfied = True
        return sampled_agent_steps, self._learning_starts_satisfied

    def _perform_update(
        self,
        learner_batch: MultiAgentBatch,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
        return_priority_errors: bool = False,
    ) -> LearnerUpdateResult:
        learner_results = self._learner_group.update(
            batch=learner_batch,
            timesteps={
                NUM_ENV_STEPS_SAMPLED_LIFETIME: sampled_env_steps,
                NUM_AGENT_STEPS_SAMPLED_LIFETIME: sampled_agent_steps,
                "rllib_async_updated_modules": tuple(learner_batch.policy_batches),
            },
        )
        priority_errors = self._take_priority_errors(
            learner_results,
            learner_batch,
            enabled=return_priority_errors,
        )
        self._learner_updates += 1
        published_weights = (
            self.maybe_publish_weights() if self._publish_updates else None
        )
        return LearnerUpdateResult(
            performed=True,
            learner_updates=self._learner_updates,
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
            learner_results=learner_results,
            published_weights=published_weights,
            priority_errors=priority_errors,
        )

    def _take_priority_errors(
        self,
        results: list[dict[str, Any]],
        batch: MultiAgentBatch,
        *,
        enabled: bool,
    ) -> dict[str, np.ndarray] | None:
        """Consume RLlib's item-series signal once, without retaining it in reports.

        DQN uses absolute TD error, C51 uses its distributional cross entropy,
        and SAC uses the mean absolute error of its critics. No loss is changed.
        """

        pieces: dict[str, list[np.ndarray]] = defaultdict(list)
        for result in results:
            for module_id, module_result in result.items():
                if not isinstance(module_result, dict):
                    continue
                signal = module_result.pop(TD_ERROR_KEY, None)
                if signal is not None and enabled and module_id in batch.policy_batches:
                    pieces[module_id].extend(convert_to_numpy(signal.peek()))
        if not enabled:
            return None
        if set(pieces) != set(batch.policy_batches):
            raise self._adapter_error(
                "learner priority signals do not match batch modules"
            )
        errors = {}
        for module_id, chunks in pieces.items():
            if not chunks:
                raise self._adapter_error("learner priority signal is empty")
            values = np.concatenate(chunks, axis=0).reshape(-1)
            if (
                values.shape != (batch.policy_batches[module_id].count,)
                or not np.isfinite(values).all()
            ):
                raise self._adapter_error(
                    "learner priority signal is not aligned with loss rows"
                )
            errors[module_id] = values
        return errors

    def _skipped_update_result(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
    ) -> LearnerUpdateResult:
        return LearnerUpdateResult(
            performed=False,
            learner_updates=self._learner_updates,
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
            learner_results=None,
            published_weights=None,
        )

    def maybe_publish_weights(
        self,
        *,
        force: bool = False,
    ) -> WeightsDescriptor | None:
        """Publish current inference weights when the configured interval is due."""

        self._require_open()
        updates_since_publication = self._learner_updates - self._last_published_update
        if updates_since_publication == 0:
            return None
        if not force and updates_since_publication < self._publication_interval_updates:
            return None

        module_state = self._get_inference_module_state()
        module_versions = {
            module_id: version + 1
            for module_id, version in self._module_versions.items()
        }
        descriptor = self._new_descriptor(
            module_state,
            module_versions=module_versions,
            learner_updates=self._learner_updates,
        )
        self._module_versions = module_versions
        self._last_published_update = self._learner_updates
        self._latest_weights = descriptor
        return self._latest_weights

    def get_published_weights(self) -> WeightsDescriptor:
        """Return the most recently published, immutable-by-contract weights."""

        self._require_open()
        return self._latest_weights

    def get_inference_state(self) -> Mapping[str, Any]:
        """Copy current inference state for an atomic multi-learner bundle."""

        self._require_open()
        return self._get_inference_module_state()

    def get_state(self) -> dict[str, Any]:
        """Return a complete in-memory member checkpoint."""

        self._require_open()
        learner_group_state = self._learner_group.get_state(
            not_components=(f"{COMPONENT_LEARNER}/{COMPONENT_METRICS_LOGGER}"),
        )
        torch_rng_state = torch.get_rng_state().clone()
        torch_cuda_rng_states = tuple(
            state.clone()
            for state in (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
            )
        )
        return {
            "state_version": SAC_ADAPTER_STATE_VERSION,
            "member_id": self._member_id,
            "publication_interval_updates": self._publication_interval_updates,
            "publish_updates": self._publish_updates,
            "learning_starts": self._learning_starts,
            "learning_starts_satisfied": self._learning_starts_satisfied,
            "config_contract": copy.deepcopy(self._config_contract),
            "module_config_contract": copy.deepcopy(self._module_config_contract),
            "spaces_fingerprint": self._spaces_fingerprint,
            "learner_updates": self._learner_updates,
            "sampled_env_steps": self._sampled_env_steps,
            "sampled_agent_steps": self._sampled_agent_steps,
            "last_published_update": self._last_published_update,
            "module_versions": dict(self._module_versions),
            "latest_weights": copy.deepcopy(self._latest_weights),
            "torch_rng_state": torch_rng_state,
            "torch_cuda_rng_states": torch_cuda_rng_states,
            _LEARNER_GROUP_STATE: learner_group_state,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore one validated member checkpoint into this adapter."""

        self._require_open()
        if not isinstance(state, Mapping):
            raise ValueError("adapter state must be a mapping")
        if state.get("state_version") != SAC_ADAPTER_STATE_VERSION:
            raise ValueError(f"unsupported {self._algorithm} adapter state version")
        if state.get("member_id") != self._member_id:
            raise ValueError("checkpoint member_id does not match this adapter")
        if (
            state.get("publication_interval_updates")
            != self._publication_interval_updates
        ):
            raise ValueError("checkpoint publication interval does not match")
        if state.get("publish_updates", True) != self._publish_updates:
            raise ValueError("checkpoint publication ownership does not match")
        if state.get("learning_starts") != self._learning_starts:
            raise ValueError("checkpoint learning-start threshold does not match")
        learning_starts_satisfied = state.get("learning_starts_satisfied")
        if learning_starts_satisfied is None:
            sampled_steps = (
                state.get("sampled_agent_steps")
                if self._config.count_steps_by == "agent_steps"
                else state.get("sampled_env_steps")
            )
            learning_starts_satisfied = (
                isinstance(sampled_steps, int)
                and not isinstance(sampled_steps, bool)
                and sampled_steps >= self._learning_starts
            )
        if not isinstance(learning_starts_satisfied, bool):
            raise ValueError("learning_starts_satisfied must be a bool")
        if state.get("config_contract") != self._config_contract:
            raise ValueError(
                f"checkpoint {self._algorithm} configuration does not match"
            )
        if (self._algorithm != "SAC" and "module_config_contract" not in state) or (
            "module_config_contract" in state
            and state["module_config_contract"] != self._module_config_contract
        ):
            raise ValueError("checkpoint module configuration does not match")
        if state.get("spaces_fingerprint") != self._spaces_fingerprint:
            raise ValueError("checkpoint observation/action spaces do not match")

        learner_updates = self._state_counter(state, "learner_updates")
        sampled_env_steps = self._state_counter(state, "sampled_env_steps")
        sampled_agent_steps = self._state_counter(state, "sampled_agent_steps")
        last_published_update = self._state_counter(
            state,
            "last_published_update",
        )
        if last_published_update > learner_updates:
            raise ValueError("last_published_update exceeds learner_updates")

        module_versions_raw = state.get("module_versions")
        if not isinstance(module_versions_raw, Mapping):
            raise ValueError("module_versions must be a mapping")
        module_versions = dict(module_versions_raw)
        if set(module_versions) != set(self._module_versions):
            raise ValueError("checkpoint module IDs do not match this adapter")
        for module_id, version in module_versions.items():
            self._validate_counter(f"module version {module_id!r}", version)

        latest_weights = state.get("latest_weights")
        if not isinstance(latest_weights, WeightsDescriptor):
            raise ValueError("latest_weights must be a WeightsDescriptor")
        if latest_weights.member_id != self._member_id:
            raise ValueError("latest_weights member_id does not match")
        if dict(latest_weights.module_versions) != module_versions:
            raise ValueError("latest_weights module versions do not match")
        if latest_weights.learner_updates != last_published_update:
            raise ValueError("latest_weights learner update does not match")
        if not isinstance(latest_weights.state, Mapping) or set(
            latest_weights.state
        ) != set(self._module_versions):
            raise ValueError("latest_weights state module IDs do not match")

        learner_group_state = state.get(_LEARNER_GROUP_STATE)
        if not isinstance(learner_group_state, Mapping):
            raise ValueError("learner_group state must be a mapping")
        learner_state = learner_group_state.get(COMPONENT_LEARNER)
        if not isinstance(learner_state, Mapping):
            raise ValueError("learner_group must contain learner state")
        if learner_state.get(WEIGHTS_SEQ_NO) != learner_updates:
            raise ValueError("learner update counter does not match RLlib state")
        target_update_state = learner_state.get(SAC_TARGET_UPDATE_STATE)
        if not isinstance(target_update_state, Mapping) or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > sampled_env_steps
            for value in target_update_state.values()
        ):
            raise ValueError("target update state exceeds sampled environment steps")
        torch_rng_state = self._validated_rng_state(
            state.get("torch_rng_state"),
            name="torch_rng_state",
            expected_numel=torch.get_rng_state().numel(),
        )
        cuda_rng_states_raw = state.get("torch_cuda_rng_states")
        if not isinstance(cuda_rng_states_raw, tuple):
            raise ValueError("torch_cuda_rng_states must be a tuple")
        current_cuda_rng_states = (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
        )
        if len(cuda_rng_states_raw) != len(current_cuda_rng_states):
            raise ValueError("checkpoint CUDA RNG device count does not match")
        cuda_rng_states = tuple(
            self._validated_rng_state(
                value,
                name="torch_cuda_rng_states",
                expected_numel=current.numel(),
            )
            for value, current in zip(
                cuda_rng_states_raw,
                current_cuda_rng_states,
                strict=True,
            )
        )
        learner_group_state_copy = copy.deepcopy(dict(learner_group_state))
        latest_weights_copy = copy.deepcopy(latest_weights)

        self._learner_group.set_state(learner_group_state_copy)
        self._learner_updates = learner_updates
        self._sampled_env_steps = sampled_env_steps
        self._sampled_agent_steps = sampled_agent_steps
        self._learning_starts_satisfied = learning_starts_satisfied
        self._last_published_update = last_published_update
        self._module_versions = module_versions
        self._latest_weights = latest_weights_copy
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(list(cuda_rng_states))

    def close(self) -> None:
        if self._closed:
            return
        self._learner_group.shutdown()
        self._closed = True

    def __enter__(self) -> _LocalOffPolicyLearnerAdapter:
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _new_descriptor(
        self,
        state: object,
        *,
        module_versions: Mapping[str, int],
        learner_updates: int,
    ) -> WeightsDescriptor:
        return WeightsDescriptor(
            member_id=self._member_id,
            module_versions=module_versions,
            learner_updates=learner_updates,
            published_at_monotonic=time.monotonic(),
            state=state,
        )

    def export_pbt_state(self) -> PBTModelState:
        """Export model state without optimizer, RNG, counters, or member identity."""

        self._require_open()
        learner_group_state = self._learner_group.get_state(
            components=f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
        )
        learner_state = learner_group_state.get(COMPONENT_LEARNER)
        if not isinstance(learner_state, Mapping):
            raise self._adapter_error("learner state is missing")
        module_state = learner_state.get(COMPONENT_RL_MODULE)
        learner = getattr(self._learner_group, "_learner", None)
        temperature_state = (
            learner.get_temperature_state() if self._algorithm == "SAC" else {}
        )
        if not isinstance(module_state, Mapping):
            raise self._adapter_error("learner PBT state is incomplete")
        return PBTModelState(
            state_version=PBT_MODEL_STATE_VERSION,
            spaces_fingerprint=self._spaces_fingerprint,
            model_fingerprint=self._fingerprint_model_state(module_state),
            module_state=module_state,
            temperature_state=temperature_state,
            algorithm=self._algorithm,
            config_contract=self._pbt_config_contract(),
        )

    def _pbt_config_contract(self) -> dict[str, Any]:
        return {
            "algorithm": self._algorithm,
            "learner": {
                key: value
                for key, value in self._config_contract.items()
                if key not in {"lr", "actor_lr", "critic_lr", "alpha_lr"}
            },
            "modules": copy.deepcopy(self._module_config_contract),
        }

    def validate_pbt_state(
        self, state: PBTModelState, *, require_fresh: bool = False
    ) -> None:
        """Validate a transfer without changing a live learner or its clocks."""

        self._require_open()
        if not isinstance(state, PBTModelState):
            raise TypeError("state must be PBTModelState")
        if state.state_version != PBT_MODEL_STATE_VERSION:
            raise ValueError("unsupported PBT model state version")
        if state.algorithm != self._algorithm:
            raise ValueError("PBT algorithms do not match")
        if state.config_contract is not None:
            if state.config_contract != self._pbt_config_contract():
                raise ValueError("PBT algorithm/model configuration does not match")
        elif self._algorithm != "SAC":
            raise ValueError("DQN PBT requires a complete configuration contract")
        if state.spaces_fingerprint != self._spaces_fingerprint:
            raise ValueError("PBT observation/action spaces do not match")
        if require_fresh and any(
            value != 0
            for value in (
                self._learner_updates,
                self._sampled_env_steps,
                self._sampled_agent_steps,
                self._last_published_update,
            )
        ):
            raise self._adapter_error("PBT state requires a fresh learner")
        if state.model_fingerprint != self._fingerprint_model_state(state.module_state):
            raise ValueError("PBT model state fingerprint is invalid")

        learner_group_state = self._learner_group.get_state(
            components=f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
        )
        learner_state = learner_group_state.get(COMPONENT_LEARNER)
        if not isinstance(learner_state, Mapping):
            raise self._adapter_error("learner state is missing")
        current_module_state = learner_state.get(COMPONENT_RL_MODULE)
        if not isinstance(current_module_state, Mapping):
            raise self._adapter_error("learner model state is missing")
        if state.model_fingerprint != self._fingerprint_model_state(
            current_module_state
        ):
            raise ValueError("PBT model structure does not match")
        if self._algorithm == "SAC":
            self._learner_group._learner._validate_temperatures(state.temperature_state)
        elif state.temperature_state:
            raise ValueError("DQN PBT state cannot contain SAC temperature")

    def load_pbt_state(self, state: PBTModelState) -> WeightsDescriptor:
        """Import all model/target state; optimizer, clocks and RNG stay fresh."""

        self.validate_pbt_state(state, require_fresh=True)

        imported_state = {
            COMPONENT_LEARNER: {
                COMPONENT_RL_MODULE: copy.deepcopy(state.module_state),
            }
        }
        if self._algorithm == "SAC":
            imported_state[COMPONENT_LEARNER][SAC_TEMPERATURE_STATE] = copy.deepcopy(
                state.temperature_state
            )
        self._learner_group.set_state(imported_state)

        module_versions = {
            module_id: version + 1
            for module_id, version in self._module_versions.items()
        }
        descriptor = self._new_descriptor(
            self._get_inference_module_state(),
            module_versions=module_versions,
            learner_updates=0,
        )
        self._module_versions = module_versions
        self._latest_weights = descriptor
        return copy.deepcopy(descriptor)

    def _get_inference_module_state(self) -> dict[str, Any]:
        state = self._learner_group.get_state(
            components=f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
            inference_only=True,
        )
        return copy.deepcopy(state[COMPONENT_LEARNER][COMPONENT_RL_MODULE])

    def _require_open(self) -> None:
        if self._closed:
            raise self._adapter_error(f"{self._algorithm} learner adapter is closed")

    @staticmethod
    def _fingerprint_spaces(
        spaces: Mapping[str, tuple[gym.Space, gym.Space]],
    ) -> str:
        if any(not isinstance(key, str) or not key for key in spaces):
            raise ValueError("space keys must be non-empty strings")
        try:
            signature = tuple(
                (
                    key,
                    _LocalOffPolicyLearnerAdapter._space_signature(spaces[key][0]),
                    _LocalOffPolicyLearnerAdapter._space_signature(spaces[key][1]),
                )
                for key in sorted(spaces)
            )
        except (IndexError, TypeError) as error:
            raise ValueError(
                "spaces must map IDs to observation/action pairs"
            ) from error
        payload = pickle.dumps(signature, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _fingerprint_model_state(cls, state: Mapping[str, Any]) -> str:
        if not isinstance(state, Mapping) or not state:
            raise ValueError("model state must be a non-empty mapping")
        payload = pickle.dumps(
            cls._model_state_signature(state),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _model_state_signature(cls, value: object) -> object:
        if torch.is_tensor(value):
            return ("torch", tuple(value.shape), str(value.dtype))
        if isinstance(value, np.ndarray):
            return ("numpy", tuple(value.shape), value.dtype.str)
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("model state keys must be strings")
            return (
                "mapping",
                tuple(
                    (key, cls._model_state_signature(value[key]))
                    for key in sorted(value)
                ),
            )
        if isinstance(value, list | tuple):
            return (
                type(value).__name__,
                tuple(cls._model_state_signature(item) for item in value),
            )
        if isinstance(value, np.generic):
            return ("numpy_scalar", value.dtype.str)
        if value is None or isinstance(value, bool | int | float | str):
            return (type(value).__name__, value)
        raise ValueError(f"unsupported model state value {type(value).__name__!r}")

    @staticmethod
    def _space_signature(space: gym.Space) -> object:
        if isinstance(space, gym.spaces.Box):
            return (
                "box",
                tuple(space.shape),
                np.dtype(space.dtype).str,
                np.ascontiguousarray(space.low).tobytes(),
                np.ascontiguousarray(space.high).tobytes(),
            )
        if isinstance(space, gym.spaces.Discrete):
            return (
                "discrete",
                int(space.n),
                int(space.start),
                np.dtype(space.dtype).str,
            )
        if isinstance(space, gym.spaces.MultiDiscrete):
            return (
                "multi_discrete",
                tuple(space.shape),
                np.dtype(space.dtype).str,
                np.ascontiguousarray(space.nvec).tobytes(),
                np.ascontiguousarray(space.start).tobytes(),
            )
        if isinstance(space, gym.spaces.MultiBinary):
            return (
                "multi_binary",
                tuple(space.shape),
                np.dtype(space.dtype).str,
            )
        if isinstance(space, gym.spaces.Dict):
            return (
                "dict",
                tuple(
                    (
                        key,
                        _LocalOffPolicyLearnerAdapter._space_signature(child),
                    )
                    for key, child in space.spaces.items()
                ),
            )
        if isinstance(space, gym.spaces.Tuple):
            return (
                "tuple",
                tuple(
                    _LocalOffPolicyLearnerAdapter._space_signature(child)
                    for child in space.spaces
                ),
            )
        raise ValueError(f"unsupported Gymnasium space type {type(space).__name__!r}")

    @staticmethod
    def _validated_rng_state(
        value: object,
        *,
        name: str,
        expected_numel: int,
    ) -> torch.Tensor:
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() != expected_numel
        ):
            raise ValueError(f"{name} must contain CPU byte tensors")
        return value.clone()

    @staticmethod
    def _validate_counter(
        name: str,
        value: object,
        *,
        previous: int | None = None,
    ) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        if previous is not None and value < previous:
            raise ValueError(f"{name} must be monotonic")

    @classmethod
    def _state_counter(cls, state: Mapping[str, Any], name: str) -> int:
        value = state.get(name)
        cls._validate_counter(name, value)
        assert isinstance(value, int)
        return value


class SACLearnerAdapter(_LocalOffPolicyLearnerAdapter):
    """Own one local RLlib SAC LearnerGroup and its runtime counters."""

    def __init__(
        self,
        config: SACConfig,
        *,
        spaces: Mapping[str, tuple[gym.Space, gym.Space]],
        member_id: str,
        publication_interval_updates: int,
        learning_starts_satisfied: bool = False,
        validate_startup: bool = False,
        publish_updates: bool = True,
    ) -> None:
        for module_id, (observation_space, action_space) in spaces.items():
            if (
                isinstance(action_space, gym.spaces.Discrete)
                and isinstance(observation_space, gym.spaces.Dict)
                and "action_mask" in observation_space.spaces
            ):
                module_config = config.get_config_for_module(module_id)
                if (
                    not isinstance(module_config.target_entropy, int | float)
                    or isinstance(module_config.target_entropy, bool)
                    or not np.isfinite(module_config.target_entropy)
                ):
                    raise ValueError(
                        "masked discrete SAC requires an explicit finite target_entropy"
                    )
                if not module_config.twin_q:
                    raise ValueError("masked discrete SAC requires twin_q=True")
        super().__init__(
            config,
            spaces=spaces,
            member_id=member_id,
            publication_interval_updates=publication_interval_updates,
            learning_starts_satisfied=learning_starts_satisfied,
            validate_startup=validate_startup,
            publish_updates=publish_updates,
        )

    @staticmethod
    def _build_batch(batch: Mapping[str, Any], *, validate: bool) -> MultiAgentBatch:
        return build_rllib_sac_batch(batch, validate=validate)

    @staticmethod
    def _build_multi_module_batch(
        batches: Mapping[str, Mapping[str, Any]], *, validate: bool
    ) -> MultiAgentBatch:
        return build_rllib_multi_module_sac_batch(batches, validate=validate)
