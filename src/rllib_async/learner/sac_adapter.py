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
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    WEIGHTS_SEQ_NO,
)

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
_OPTIONAL_BATCH_COLUMNS = ("n_step", "weights")
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

    def __post_init__(self) -> None:
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
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping")
            object.__setattr__(self, name, copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class SACUpdateResult:
    """Outcome of one adapter update request."""

    performed: bool
    learner_updates: int
    sampled_env_steps: int
    sampled_agent_steps: int
    learner_results: list[dict[str, Any]] | None
    published_weights: WeightsDescriptor | None


class CheckpointableSACTorchLearner(SACTorchLearner):
    """Add the SAC state omitted by RLlib 2.56.1 checkpoints.

    SAC loss, gradients, optimizers, and target updates remain implemented by
    ``SACTorchLearner``. This subclass changes state serialization only.
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


def _build_sac_sample_batch(batch: Mapping[str, Any]) -> SampleBatch:
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
    graph_observations = isinstance(observations, Mapping)
    if graph_observations != isinstance(next_observations, Mapping):
        raise SACBatchError("'obs' and 'new_obs' must use the same representation")
    if graph_observations:
        try:
            packed_observations = validate_packed_graph_batch(observations)
            packed_next_observations = validate_packed_graph_batch(next_observations)
        except ValueError as error:
            raise SACBatchError(str(error)) from error
        batch_size = len(packed_observations["graph_ptr"]) - 1
        _validate_graph_observation_pair(
            packed_observations,
            packed_next_observations,
        )
        columns[Columns.OBS] = packed_observations
        columns[Columns.NEXT_OBS] = packed_next_observations
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

    return SampleBatch(columns)


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


def build_rllib_sac_batch(batch: Mapping[str, Any]) -> MultiAgentBatch:
    """Validate one NumPy transition batch and build RLlib's exact SAC input."""

    sample_batch = _build_sac_sample_batch(batch)
    return MultiAgentBatch(
        {DEFAULT_MODULE_ID: sample_batch},
        env_steps=sample_batch.count,
    )


def build_rllib_multi_module_sac_batch(
    batches: Mapping[str, Mapping[str, Any]],
) -> MultiAgentBatch:
    """Validate independent batches for one heterogeneous MultiRLModule."""

    if not isinstance(batches, Mapping) or not batches:
        raise SACBatchError("multi-module SAC batches must be a non-empty mapping")
    if any(not isinstance(module_id, str) or not module_id for module_id in batches):
        raise SACBatchError("module IDs must be non-empty strings")
    sample_batches = {
        module_id: _build_sac_sample_batch(batch)
        for module_id, batch in batches.items()
    }
    return MultiAgentBatch(
        sample_batches,
        env_steps=max(batch.count for batch in sample_batches.values()),
    )


class SACLearnerAdapter:
    """Own one local RLlib SAC LearnerGroup and its runtime counters."""

    def __init__(
        self,
        config: SACConfig,
        *,
        spaces: Mapping[str, tuple[gym.Space, gym.Space]],
        member_id: str,
        publication_interval_updates: int,
        learning_starts_satisfied: bool = False,
    ) -> None:
        if not isinstance(config, SACConfig):
            raise TypeError("config must be an SACConfig")
        if not isinstance(member_id, str) or not member_id:
            raise ValueError("member_id must be a non-empty string")
        if (
            not isinstance(publication_interval_updates, int)
            or isinstance(publication_interval_updates, bool)
            or publication_interval_updates < 1
        ):
            raise ValueError("publication_interval_updates must be positive")
        if config.framework_str != "torch":
            raise ValueError("SACLearnerAdapter supports PyTorch only")
        if not config.enable_rl_module_and_learner:
            raise ValueError(
                "SACLearnerAdapter requires the RLModule/Learner API stack"
            )
        if not config.enable_env_runner_and_connector_v2:
            raise ValueError("SACLearnerAdapter requires ConnectorV2")
        if config.num_learners != 0:
            raise ValueError("SACLearnerAdapter requires one local RLlib learner")
        if not isinstance(learning_starts_satisfied, bool):
            raise TypeError("learning_starts_satisfied must be a bool")
        resolved_learner_class = config.learner_class
        if resolved_learner_class not in {
            None,
            SACTorchLearner,
            CheckpointableSACTorchLearner,
        }:
            raise ValueError("custom SAC learner classes are not supported in v0.1")

        adapter_config = config.copy(copy_frozen=False)
        adapter_config.learners(learner_class=CheckpointableSACTorchLearner)

        self._config = adapter_config
        self._member_id = member_id
        self._publication_interval_updates = publication_interval_updates
        self._learning_starts = adapter_config.num_steps_sampled_before_learning_starts
        self._learning_starts_satisfied = learning_starts_satisfied
        self._config_contract = {
            name: copy.deepcopy(getattr(adapter_config, name))
            for name in _CONFIG_CONTRACT_FIELDS
        }
        try:
            pickle.dumps(
                self._config_contract,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (pickle.PickleError, TypeError) as error:
            raise ValueError(
                "SAC configuration contract must be pickle-safe"
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
            raise SACLearnerAdapterError(
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
    ) -> SACUpdateResult:
        """Perform at most one RLlib update at absolute sampled-step counters."""

        self._require_open()
        if set(self._module_versions) != {DEFAULT_MODULE_ID}:
            raise SACBatchError(
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
        return self._perform_update(
            build_rllib_sac_batch(batch),
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
        )

    def update_modules(
        self,
        batches: Mapping[str, Mapping[str, Any]],
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
    ) -> SACUpdateResult:
        """Update every configured SAC module from its own sampled batch view."""

        self._require_open()
        if not isinstance(batches, Mapping) or not batches:
            raise SACBatchError("multi-module SAC batches must be a non-empty mapping")
        if set(batches) != set(self._module_versions):
            raise SACBatchError(
                "multi-module batch IDs must match the learner module IDs"
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
        return self._perform_update(
            build_rllib_multi_module_sac_batch(batches),
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
        )

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
    ) -> SACUpdateResult:
        learner_results = self._learner_group.update(
            batch=learner_batch,
            timesteps={
                NUM_ENV_STEPS_SAMPLED_LIFETIME: sampled_env_steps,
                NUM_AGENT_STEPS_SAMPLED_LIFETIME: sampled_agent_steps,
            },
        )
        self._learner_updates += 1
        published_weights = self.maybe_publish_weights()
        return SACUpdateResult(
            performed=True,
            learner_updates=self._learner_updates,
            sampled_env_steps=sampled_env_steps,
            sampled_agent_steps=sampled_agent_steps,
            learner_results=learner_results,
            published_weights=published_weights,
        )

    def _skipped_update_result(
        self,
        *,
        sampled_env_steps: int,
        sampled_agent_steps: int,
    ) -> SACUpdateResult:
        return SACUpdateResult(
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
        return copy.deepcopy(self._latest_weights)

    def get_published_weights(self) -> WeightsDescriptor:
        """Return the most recently published, immutable-by-contract weights."""

        self._require_open()
        return copy.deepcopy(self._latest_weights)

    def export_pbt_state(self) -> PBTModelState:
        """Export model state without optimizer, RNG, counters, or member identity."""

        self._require_open()
        learner_group_state = self._learner_group.get_state(
            components=f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
        )
        learner_state = learner_group_state.get(COMPONENT_LEARNER)
        if not isinstance(learner_state, Mapping):
            raise SACLearnerAdapterError("learner state is missing")
        module_state = learner_state.get(COMPONENT_RL_MODULE)
        learner = getattr(self._learner_group, "_learner", None)
        if not isinstance(learner, CheckpointableSACTorchLearner):
            raise SACLearnerAdapterError("checkpointable SAC learner is missing")
        temperature_state = learner.get_temperature_state()
        if not isinstance(module_state, Mapping):
            raise SACLearnerAdapterError("learner PBT state is incomplete")
        return PBTModelState(
            state_version=PBT_MODEL_STATE_VERSION,
            spaces_fingerprint=self._spaces_fingerprint,
            model_fingerprint=self._fingerprint_model_state(module_state),
            module_state=module_state,
            temperature_state=temperature_state,
        )

    def load_pbt_state(self, state: PBTModelState) -> WeightsDescriptor:
        """Import model state into a fresh learner and publish target-owned weights."""

        self._require_open()
        if not isinstance(state, PBTModelState):
            raise TypeError("state must be PBTModelState")
        if state.spaces_fingerprint != self._spaces_fingerprint:
            raise ValueError("PBT observation/action spaces do not match")
        if any(
            value != 0
            for value in (
                self._learner_updates,
                self._sampled_env_steps,
                self._sampled_agent_steps,
                self._last_published_update,
            )
        ):
            raise SACLearnerAdapterError("PBT state requires a fresh learner")
        if state.model_fingerprint != self._fingerprint_model_state(state.module_state):
            raise ValueError("PBT model state fingerprint is invalid")

        learner_group_state = self._learner_group.get_state(
            components=f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
        )
        learner_state = learner_group_state.get(COMPONENT_LEARNER)
        if not isinstance(learner_state, Mapping):
            raise SACLearnerAdapterError("fresh learner state is missing")
        current_module_state = learner_state.get(COMPONENT_RL_MODULE)
        if not isinstance(current_module_state, Mapping):
            raise SACLearnerAdapterError("fresh learner model state is missing")
        if state.model_fingerprint != self._fingerprint_model_state(
            current_module_state
        ):
            raise ValueError("PBT model structure does not match")

        imported_state = {
            COMPONENT_LEARNER: {
                COMPONENT_RL_MODULE: copy.deepcopy(state.module_state),
                SAC_TEMPERATURE_STATE: copy.deepcopy(state.temperature_state),
            }
        }
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
            "learning_starts": self._learning_starts,
            "learning_starts_satisfied": self._learning_starts_satisfied,
            "config_contract": copy.deepcopy(self._config_contract),
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
            raise ValueError("unsupported SAC adapter state version")
        if state.get("member_id") != self._member_id:
            raise ValueError("checkpoint member_id does not match this adapter")
        if (
            state.get("publication_interval_updates")
            != self._publication_interval_updates
        ):
            raise ValueError("checkpoint publication interval does not match")
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
            raise ValueError("checkpoint SAC configuration does not match")
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

    def __enter__(self) -> SACLearnerAdapter:
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
            state=copy.deepcopy(state),
        )

    def _get_inference_module_state(self) -> dict[str, Any]:
        state = self._learner_group.get_state(
            components=f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
            inference_only=True,
        )
        return copy.deepcopy(state[COMPONENT_LEARNER][COMPONENT_RL_MODULE])

    def _require_open(self) -> None:
        if self._closed:
            raise SACLearnerAdapterError("SAC learner adapter is closed")

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
                    SACLearnerAdapter._space_signature(spaces[key][0]),
                    SACLearnerAdapter._space_signature(spaces[key][1]),
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
                        SACLearnerAdapter._space_signature(child),
                    )
                    for key, child in space.spaces.items()
                ),
            )
        if isinstance(space, gym.spaces.Tuple):
            return (
                "tuple",
                tuple(
                    SACLearnerAdapter._space_signature(child) for child in space.spaces
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
