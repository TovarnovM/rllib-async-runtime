"""Trusted-local graph observation validation and episode codec."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from ray.rllib.core.columns import Columns

from rllib_async.protocols.episodes import (
    EncodedModuleTransition,
    EpisodeEnvelope,
    EpisodeValidationError,
    MultiModuleEpisodeCodec,
    MultiModuleEpisodePayload,
    MultiModuleTransition,
)

NODE_FEATURES = "node_features"
EDGE_INDEX = "edge_index"
EDGE_FEATURES = "edge_features"
CONTROLLED_NODE = "controlled_node"
ACTION_MASK = "action_mask"
NODE_COUNT = "node_count"
EDGE_COUNT = "edge_count"

_GRAPH_KEYS = frozenset(
    {
        NODE_FEATURES,
        EDGE_INDEX,
        EDGE_FEATURES,
        CONTROLLED_NODE,
        ACTION_MASK,
        NODE_COUNT,
        EDGE_COUNT,
    }
)
_REQUIRED_GRAPH_KEYS = frozenset(
    {
        NODE_FEATURES,
        EDGE_INDEX,
        CONTROLLED_NODE,
    }
)
_TRANSITION_KEYS = frozenset(
    {
        Columns.OBS,
        Columns.NEXT_OBS,
        Columns.ACTIONS,
        Columns.REWARDS,
        Columns.TERMINATEDS,
        Columns.TRUNCATEDS,
    }
)


@dataclass(frozen=True, slots=True)
class GraphEpisodePayload(MultiModuleEpisodePayload):
    """Codec-distinct immutable payload containing normalized ego-graphs."""

    node_feature_dim: int
    edge_feature_dim: int
    action_mask_dim: int

    def __post_init__(self) -> None:
        MultiModuleEpisodePayload.__post_init__(self)
        _positive_dimension(
            self.node_feature_dim,
            name="node_feature_dim",
        )
        _optional_dimension(
            self.edge_feature_dim,
            name="edge_feature_dim",
        )
        _optional_dimension(
            self.action_mask_dim,
            name="action_mask_dim",
        )


def normalize_graph_observation(observation: object) -> dict[str, Any]:
    """Validate one graph and strip optional fixed-space transport padding."""

    if not isinstance(observation, Mapping):
        raise EpisodeValidationError("graph observation must be a mapping")
    keys = set(observation)
    missing = _REQUIRED_GRAPH_KEYS - keys
    if missing:
        raise EpisodeValidationError(
            f"graph observation is missing keys {sorted(missing)!r}"
        )
    extra = keys - _GRAPH_KEYS
    if extra:
        raise EpisodeValidationError(
            f"graph observation has unsupported keys {sorted(extra)!r}"
        )
    has_node_count = NODE_COUNT in observation
    has_edge_count = EDGE_COUNT in observation
    if has_node_count != has_edge_count:
        raise EpisodeValidationError(
            "padded graph observations require node_count and edge_count together"
        )

    node_features = _real_array(
        observation[NODE_FEATURES],
        name=NODE_FEATURES,
        ndim=2,
    )
    if node_features.shape[0] < 1 or node_features.shape[1] < 1:
        raise EpisodeValidationError(
            "node_features must contain at least one non-empty node"
        )
    edge_index = np.asarray(observation[EDGE_INDEX])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise EpisodeValidationError("edge_index must have shape (2, edge_count)")
    if edge_index.dtype.kind not in "iu":
        raise EpisodeValidationError("edge_index must contain integers")

    node_count = (
        _integer_scalar(observation[NODE_COUNT], name=NODE_COUNT)
        if has_node_count
        else node_features.shape[0]
    )
    edge_count = (
        _integer_scalar(observation[EDGE_COUNT], name=EDGE_COUNT)
        if has_edge_count
        else edge_index.shape[1]
    )
    if node_count < 1 or node_count > node_features.shape[0]:
        raise EpisodeValidationError("node_count is outside node_features padding")
    if edge_count < 0 or edge_count > edge_index.shape[1]:
        raise EpisodeValidationError("edge_count is outside edge_index padding")

    node_features = _float32_array(
        node_features[:node_count],
        name=NODE_FEATURES,
    )
    edge_index = np.ascontiguousarray(
        edge_index[:, :edge_count],
        dtype=np.int64,
    )
    if edge_count and (np.any(edge_index < 0) or np.any(edge_index >= node_count)):
        raise EpisodeValidationError(
            "edge_index endpoints must reference nodes in this graph"
        )
    controlled_node = _integer_scalar(
        observation[CONTROLLED_NODE],
        name=CONTROLLED_NODE,
    )
    if controlled_node < 0 or controlled_node >= node_count:
        raise EpisodeValidationError(
            "controlled_node must reference a node in this graph"
        )

    normalized: dict[str, Any] = {
        NODE_FEATURES: node_features,
        EDGE_INDEX: edge_index,
        CONTROLLED_NODE: controlled_node,
    }
    edge_features_raw = observation.get(EDGE_FEATURES)
    if edge_features_raw is not None:
        edge_features = _real_array(
            edge_features_raw,
            name=EDGE_FEATURES,
            ndim=2,
        )
        aligned_edge_count = (
            edge_features.shape[0] >= edge_count
            if has_edge_count
            else edge_features.shape[0] == edge_count
        )
        if not aligned_edge_count or edge_features.shape[1] < 1:
            raise EpisodeValidationError(
                "edge_features must align with edge_index and have features"
            )
        normalized[EDGE_FEATURES] = _float32_array(
            edge_features[:edge_count],
            name=EDGE_FEATURES,
        )

    action_mask_raw = observation.get(ACTION_MASK)
    if action_mask_raw is not None:
        action_mask = _real_array(
            action_mask_raw,
            name=ACTION_MASK,
            ndim=1,
        )
        if (
            not action_mask.size
            or not np.all((action_mask == 0) | (action_mask == 1))
            or not np.any(action_mask > 0)
        ):
            raise EpisodeValidationError(
                "action_mask must be binary and enable at least one action"
            )
        normalized[ACTION_MASK] = np.ascontiguousarray(
            action_mask,
            dtype=np.float32,
        )
    return normalized


class GraphEpisodeCodec(MultiModuleEpisodeCodec):
    """Graph-aware specialization of the sparse multi-module pickle codec."""

    schema_version = 1

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int = 0,
        action_mask_dim: int = 0,
    ) -> None:
        self._node_feature_dim = _positive_dimension(
            node_feature_dim,
            name="node_feature_dim",
        )
        self._edge_feature_dim = _optional_dimension(
            edge_feature_dim,
            name="edge_feature_dim",
        )
        self._action_mask_dim = _optional_dimension(
            action_mask_dim,
            name="action_mask_dim",
        )
        self.codec_id = (
            "graph-multi-module-pickle-v1"
            f"-n{self._node_feature_dim}"
            f"-e{self._edge_feature_dim}"
            f"-a{self._action_mask_dim}"
        )

    def encode(
        self,
        transitions: Iterable[MultiModuleTransition],
    ) -> GraphEpisodePayload:
        normalized = [
            self._normalize_transition(transition) for transition in transitions
        ]
        payload = super().encode(normalized)
        return GraphEpisodePayload(
            payload.encoded_module_transitions,
            node_feature_dim=self._node_feature_dim,
            edge_feature_dim=self._edge_feature_dim,
            action_mask_dim=self._action_mask_dim,
        )

    def _payload(self, episode: EpisodeEnvelope) -> GraphEpisodePayload:
        if not isinstance(episode.payload, GraphEpisodePayload):
            raise EpisodeValidationError(
                "GraphEpisodeCodec requires GraphEpisodePayload"
            )
        if (
            episode.payload.node_feature_dim != self._node_feature_dim
            or episode.payload.edge_feature_dim != self._edge_feature_dim
            or episode.payload.action_mask_dim != self._action_mask_dim
        ):
            raise EpisodeValidationError(
                "graph payload feature schema does not match the codec"
            )
        return episode.payload

    def _decode(
        self,
        module_id: str,
        transition: EncodedModuleTransition,
    ) -> MultiModuleTransition:
        decoded = MultiModuleEpisodeCodec._decode(module_id, transition)
        return self._normalize_transition(decoded)

    def _normalize_transition(
        self,
        transition: MultiModuleTransition,
    ) -> MultiModuleTransition:
        if not isinstance(transition, MultiModuleTransition):
            raise EpisodeValidationError(
                "GraphEpisodeCodec requires MultiModuleTransition values"
            )
        if set(transition.data) != _TRANSITION_KEYS:
            raise EpisodeValidationError(
                "graph transitions must contain the exact SAC transition columns"
            )
        data = dict(transition.data)
        data[Columns.OBS] = normalize_graph_observation(data[Columns.OBS])
        data[Columns.NEXT_OBS] = normalize_graph_observation(data[Columns.NEXT_OBS])
        self._validate_feature_schema(data[Columns.OBS])
        self._validate_feature_schema(data[Columns.NEXT_OBS])
        return MultiModuleTransition(
            env_t=transition.env_t,
            agent_t=transition.agent_t,
            agent_id=transition.agent_id,
            module_id=transition.module_id,
            data=data,
        )

    def _validate_feature_schema(self, observation: Mapping[str, Any]) -> None:
        if observation[NODE_FEATURES].shape[1] != self._node_feature_dim:
            raise EpisodeValidationError(
                "node feature dimension does not match the graph codec"
            )
        edge_features = observation.get(EDGE_FEATURES)
        if self._edge_feature_dim:
            if (
                edge_features is None
                or edge_features.shape[1] != self._edge_feature_dim
            ):
                raise EpisodeValidationError(
                    "edge feature dimension does not match the graph codec"
                )
        elif edge_features is not None:
            raise EpisodeValidationError("graph codec does not accept edge_features")
        action_mask = observation.get(ACTION_MASK)
        if self._action_mask_dim:
            if action_mask is None or action_mask.shape != (self._action_mask_dim,):
                raise EpisodeValidationError(
                    "action mask dimension does not match the graph codec"
                )
        elif action_mask is not None:
            raise EpisodeValidationError("graph codec does not accept action_mask")


def _real_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise EpisodeValidationError(f"{name} must be a numeric array") from error
    if array.ndim != ndim or array.dtype.kind not in "biuf":
        raise EpisodeValidationError(f"{name} must be a {ndim}-dimensional real array")
    if not np.isfinite(array).all():
        raise EpisodeValidationError(f"{name} must contain finite values")
    return array


def _float32_array(array: np.ndarray, *, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise EpisodeValidationError(
            f"{name} must contain values representable as float32"
        )
    return normalized


def _integer_scalar(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise EpisodeValidationError(f"{name} must be an integer scalar")
    return int(array)


def _positive_dimension(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_dimension(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
