"""Packed variable-size graph collation for one or more shared modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
from ray.rllib.core.columns import Columns

from rllib_async.gnn.episodes import (
    ACTION_MASK,
    CONTROLLED_NODE,
    EDGE_FEATURES,
    EDGE_INDEX,
    NODE_FEATURES,
    normalize_graph_observation,
)
from rllib_async.protocols.episodes import MultiModuleTransition
from rllib_async.replay.batching import BatchCollationError, FlatBatchCollator

GraphBatch: TypeAlias = dict[str, np.ndarray]
GraphModuleBatch: TypeAlias = dict[str, np.ndarray | GraphBatch]
MultiModuleGraphBatch: TypeAlias = dict[str, GraphModuleBatch]


def pack_graph_observations(
    observations: Sequence[object],
) -> GraphBatch:
    """Pack validated ego-graphs without padding or copied per-graph models."""

    if not observations:
        raise BatchCollationError("graph collation requires at least one observation")
    try:
        graphs = [normalize_graph_observation(item) for item in observations]
    except ValueError as error:
        raise BatchCollationError(str(error)) from error

    node_feature_dims = {graph[NODE_FEATURES].shape[1] for graph in graphs}
    if len(node_feature_dims) != 1:
        raise BatchCollationError("all graphs must share one node feature dimension")
    edge_feature_presence = [EDGE_FEATURES in graph for graph in graphs]
    if len(set(edge_feature_presence)) != 1:
        raise BatchCollationError(
            "edge_features must be present for every graph or for none"
        )
    action_mask_presence = [ACTION_MASK in graph for graph in graphs]
    if len(set(action_mask_presence)) != 1:
        raise BatchCollationError(
            "action_mask must be present for every graph or for none"
        )

    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    edge_feature_parts: list[np.ndarray] = []
    controlled_nodes: list[int] = []
    graph_ptr = [0]
    for graph in graphs:
        node_features = graph[NODE_FEATURES]
        edge_index = graph[EDGE_INDEX]
        offset = graph_ptr[-1]
        node_parts.append(node_features)
        edge_parts.append(edge_index + offset)
        controlled_nodes.append(offset + graph[CONTROLLED_NODE])
        graph_ptr.append(offset + len(node_features))
        if edge_feature_presence[0]:
            edge_feature_parts.append(graph[EDGE_FEATURES])

    packed: GraphBatch = {
        NODE_FEATURES: np.ascontiguousarray(
            np.concatenate(node_parts, axis=0),
            dtype=np.float32,
        ),
        EDGE_INDEX: np.ascontiguousarray(
            np.concatenate(edge_parts, axis=1),
            dtype=np.int64,
        ),
        "graph_ptr": np.ascontiguousarray(graph_ptr, dtype=np.int64),
        CONTROLLED_NODE: np.ascontiguousarray(controlled_nodes, dtype=np.int64),
    }
    if edge_feature_presence[0]:
        feature_dims = {part.shape[1] for part in edge_feature_parts}
        if len(feature_dims) != 1:
            raise BatchCollationError(
                "all graphs must share one edge feature dimension"
            )
        packed[EDGE_FEATURES] = np.ascontiguousarray(
            np.concatenate(edge_feature_parts, axis=0),
            dtype=np.float32,
        )
    if action_mask_presence[0]:
        masks = [graph[ACTION_MASK] for graph in graphs]
        if len({mask.shape for mask in masks}) != 1:
            raise BatchCollationError("all action masks must have the same shape")
        packed[ACTION_MASK] = np.ascontiguousarray(
            np.stack(masks, axis=0),
            dtype=np.float32,
        )
    return validate_packed_graph_batch(packed)


def validate_packed_graph_batch(batch: object) -> GraphBatch:
    """Validate and normalize the NumPy packed-graph learner boundary."""

    if not isinstance(batch, Mapping):
        raise BatchCollationError("packed graph batch must be a mapping")
    required = {NODE_FEATURES, EDGE_INDEX, "graph_ptr", CONTROLLED_NODE}
    missing = required - set(batch)
    if missing:
        raise BatchCollationError(
            f"packed graph batch is missing keys {sorted(missing)!r}"
        )
    extra = set(batch) - required - {EDGE_FEATURES, ACTION_MASK}
    if extra:
        raise BatchCollationError(
            f"packed graph batch has unsupported keys {sorted(extra)!r}"
        )

    nodes = _batch_array(batch[NODE_FEATURES], name=NODE_FEATURES, ndim=2)
    if nodes.shape[0] < 1 or nodes.shape[1] < 1:
        raise BatchCollationError("packed node_features must be non-empty")
    edges = np.asarray(batch[EDGE_INDEX])
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise BatchCollationError("packed edge_index must have integer shape (2, E)")
    graph_ptr = np.asarray(batch["graph_ptr"])
    if (
        graph_ptr.ndim != 1
        or len(graph_ptr) < 2
        or graph_ptr.dtype.kind not in "iu"
        or graph_ptr[0] != 0
        or graph_ptr[-1] != len(nodes)
        or np.any(np.diff(graph_ptr) < 1)
    ):
        raise BatchCollationError(
            "graph_ptr must delimit non-empty graphs and all packed nodes"
        )
    controlled = np.asarray(batch[CONTROLLED_NODE])
    graph_count = len(graph_ptr) - 1
    if (
        controlled.shape != (graph_count,)
        or controlled.dtype.kind not in "iu"
        or np.any(controlled < graph_ptr[:-1])
        or np.any(controlled >= graph_ptr[1:])
    ):
        raise BatchCollationError(
            "controlled_node must reference one node inside each graph"
        )
    if edges.shape[1] and (np.any(edges < 0) or np.any(edges >= len(nodes))):
        raise BatchCollationError("packed edge_index references an unknown node")
    if edges.shape[1]:
        edge_graph_ids = np.searchsorted(
            graph_ptr[1:],
            edges,
            side="right",
        )
        if np.any(edge_graph_ids[0] != edge_graph_ids[1]):
            raise BatchCollationError(
                "packed edge_index cannot connect different graphs"
            )

    normalized: GraphBatch = {
        NODE_FEATURES: np.ascontiguousarray(nodes, dtype=np.float32),
        EDGE_INDEX: np.ascontiguousarray(edges, dtype=np.int64),
        "graph_ptr": np.ascontiguousarray(graph_ptr, dtype=np.int64),
        CONTROLLED_NODE: np.ascontiguousarray(controlled, dtype=np.int64),
    }
    if EDGE_FEATURES in batch:
        edge_features = _batch_array(
            batch[EDGE_FEATURES],
            name=EDGE_FEATURES,
            ndim=2,
        )
        if edge_features.shape[0] != edges.shape[1] or edge_features.shape[1] < 1:
            raise BatchCollationError(
                "packed edge_features must align with packed edges"
            )
        normalized[EDGE_FEATURES] = np.ascontiguousarray(
            edge_features,
            dtype=np.float32,
        )
    if ACTION_MASK in batch:
        action_mask = _batch_array(
            batch[ACTION_MASK],
            name=ACTION_MASK,
            ndim=2,
        )
        if (
            action_mask.shape[0] != graph_count
            or action_mask.shape[1] < 1
            or not np.all((action_mask == 0) | (action_mask == 1))
            or np.any(np.sum(action_mask > 0, axis=1) < 1)
        ):
            raise BatchCollationError(
                "packed action_mask must be binary and enable an action per graph"
            )
        normalized[ACTION_MASK] = np.ascontiguousarray(
            action_mask,
            dtype=np.float32,
        )
    return normalized


class GraphBatchCollator:
    """Build one packed graph SAC batch for a configured shared module."""

    def __init__(
        self,
        *,
        module_id: str,
    ) -> None:
        if not isinstance(module_id, str) or not module_id:
            raise ValueError("module_id must be a non-empty string")
        self._module_id = module_id

    def collate(self, transitions: Sequence[object]) -> MultiModuleGraphBatch:
        if not transitions:
            raise BatchCollationError("collation requires at least one transition")
        validated: list[MultiModuleTransition] = []
        for transition in transitions:
            if not isinstance(transition, MultiModuleTransition):
                raise BatchCollationError(
                    "graph batches require MultiModuleTransition values"
                )
            if transition.module_id != self._module_id:
                raise BatchCollationError(
                    "sampled module ID does not match the shared graph module"
                )
            validated.append(transition)
        return {self._module_id: self._collate_module(validated)}

    @staticmethod
    def _collate_module(
        transitions: Sequence[MultiModuleTransition],
    ) -> GraphModuleBatch:
        observations = [item.data[Columns.OBS] for item in transitions]
        next_observations = [item.data[Columns.NEXT_OBS] for item in transitions]
        scalar_rows = [
            {
                key: value
                for key, value in item.data.items()
                if key not in {Columns.OBS, Columns.NEXT_OBS}
            }
            for item in transitions
        ]
        batch: GraphModuleBatch = FlatBatchCollator().collate(scalar_rows)
        batch[Columns.OBS] = pack_graph_observations(observations)
        batch[Columns.NEXT_OBS] = pack_graph_observations(next_observations)
        return batch


def _batch_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise BatchCollationError(f"{name} must be a numeric array") from error
    if array.ndim != ndim or array.dtype.kind not in "biuf":
        raise BatchCollationError(f"{name} must be a {ndim}-dimensional real array")
    if not np.isfinite(array).all():
        raise BatchCollationError(f"{name} must contain finite values")
    return array
