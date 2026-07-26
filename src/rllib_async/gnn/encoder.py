"""Pure-PyTorch ego-graph encoder exposed through RLlib's SAC catalog seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel

from rllib_async.gnn.episodes import (
    CONTROLLED_NODE,
    EDGE_COUNT,
    EDGE_FEATURES,
    EDGE_INDEX,
    NODE_COUNT,
    NODE_FEATURES,
)


@dataclass
class GraphEncoderConfig(ModelConfig):
    """Static dimensions for a packed or fixed-space ego-graph encoder."""

    node_feature_dim: int = 1
    edge_feature_dim: int = 0
    hidden_dim: int = 32
    message_layers: int = 2

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (self.hidden_dim,)

    def build(self, framework: str = "torch") -> Encoder:
        if framework != "torch":
            raise ValueError("GraphEncoderConfig supports PyTorch only")
        for name, value in (
            ("node_feature_dim", self.node_feature_dim),
            ("hidden_dim", self.hidden_dim),
            ("message_layers", self.message_layers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.edge_feature_dim, int)
            or isinstance(self.edge_feature_dim, bool)
            or self.edge_feature_dim < 0
        ):
            raise ValueError("edge_feature_dim must be a non-negative integer")
        return TorchGraphEncoder(self)


class TorchGraphEncoder(TorchModel, Encoder):
    """Mean-aggregation message passing over a batch of independent ego-graphs."""

    def __init__(self, config: GraphEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.node_projection = torch.nn.Linear(
            config.node_feature_dim,
            config.hidden_dim,
        )
        self.self_updates = torch.nn.ModuleList(
            torch.nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(config.message_layers)
        )
        self.neighbor_updates = torch.nn.ModuleList(
            torch.nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            for _ in range(config.message_layers)
        )
        self.edge_updates = (
            torch.nn.ModuleList(
                torch.nn.Linear(
                    config.edge_feature_dim,
                    config.hidden_dim,
                    bias=False,
                )
                for _ in range(config.message_layers)
            )
            if config.edge_feature_dim
            else None
        )
        self.readout = torch.nn.Linear(2 * config.hidden_dim, config.hidden_dim)

    def _forward(self, inputs: dict[str, Any], **kwargs: object) -> dict[str, Any]:
        del kwargs
        observations = inputs.get(Columns.OBS)
        if not isinstance(observations, Mapping):
            raise ValueError("graph encoder observations must be a mapping")
        packed = self._packed_tensors(observations)
        node_features = packed[NODE_FEATURES]
        edge_index = packed[EDGE_INDEX]
        graph_ptr = packed["graph_ptr"]
        controlled = packed[CONTROLLED_NODE]
        edge_features = packed.get(EDGE_FEATURES)

        hidden = torch.relu(self.node_projection(node_features))
        for index, (self_update, neighbor_update) in enumerate(
            zip(self.self_updates, self.neighbor_updates, strict=True)
        ):
            aggregated = torch.zeros_like(hidden)
            degree = torch.zeros(
                hidden.shape[0],
                dtype=hidden.dtype,
                device=hidden.device,
            )
            if edge_index.shape[1]:
                sources = edge_index[0]
                targets = edge_index[1]
                messages = hidden[sources]
                if edge_features is not None:
                    assert self.edge_updates is not None
                    messages = messages + self.edge_updates[index](edge_features)
                aggregated.index_add_(0, targets, messages)
                degree.index_add_(
                    0,
                    targets,
                    torch.ones(
                        len(targets),
                        dtype=hidden.dtype,
                        device=hidden.device,
                    ),
                )
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
            hidden = torch.relu(self_update(hidden) + neighbor_update(aggregated))

        graph_lengths = graph_ptr[1:] - graph_ptr[:-1]
        graph_ids = torch.repeat_interleave(
            torch.arange(
                len(graph_lengths),
                dtype=torch.long,
                device=hidden.device,
            ),
            graph_lengths,
        )
        pooled = torch.zeros(
            (len(graph_lengths), hidden.shape[1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        pooled.index_add_(0, graph_ids, hidden)
        pooled = pooled / graph_lengths.to(hidden.dtype).unsqueeze(-1)
        controlled_hidden = hidden[controlled]
        return {
            ENCODER_OUT: torch.relu(
                self.readout(torch.cat((controlled_hidden, pooled), dim=-1))
            )
        }

    def _packed_tensors(
        self,
        observations: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        if "graph_ptr" in observations:
            return self._validate_packed(observations)
        return self._pack_padded(observations)

    def _validate_packed(
        self,
        observations: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        device = self.node_projection.weight.device
        nodes = self._tensor(
            observations.get(NODE_FEATURES),
            dtype=torch.float32,
            device=device,
        )
        edges = self._tensor(
            observations.get(EDGE_INDEX),
            dtype=torch.long,
            device=device,
        )
        graph_ptr = self._tensor(
            observations.get("graph_ptr"),
            dtype=torch.long,
            device=device,
        )
        controlled = self._tensor(
            observations.get(CONTROLLED_NODE),
            dtype=torch.long,
            device=device,
        )
        if (
            nodes.ndim != 2
            or nodes.shape[1] != self.config.node_feature_dim
            or edges.ndim != 2
            or edges.shape[0] != 2
            or graph_ptr.ndim != 1
            or len(graph_ptr) < 2
            or controlled.shape != (len(graph_ptr) - 1,)
        ):
            raise ValueError("invalid packed graph tensor shapes")
        if (
            graph_ptr[0].item() != 0
            or graph_ptr[-1].item() != len(nodes)
            or torch.any(graph_ptr[1:] <= graph_ptr[:-1])
        ):
            raise ValueError("graph_ptr must delimit non-empty packed graphs")
        if edges.shape[1] and (torch.any(edges < 0) or torch.any(edges >= len(nodes))):
            raise ValueError("edge_index references an unknown packed node")
        if edges.shape[1]:
            edge_graph_ids = torch.bucketize(
                edges,
                graph_ptr[1:],
                right=True,
            )
            if torch.any(edge_graph_ids[0] != edge_graph_ids[1]):
                raise ValueError("edge_index cannot connect different packed graphs")
        if torch.any(controlled < graph_ptr[:-1]) or torch.any(
            controlled >= graph_ptr[1:]
        ):
            raise ValueError("controlled_node is outside its packed graph")
        if not torch.isfinite(nodes).all():
            raise ValueError("node_features must be finite")

        packed = {
            NODE_FEATURES: nodes,
            EDGE_INDEX: edges,
            "graph_ptr": graph_ptr,
            CONTROLLED_NODE: controlled,
        }
        edge_features_raw = observations.get(EDGE_FEATURES)
        if self.config.edge_feature_dim:
            edge_features = self._tensor(
                edge_features_raw,
                dtype=torch.float32,
                device=device,
            )
            if edge_features.shape != (
                edges.shape[1],
                self.config.edge_feature_dim,
            ):
                raise ValueError("edge_features do not align with packed edges")
            if not torch.isfinite(edge_features).all():
                raise ValueError("edge_features must be finite")
            packed[EDGE_FEATURES] = edge_features
        elif edge_features_raw is not None:
            raise ValueError("unexpected edge_features for this graph encoder")
        return packed

    def _pack_padded(
        self,
        observations: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        device = self.node_projection.weight.device
        nodes = self._tensor(
            observations.get(NODE_FEATURES),
            dtype=torch.float32,
            device=device,
        )
        edges = self._tensor(
            observations.get(EDGE_INDEX),
            dtype=torch.long,
            device=device,
        )
        node_counts = self._tensor(
            observations.get(NODE_COUNT),
            dtype=torch.long,
            device=device,
        )
        edge_counts = self._tensor(
            observations.get(EDGE_COUNT),
            dtype=torch.long,
            device=device,
        )
        controlled = self._tensor(
            observations.get(CONTROLLED_NODE),
            dtype=torch.long,
            device=device,
        )
        if nodes.ndim == 2:
            nodes = nodes.unsqueeze(0)
        if edges.ndim == 2:
            edges = edges.unsqueeze(0)
        node_counts = node_counts.reshape(-1)
        edge_counts = edge_counts.reshape(-1)
        controlled = controlled.reshape(-1)
        batch_size = nodes.shape[0]
        if (
            nodes.ndim != 3
            or nodes.shape[2] != self.config.node_feature_dim
            or edges.ndim != 3
            or edges.shape[:2] != (batch_size, 2)
            or node_counts.shape != (batch_size,)
            or edge_counts.shape != (batch_size,)
            or controlled.shape != (batch_size,)
        ):
            raise ValueError("invalid padded graph tensor shapes")
        if not torch.isfinite(nodes).all():
            raise ValueError("padded node_features must be finite")
        if (
            torch.any(node_counts < 1)
            or torch.any(node_counts > nodes.shape[1])
            or torch.any(edge_counts < 0)
            or torch.any(edge_counts > edges.shape[2])
            or torch.any(controlled < 0)
            or torch.any(controlled >= node_counts)
        ):
            raise ValueError("invalid padded graph counts or controlled nodes")

        edge_features = None
        if self.config.edge_feature_dim:
            edge_features = self._tensor(
                observations.get(EDGE_FEATURES),
                dtype=torch.float32,
                device=device,
            )
            if edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0)
            if edge_features.shape != (
                batch_size,
                edges.shape[2],
                self.config.edge_feature_dim,
            ):
                raise ValueError("invalid padded edge_features shape")
            if not torch.isfinite(edge_features).all():
                raise ValueError("padded edge_features must be finite")

        node_parts: list[torch.Tensor] = []
        edge_parts: list[torch.Tensor] = []
        edge_feature_parts: list[torch.Tensor] = []
        global_controlled: list[int] = []
        graph_ptr = [0]
        for batch_index, (node_count, edge_count) in enumerate(
            zip(node_counts.tolist(), edge_counts.tolist(), strict=True)
        ):
            node_part = nodes[batch_index, :node_count]
            edge_part = edges[batch_index, :, :edge_count]
            if edge_count and (
                torch.any(edge_part < 0) or torch.any(edge_part >= node_count)
            ):
                raise ValueError("padded edge_index references a padded node")
            offset = graph_ptr[-1]
            node_parts.append(node_part)
            edge_parts.append(edge_part + offset)
            global_controlled.append(offset + int(controlled[batch_index]))
            graph_ptr.append(offset + node_count)
            if edge_features is not None:
                edge_feature_parts.append(edge_features[batch_index, :edge_count])
        packed = {
            NODE_FEATURES: torch.cat(node_parts, dim=0),
            EDGE_INDEX: torch.cat(edge_parts, dim=1),
            "graph_ptr": torch.tensor(
                graph_ptr,
                dtype=torch.long,
                device=device,
            ),
            CONTROLLED_NODE: torch.tensor(
                global_controlled,
                dtype=torch.long,
                device=device,
            ),
        }
        if edge_features is not None:
            packed[EDGE_FEATURES] = torch.cat(edge_feature_parts, dim=0)
        return packed

    @staticmethod
    def _tensor(
        value: object,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError("graph observation is missing a required tensor")
        return torch.as_tensor(value, dtype=dtype, device=device)


class SharedGraphSACCatalog(SACCatalog):
    """SAC catalog that replaces stock MLP observation encoders with a GNN."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict[str, Any],
        view_requirements: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
            view_requirements=view_requirements,
        )
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "the Phase 10 shared-graph SAC example requires a Discrete action space"
            )

    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict[str, Any],
        action_space: gym.Space | None = None,
    ) -> ModelConfig:
        del action_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("shared graph observations require a Gymnasium Dict")
        spaces = observation_space.spaces
        required = {
            NODE_FEATURES,
            EDGE_INDEX,
            CONTROLLED_NODE,
            NODE_COUNT,
            EDGE_COUNT,
        }
        if not required.issubset(spaces):
            raise ValueError("graph observation space is missing transport fields")
        node_space = spaces[NODE_FEATURES]
        edge_space = spaces[EDGE_INDEX]
        if (
            not isinstance(node_space, gym.spaces.Box)
            or len(node_space.shape) != 2
            or not isinstance(edge_space, gym.spaces.Box)
            or len(edge_space.shape) != 2
            or edge_space.shape[0] != 2
        ):
            raise ValueError("invalid graph node_features or edge_index space")
        edge_feature_dim = 0
        if EDGE_FEATURES in spaces:
            edge_feature_space = spaces[EDGE_FEATURES]
            if (
                not isinstance(edge_feature_space, gym.spaces.Box)
                or len(edge_feature_space.shape) != 2
                or edge_feature_space.shape[0] != edge_space.shape[1]
            ):
                raise ValueError("invalid graph edge_features space")
            edge_feature_dim = edge_feature_space.shape[1]
        hidden_dims = model_config_dict["fcnet_hiddens"]
        if (
            not isinstance(hidden_dims, list | tuple)
            or not hidden_dims
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in hidden_dims
            )
        ):
            raise ValueError("fcnet_hiddens must configure graph message layers")
        return GraphEncoderConfig(
            input_dims=node_space.shape,
            node_feature_dim=node_space.shape[1],
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dims[-1],
            message_layers=len(hidden_dims),
        )
