"""Finite homogeneous multi-agent example with one shared ego-GNN policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

from rllib_async.gnn import SharedGraphSACCatalog
from rllib_async.gnn.episodes import (
    CONTROLLED_NODE,
    EDGE_COUNT,
    EDGE_FEATURES,
    EDGE_INDEX,
    NODE_COUNT,
    NODE_FEATURES,
)

SHARED_GNN_MODULE_ID = "shared_graph"
DEFAULT_GRAPH_AGENT_COUNT = 4
GRAPH_NODE_FEATURE_DIM = 4
GRAPH_EDGE_FEATURE_DIM = 1
GRAPH_ACTION_COUNT = 3


def graph_agent_ids(agent_count: int = DEFAULT_GRAPH_AGENT_COUNT) -> tuple[str, ...]:
    """Return stable logical agent IDs without creating policy IDs."""

    if (
        not isinstance(agent_count, int)
        or isinstance(agent_count, bool)
        or agent_count < 2
    ):
        raise ValueError("agent_count must be an integer of at least two")
    return tuple(f"agent_{index}" for index in range(agent_count))


def shared_gnn_observation_space(
    agent_count: int = DEFAULT_GRAPH_AGENT_COUNT,
) -> gym.spaces.Dict:
    """Return the fixed transport space for padded variable-size ego-graphs."""

    graph_agent_ids(agent_count)
    max_edges = 2 * (agent_count - 1)
    return gym.spaces.Dict(
        {
            NODE_FEATURES: gym.spaces.Box(
                -2.0,
                2.0,
                (agent_count, GRAPH_NODE_FEATURE_DIM),
                np.float32,
            ),
            EDGE_INDEX: gym.spaces.Box(
                0,
                agent_count - 1,
                (2, max_edges),
                np.int64,
            ),
            EDGE_FEATURES: gym.spaces.Box(
                -2.0,
                2.0,
                (max_edges, GRAPH_EDGE_FEATURE_DIM),
                np.float32,
            ),
            CONTROLLED_NODE: gym.spaces.Discrete(agent_count),
            NODE_COUNT: gym.spaces.Discrete(agent_count, start=1),
            EDGE_COUNT: gym.spaces.Discrete(max_edges + 1),
        }
    )


def shared_gnn_module_spaces(
    agent_count: int = DEFAULT_GRAPH_AGENT_COUNT,
) -> dict[str, tuple[gym.Space, gym.Space]]:
    """Return the single shared module's observation and action spaces."""

    return {
        SHARED_GNN_MODULE_ID: (
            shared_gnn_observation_space(agent_count),
            gym.spaces.Discrete(GRAPH_ACTION_COUNT),
        )
    }


def shared_gnn_policy_mapping_fn(
    agent_id: str,
    *args: object,
    **kwargs: object,
) -> str:
    """Map every logical graph agent to one shared RLModule."""

    del args, kwargs
    if not isinstance(agent_id, str) or not agent_id.startswith("agent_"):
        raise KeyError(f"unknown graph agent_id {agent_id!r}")
    return SHARED_GNN_MODULE_ID


class EgoGraphCoordinationEnv(MultiAgentEnv):
    """Small finite coordination task producing different local graph sizes."""

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        env_config = {} if env_config is None else env_config
        self._agent_count = self._positive_int(
            env_config.get("agent_count", DEFAULT_GRAPH_AGENT_COUNT),
            name="agent_count",
            minimum=2,
        )
        self._episode_length = self._positive_int(
            env_config.get("episode_length", 8),
            name="episode_length",
        )
        self.possible_agents = list(graph_agent_ids(self._agent_count))
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        observation_space = shared_gnn_observation_space(self._agent_count)
        action_space = gym.spaces.Discrete(GRAPH_ACTION_COUNT)
        self.observation_spaces = {
            agent_id: observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: action_space for agent_id in self.possible_agents
        }
        self._env_t = 0
        self._positions = np.zeros(self._agent_count, dtype=np.float32)
        self._velocities = np.zeros(self._agent_count, dtype=np.float32)
        self._done = False

    @property
    def agent_count(self) -> int:
        return self._agent_count

    @property
    def episode_length(self) -> int:
        return self._episode_length

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        super().reset(seed=seed)
        del options
        self.agents = list(self.possible_agents)
        self._env_t = 0
        self._positions = np.linspace(
            -0.75,
            0.75,
            self._agent_count,
            dtype=np.float32,
        )
        self._velocities = np.zeros(self._agent_count, dtype=np.float32)
        self._done = False
        return self._observations(), {}

    def step(
        self,
        action_dict: Mapping[str, object],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self._done:
            raise RuntimeError("reset the graph environment after termination")
        if set(action_dict) != self._agent_ids:
            raise ValueError("every logical graph agent must act exactly once")
        actions = np.empty(self._agent_count, dtype=np.int64)
        for index, agent_id in enumerate(self.possible_agents):
            action = action_dict[agent_id]
            if not self.action_spaces[agent_id].contains(action):
                raise ValueError("graph agent actions must be 0, 1, or 2")
            actions[index] = int(action)

        accelerations = actions.astype(np.float32) - 1.0
        next_velocities = np.clip(
            0.65 * self._velocities + 0.35 * accelerations,
            -1.0,
            1.0,
        ).astype(np.float32)
        next_positions = np.clip(
            self._positions + 0.15 * next_velocities,
            -1.0,
            1.0,
        ).astype(np.float32)
        target = 0.5 if self._env_t < self._episode_length // 2 else -0.5
        rewards = {
            agent_id: float(
                1.0
                - abs(float(next_positions[index]) - target)
                - 0.05 * abs(float(accelerations[index]))
            )
            for index, agent_id in enumerate(self.possible_agents)
        }

        self._velocities = next_velocities
        self._positions = next_positions
        self._env_t += 1
        self._done = self._env_t >= self._episode_length
        return (
            self._observations(),
            rewards,
            {"__all__": self._done},
            {"__all__": False},
            {},
        )

    def _observations(self) -> dict[str, dict[str, Any]]:
        return {
            agent_id: self._ego_graph(index)
            for index, agent_id in enumerate(self.possible_agents)
        }

    def _ego_graph(self, controlled_index: int) -> dict[str, Any]:
        visible_count = 1 + ((controlled_index + self._env_t) % self._agent_count)
        visible = [controlled_index]
        visible.extend(
            (controlled_index + offset) % self._agent_count
            for offset in range(1, visible_count)
        )
        max_edges = 2 * (self._agent_count - 1)
        node_features = np.zeros(
            (self._agent_count, GRAPH_NODE_FEATURE_DIM),
            dtype=np.float32,
        )
        controlled_position = float(self._positions[controlled_index])
        progress = 2.0 * self._env_t / self._episode_length - 1.0
        for local_index, global_index in enumerate(visible):
            node_features[local_index] = (
                float(self._positions[global_index]) - controlled_position,
                float(self._velocities[global_index]),
                1.0 if local_index == 0 else 0.0,
                progress,
            )

        edge_index = np.zeros((2, max_edges), dtype=np.int64)
        edge_features = np.zeros(
            (max_edges, GRAPH_EDGE_FEATURE_DIM),
            dtype=np.float32,
        )
        edge_count = 0
        for local_index in range(1, visible_count):
            relative = node_features[local_index, 0]
            edge_index[:, edge_count] = (0, local_index)
            edge_features[edge_count, 0] = relative
            edge_count += 1
            edge_index[:, edge_count] = (local_index, 0)
            edge_features[edge_count, 0] = -relative
            edge_count += 1
        return {
            NODE_FEATURES: node_features,
            EDGE_INDEX: edge_index,
            EDGE_FEATURES: edge_features,
            CONTROLLED_NODE: np.int64(0),
            NODE_COUNT: np.int64(visible_count),
            EDGE_COUNT: np.int64(edge_count),
        }

    @staticmethod
    def _positive_int(
        value: object,
        *,
        name: str,
        minimum: int = 1,
    ) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
        return value


def build_shared_gnn_sac_config(
    *,
    agent_count: int = DEFAULT_GRAPH_AGENT_COUNT,
    episode_length: int = 8,
    hidden_dim: int = 32,
    message_layers: int = 2,
    num_gpus_per_learner: int = 0,
    seed: int = 20260726,
) -> SACConfig:
    """Build the fixed shared-policy discrete SAC configuration for Phase 10."""

    graph_agent_ids(agent_count)
    for name, value in (
        ("episode_length", episode_length),
        ("hidden_dim", hidden_dim),
        ("message_layers", message_layers),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(num_gpus_per_learner, int)
        or isinstance(num_gpus_per_learner, bool)
        or num_gpus_per_learner not in {0, 1}
    ):
        raise ValueError("num_gpus_per_learner must be 0 or 1")
    spaces = shared_gnn_module_spaces(agent_count)
    observation_space, action_space = spaces[SHARED_GNN_MODULE_ID]
    model_config = {
        "twin_q": True,
        "fcnet_hiddens": [hidden_dim] * message_layers,
        "fcnet_activation": "relu",
        "head_fcnet_hiddens": [hidden_dim],
        "head_fcnet_activation": "relu",
    }
    return (
        SACConfig()
        .environment(
            EgoGraphCoordinationEnv,
            env_config={
                "agent_count": agent_count,
                "episode_length": episode_length,
            },
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .env_runners(
            num_env_runners=0,
            create_local_env_runner=True,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            episodes_to_numpy=True,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        .multi_agent(
            policies={
                SHARED_GNN_MODULE_ID: PolicySpec(
                    observation_space=observation_space,
                    action_space=action_space,
                )
            },
            policy_mapping_fn=shared_gnn_policy_mapping_fn,
            count_steps_by="agent_steps",
        )
        .rl_module(
            model_config=model_config,
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    SHARED_GNN_MODULE_ID: RLModuleSpec(
                        module_class=DefaultSACTorchRLModule,
                        catalog_class=SharedGraphSACCatalog,
                    )
                }
            ),
        )
        .training(
            actor_lr=3e-4,
            alpha_lr=3e-4,
            critic_lr=3e-4,
            n_step=1,
            num_steps_sampled_before_learning_starts=0,
            policy_model_config={
                "fcnet_hiddens": [hidden_dim] * message_layers,
                "post_fcnet_hiddens": [hidden_dim],
            },
            q_model_config={
                "fcnet_hiddens": [hidden_dim] * message_layers,
                "post_fcnet_hiddens": [hidden_dim],
            },
            replay_buffer_config={
                "type": "MultiAgentEpisodeReplayBuffer",
                "capacity": 4_096,
            },
            target_network_update_freq=1,
            train_batch_size_per_learner=8,
            twin_q=True,
        )
        .debugging(seed=seed)
    )
