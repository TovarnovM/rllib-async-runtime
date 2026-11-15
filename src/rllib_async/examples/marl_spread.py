"""MPE2 Simple Spread with ego graphs and optional discrete DQN teammates.

Install the ``marl`` extra. Physics and rewards belong to MPE2; this adapter
only changes observations and maps discrete commands to native actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

from rllib_async.gnn.rl_modules import GraphDQNTorchRLModule, GraphSACTorchRLModule


def spread_policy_mapping(
    agent_id: str, *args: Any, discrete_agents: int = 0, **kwargs: Any
) -> str:
    del args, kwargs
    return "dqn" if int(agent_id.removeprefix("agent_")) < discrete_agents else "sac"


class GraphSpreadParallelEnv(BaseParallelWrapper):
    """A fixed-size graph built exclusively from each agent's native observation.

    Nodes: ego, other agents (native order), landmarks. Node features are
    relative xy, observed velocity xy (ego only), agent/landmark/ego flags,
    and the publicly known discrete-role flag. Bidirectional ego edges carry
    relative displacement. No velocities or hidden world state are fetched.
    """

    def __init__(
        self,
        *,
        agents: int = 8,
        discrete_agents: int = 0,
        max_cycles: int = 100,
        render_mode: str | None = None,
    ) -> None:
        if type(agents) is not int or agents < 2:
            raise ValueError("agents must be an integer of at least two")
        if type(discrete_agents) is not int or not 0 <= discrete_agents < agents:
            raise ValueError("discrete_agents must be between zero and agents - 1")
        if type(max_cycles) is not int or max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")
        super().__init__(
            simple_spread_v3.parallel_env(
                N=agents,
                continuous_actions=True,
                max_cycles=max_cycles,
                local_ratio=0.5,
                render_mode=render_mode,
            )
        )
        self._agent_count = agents
        self._discrete_agents = discrete_agents
        nodes = 2 * agents
        edges = 2 * (nodes - 1)
        self._graph_space = gym.spaces.Dict(
            {
                "node_features": gym.spaces.Box(
                    -np.inf, np.inf, (nodes, 8), np.float32
                ),
                "edge_index": gym.spaces.Box(0, nodes - 1, (2, edges), np.int64),
                "edge_features": gym.spaces.Box(
                    -np.inf, np.inf, (edges, 2), np.float32
                ),
                "controlled_node": gym.spaces.Discrete(nodes),
                "node_count": gym.spaces.Discrete(nodes, start=1),
                "edge_count": gym.spaces.Discrete(edges + 1),
            }
        )
        self._edges = np.empty((2, edges), dtype=np.int64)
        self._edges[:, 0::2] = (np.zeros(nodes - 1), np.arange(1, nodes))
        self._edges[:, 1::2] = self._edges[::-1, 0::2]
        self._codebook = np.eye(5, dtype=np.float32)
        self.observation_spaces = dict.fromkeys(self.possible_agents, self._graph_space)
        self.action_spaces = {
            agent: (
                gym.spaces.Discrete(5)
                if spread_policy_mapping(agent, discrete_agents=discrete_agents)
                == "dqn"
                else self.env.action_space(agent)
            )
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._graphs(observations), infos

    def step(self, actions: Mapping[str, Any]):
        native_actions = {}
        for agent, action in actions.items():
            if isinstance(self.action_spaces[agent], gym.spaces.Discrete):
                if not self.action_spaces[agent].contains(action):
                    raise ValueError("discrete action must be in [0, 4]")
                native_actions[agent] = self._codebook[int(action)]
            else:
                native_actions[agent] = action
        observations, rewards, terminated, truncated, infos = self.env.step(
            native_actions
        )
        return self._graphs(observations), rewards, terminated, truncated, infos

    def _graphs(self, observations: Mapping[str, np.ndarray]) -> dict[str, Any]:
        count = self._agent_count
        graphs = {}
        for agent, observation in observations.items():
            ego = int(agent.removeprefix("agent_"))
            features = np.zeros((2 * count, 8), dtype=np.float32)
            features[0, 2:4] = observation[:2]
            features[:count, 4] = 1.0
            features[count:, 5] = 1.0
            features[0, 6] = 1.0
            order = [ego, *(i for i in range(count) if i != ego)]
            features[:count, 7] = np.asarray(order) < self._discrete_agents
            features[count:, :2] = observation[4 : 4 + 2 * count].reshape(-1, 2)
            features[1:count, :2] = observation[
                4 + 2 * count : 4 + 2 * count + 2 * (count - 1)
            ].reshape(-1, 2)
            graphs[agent] = {
                "node_features": features,
                "edge_index": self._edges.copy(),
                "edge_features": (
                    features[self._edges[1], :2] - features[self._edges[0], :2]
                ),
                "controlled_node": np.int64(0),
                "node_count": np.int64(2 * count),
                "edge_count": np.int64(self._edges.shape[1]),
            }
        return graphs


class GraphSpreadEnv(ParallelPettingZooEnv):
    """RLlib's supported Parallel API adapter with a physical-time clock."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.world_t = 0
        super().__init__(GraphSpreadParallelEnv(**dict(config or {})))

    def reset(self, *, seed=None, options=None):
        self.world_t = 0
        result = super().reset(seed=seed, options=options)
        self.agents = self.par_env.agents.copy()
        return result

    def step(self, action_dict):
        result = super().step(action_dict)
        self.world_t += 1
        self.agents = self.par_env.agents.copy()
        return result


def build_spread_configs(
    *,
    agents: int = 8,
    discrete_agents: int = 0,
    max_cycles: int = 100,
    hidden_dim: int = 64,
    batch_size: int = 256,
    learning_starts: int = 2048,
    training_intensity: float = 1.0,
    num_gpus: float = 0.0,
    seed: int = 20261115,
) -> tuple[AlgorithmConfig, dict[str, AlgorithmConfig]]:
    """SAC-only or DQN/SAC team; all representations are learned from raw replay."""

    env_config = {
        "agents": agents,
        "discrete_agents": discrete_agents,
        "max_cycles": max_cycles,
    }
    env = GraphSpreadParallelEnv(**env_config)
    try:
        spaces = {
            spread_policy_mapping(agent, discrete_agents=discrete_agents): (
                env.observation_space(agent),
                env.action_space(agent),
            )
            for agent in env.possible_agents
        }
    finally:
        env.close()
    modules = {}
    for index, mid in enumerate(sorted(spaces)):
        config = DQNConfig() if mid == "dqn" else SACConfig()
        config.framework("torch").api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        ).learners(num_learners=0, num_gpus_per_learner=num_gpus).training(
            n_step=1,
            train_batch_size_per_learner=batch_size,
            num_steps_sampled_before_learning_starts=learning_starts,
            training_intensity=training_intensity,
        ).debugging(seed=seed + index)
        if mid == "dqn":
            config.training(
                lr=3e-4,
                double_q=True,
                dueling=True,
                num_atoms=1,
                epsilon=[(0, 1.0), (100_000, 0.05)],
                target_network_update_freq=1000,
            )
        else:
            config.training(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-4, twin_q=True)
        config.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=(
                    GraphDQNTorchRLModule if mid == "dqn" else GraphSACTorchRLModule
                ),
                observation_space=spaces[mid][0],
                action_space=spaces[mid][1],
                model_config={
                    "fcnet_hiddens": [hidden_dim],
                    "head_fcnet_hiddens": [hidden_dim],
                    "graph_encoder": {"hidden_dim": hidden_dim, "message_layers": 2},
                },
            )
        )
        modules[mid] = config
    rollout = (
        SACConfig()
        .environment(GraphSpreadEnv, env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .env_runners(
            num_env_runners=0,
            create_local_env_runner=True,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            episodes_to_numpy=True,
        )
        .learners(num_learners=0, num_gpus_per_learner=num_gpus)
        .multi_agent(
            policies={
                mid: PolicySpec(observation_space=obs, action_space=action)
                for mid, (obs, action) in spaces.items()
            },
            policy_mapping_fn=partial(
                spread_policy_mapping, discrete_agents=discrete_agents
            ),
            count_steps_by="agent_steps",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={mid: config.rl_module_spec for mid, config in modules.items()}
            )
        )
        .debugging(seed=seed)
    )
    return rollout, modules
