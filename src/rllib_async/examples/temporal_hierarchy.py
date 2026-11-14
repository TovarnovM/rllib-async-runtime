"""Causal, finite hierarchy and delayed-cue tasks for the shared replay runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

from rllib_async.examples.hierarchy import (
    HIERARCHY_MODULE_IDS,
    MANAGER_MODULE_ID,
    WORKER_0_MODULE_ID,
    WORKER_1_MODULE_ID,
)
from rllib_async.examples.shared_gnn import shared_gnn_observation_space
from rllib_async.gnn.episodes import (
    CONTROLLED_NODE,
    EDGE_COUNT,
    EDGE_FEATURES,
    EDGE_INDEX,
    NODE_COUNT,
    NODE_FEATURES,
)
from rllib_async.gnn.rl_modules import GraphDQNTorchRLModule, GraphSACTorchRLModule


def temporal_policy_mapping_fn(agent_id: str, *args: Any, **kwargs: Any) -> str:
    del args, kwargs
    module_id = agent_id.split(":", 1)[0]
    if module_id not in HIERARCHY_MODULE_IDS and module_id != "memory":
        raise KeyError(f"unknown temporal agent {agent_id!r}")
    return module_id


def temporal_hierarchy_spaces(
    *, vector_worker: bool = False
) -> dict[str, tuple[gym.Space, gym.Space]]:
    return {
        MANAGER_MODULE_ID: (shared_gnn_observation_space(3), gym.spaces.Discrete(2)),
        WORKER_0_MODULE_ID: (
            gym.spaces.Box(-2.0, 2.0, (4,), np.float32)
            if vector_worker
            else shared_gnn_observation_space(3),
            gym.spaces.Discrete(2),
        ),
        WORKER_1_MODULE_ID: (
            shared_gnn_observation_space(3),
            gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
        ),
    }


def _graph(features: np.ndarray, count: int) -> dict[str, Any]:
    """One small padded transport graph; the codec removes the padding."""

    nodes = np.zeros((3, 4), dtype=np.float32)
    nodes[:count] = features
    # Neighbour positions differ, while the goal is available to every node.
    nodes[1:count, 0] += np.arange(1, count, dtype=np.float32) * 0.05
    edges = np.zeros((2, 4), dtype=np.int64)
    for index in range(1, count):
        edges[:, 2 * (index - 1)] = (0, index)
        edges[:, 2 * (index - 1) + 1] = (index, 0)
    return {
        NODE_FEATURES: nodes,
        EDGE_INDEX: edges,
        EDGE_FEATURES: np.zeros((4, 1), dtype=np.float32),
        CONTROLLED_NODE: np.int64(0),
        NODE_COUNT: np.int64(count),
        EDGE_COUNT: np.int64(2 * (count - 1)),
    }


class CausalHierarchyEnv(MultiAgentEnv):
    """Choose a goal, then act on it; manager routing does not advance the world.

    Worker 0 controls a discrete actuator and worker 1 a continuous actuator.
    Each invocation is a finite local goal-reaching task, with its own agent
    identity and memory. Finishing that task does not terminate the manager's
    task. The manager is rewarded for tracking the external target; workers
    are rewarded for tracking the goal they were actually given.
    """

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        config = {} if env_config is None else env_config
        self.episode_length = _positive_int(config.get("episode_length", 8))
        self.manager_period = _positive_int(config.get("manager_period", 3))
        self.variable_duration = bool(config.get("variable_duration", True))
        self.vector_worker = bool(config.get("vector_worker", False))
        self.time_limit = bool(config.get("time_limit", False))
        self._spaces = temporal_hierarchy_spaces(vector_worker=self.vector_worker)
        self.possible_agents = [MANAGER_MODULE_ID] + [
            f"{module_id}:{invocation}"
            for invocation in range(self.episode_length)
            for module_id in (WORKER_0_MODULE_ID, WORKER_1_MODULE_ID)
        ]
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {
            agent: self.get_observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.get_action_space(agent) for agent in self.possible_agents
        }
        self.world_t = 0
        self._position = 0.0
        self._velocity = 0.0
        self._goal = 0.0
        self._invocation = 0
        self._worker: str | None = None
        self._strategy_steps = 0
        self._done = False
        self.agents = [MANAGER_MODULE_ID]

    def get_observation_space(self, agent_id: str) -> gym.Space:
        return self._spaces[temporal_policy_mapping_fn(agent_id)][0]

    def get_action_space(self, agent_id: str) -> gym.Space:
        return self._spaces[temporal_policy_mapping_fn(agent_id)][1]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.world_t = 0
        self._position = self._velocity = self._goal = 0.0
        self._invocation = 0
        self._worker = None
        self._strategy_steps = 0
        self._done = False
        self.agents = [MANAGER_MODULE_ID]
        return {MANAGER_MODULE_ID: self._observation(MANAGER_MODULE_ID)}, {}

    def step(self, action_dict: Mapping[str, Any]) -> tuple[Any, ...]:
        if self._done:
            raise RuntimeError("reset the temporal hierarchy after its episode ends")
        active = MANAGER_MODULE_ID if self._worker is None else self._worker
        if set(action_dict) != {active}:
            raise ValueError("actions must contain exactly the currently active agent")
        action = action_dict[active]
        if not self.get_action_space(active).contains(action):
            raise ValueError("action is outside the active agent's action space")
        if active == MANAGER_MODULE_ID:
            module_id = WORKER_0_MODULE_ID if int(action) == 0 else WORKER_1_MODULE_ID
            self._worker = f"{module_id}:{self._invocation}"
            self._goal = -0.6 if int(action) == 0 else 0.6
            self._strategy_steps = 0
            self.agents = [MANAGER_MODULE_ID, self._worker]
            return (
                {self._worker: self._observation(self._worker)},
                {},
                {"__all__": False},
                {"__all__": False},
                {},
            )

        module_id = temporal_policy_mapping_fn(active)
        force = (
            2.0 * int(action) - 1.0
            if module_id == WORKER_0_MODULE_ID
            else float(np.asarray(action)[0])
        )
        self._velocity = float(np.clip(0.4 * self._velocity + 0.6 * force, -1, 1))
        self._position = float(np.clip(self._position + 0.4 * self._velocity, -1, 1))
        manager_reward = 1.0 - abs(self._target() - self._position) - 0.05 * force**2
        worker_reward = 1.0 - abs(self._goal - self._position) - 0.05 * force**2
        self.world_t += 1
        self._strategy_steps += 1
        self._done = self.world_t >= self.episode_length
        goal_finished = self._strategy_steps >= self.manager_period or (
            self.variable_duration and abs(self._goal - self._position) <= 0.12
        )
        observations = {active: self._observation(active)}
        terminateds = {"__all__": self._done and not self.time_limit}
        truncateds = {"__all__": self._done and self.time_limit}
        infos: dict[str, Any] = {}
        if goal_finished or self._done:
            observations[MANAGER_MODULE_ID] = self._observation(MANAGER_MODULE_ID)
            # A time-limit interruption before goal completion is not a local
            # terminal either: bootstrap from the actual last observation.
            local_terminal = goal_finished
            if local_terminal:
                terminateds[active] = True
                infos[active] = {"subtask_terminated": True}
            self._worker = None
            self._invocation += 1
            self.agents = [] if self._done else [MANAGER_MODULE_ID]
        return (
            observations,
            {MANAGER_MODULE_ID: float(manager_reward), active: float(worker_reward)},
            terminateds,
            truncateds,
            infos,
        )

    def _target(self) -> float:
        return -0.6 if self.world_t < self.episode_length // 2 else 0.6

    def _observation(self, agent_id: str) -> Any:
        module_id = temporal_policy_mapping_fn(agent_id)
        features = np.asarray(
            [
                self._position,
                self._velocity,
                self._target() if module_id == MANAGER_MODULE_ID else self._goal,
                2.0 * self.world_t / self.episode_length - 1.0,
            ],
            dtype=np.float32,
        )
        if module_id == WORKER_0_MODULE_ID and self.vector_worker:
            return features
        return _graph(features, 1 + self.world_t % 3)


class CueMemoryEnv(MultiAgentEnv):
    """A balanced cue is visible only at reset and queried at the final action.

    Intermediate actions do not affect later observations. At the query, a
    memoryless policy therefore has expected reward zero; retaining the cue
    permits reward one. Both graph and vector variants expose identical facts.
    """

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        config = {} if env_config is None else env_config
        self.episode_length = _positive_int(config.get("episode_length", 5))
        if self.episode_length < 3:
            raise ValueError("the memory task needs at least three decisions")
        self.graph = bool(config.get("graph", False))
        self.continuous = bool(config.get("continuous", False))
        self.possible_agents = self.agents = ["memory"]
        self._agent_ids = {"memory"}
        self.observation_spaces = {
            "memory": shared_gnn_observation_space(3)
            if self.graph
            else gym.spaces.Box(-2.0, 2.0, (4,), np.float32)
        }
        self.action_spaces = {
            "memory": gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
            if self.continuous
            else gym.spaces.Discrete(2)
        }
        self.world_t = 0
        self._cue = 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.agents = ["memory"]
        self.world_t = 0
        self._cue = float(self.np_random.choice((-1, 1)))
        return {"memory": self._observation()}, {}

    def step(self, action_dict):
        if self.world_t >= self.episode_length:
            raise RuntimeError("reset the memory task after termination")
        if set(action_dict) != {"memory"} or not self.action_spaces["memory"].contains(
            action_dict["memory"]
        ):
            raise ValueError("memory task requires one legal action")
        value = (
            float(np.asarray(action_dict["memory"])[0])
            if self.continuous
            else 2.0 * int(action_dict["memory"]) - 1.0
        )
        self.world_t += 1
        done = self.world_t == self.episode_length
        reward = (1.0 if value * self._cue > 0 else -1.0) if done else 0.0
        return (
            {"memory": self._observation()},
            {"memory": reward},
            {"__all__": done},
            {"__all__": False},
            {},
        )

    def _observation(self):
        features = np.asarray(
            [
                self._cue if self.world_t == 0 else 0.0,
                float(self.world_t == self.episode_length - 1),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return _graph(features, 2) if self.graph else features


def _positive_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("episode_length and manager_period must be positive integers")
    return value


def _module_config(
    algorithm: str, *, graph: bool, recurrent: bool, seed: int
) -> AlgorithmConfig:
    config = DQNConfig() if algorithm == "dqn" else SACConfig()
    config = (
        config.framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(
            n_step=1,
            num_steps_sampled_before_learning_starts=0,
            target_network_update_freq=2,
            train_batch_size_per_learner=4,
        )
        .debugging(seed=seed)
    )
    model_config: dict[str, Any] = {"fcnet_hiddens": [16], "head_fcnet_hiddens": [16]}
    module_class = None
    if graph:
        module_class = (
            GraphDQNTorchRLModule if algorithm == "dqn" else GraphSACTorchRLModule
        )
        model_config["graph_encoder"] = {"hidden_dim": 16, "message_layers": 1}
        if algorithm == "sac":
            model_config["graph_encoders"] = {
                "actor": {"hidden_dim": 16, "message_layers": 1},
                "q1": {"hidden_dim": 20, "message_layers": 1},
                "q2": {"hidden_dim": 24, "message_layers": 1},
            }
    if recurrent:
        # Explicit modules are required; stock SAC's use_lstm flag does not
        # supply its missing sequence state/loss handling.
        from rllib_async.learner.recurrent import (
            RecurrentDQNTorchRLModule,
            RecurrentSACTorchRLModule,
        )

        module_class = (
            RecurrentDQNTorchRLModule
            if algorithm == "dqn"
            else RecurrentSACTorchRLModule
        )
        model_config["recurrent_hidden_dim"] = 16
    config.rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=module_class, model_config=model_config
        )
    )
    if algorithm == "dqn":
        config.training(
            dueling=True, double_q=True, num_atoms=1, epsilon=[(0, 1.0), (100, 0.1)]
        )
    else:
        config.training(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, twin_q=True)
    return config


def _rollout_config(
    env: type[MultiAgentEnv],
    env_config: dict[str, Any],
    spaces: dict[str, tuple[gym.Space, gym.Space]],
    module_configs: dict[str, AlgorithmConfig],
    seed: int,
) -> AlgorithmConfig:
    return (
        SACConfig()
        .environment(env, env_config=env_config)
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
        .learners(num_learners=0, num_gpus_per_learner=0)
        .multi_agent(
            policies={
                module_id: PolicySpec(observation_space=obs, action_space=act)
                for module_id, (obs, act) in spaces.items()
            },
            policy_mapping_fn=temporal_policy_mapping_fn,
            count_steps_by="agent_steps",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    module_id: RLModuleSpec(
                        module_class=module_configs[
                            module_id
                        ].rl_module_spec.module_class,
                        observation_space=obs,
                        action_space=act,
                        model_config=module_configs[
                            module_id
                        ].rl_module_spec.model_config,
                    )
                    for module_id, (obs, act) in spaces.items()
                }
            )
        )
        .debugging(seed=seed)
    )


def build_temporal_hierarchy_configs(
    *,
    episode_length: int = 8,
    manager_period: int = 3,
    variable_duration: bool = True,
    vector_worker: bool = False,
    recurrent: bool = False,
    time_limit: bool = False,
    seed: int = 20261114,
) -> tuple[
    AlgorithmConfig, dict[str, AlgorithmConfig], dict[str, tuple[gym.Space, gym.Space]]
]:
    """Build DEMO-02; pass trainable_modules=('manager',) to freeze workers."""

    spaces = temporal_hierarchy_spaces(vector_worker=vector_worker)
    module_configs = {
        module_id: _module_config(
            "sac" if module_id == WORKER_1_MODULE_ID else "dqn",
            graph=not (module_id == WORKER_0_MODULE_ID and vector_worker),
            recurrent=recurrent,
            seed=seed + index,
        )
        for index, module_id in enumerate(HIERARCHY_MODULE_IDS)
    }
    rollout = _rollout_config(
        CausalHierarchyEnv,
        {
            "episode_length": episode_length,
            "manager_period": manager_period,
            "variable_duration": variable_duration,
            "vector_worker": vector_worker,
            "time_limit": time_limit,
        },
        spaces,
        module_configs,
        seed,
    )
    return rollout, module_configs, spaces


def build_memory_configs(
    *,
    graph: bool = False,
    algorithm: str = "dqn",
    episode_length: int = 5,
    recurrent: bool = True,
    seed: int = 20261114,
) -> tuple[
    AlgorithmConfig, dict[str, AlgorithmConfig], dict[str, tuple[gym.Space, gym.Space]]
]:
    """Build DEMO-03 or its memoryless baseline with the same observations."""

    if algorithm not in ("dqn", "sac"):
        raise ValueError("memory algorithm must be dqn or sac")
    env_config = {
        "graph": graph,
        "continuous": algorithm == "sac",
        "episode_length": episode_length,
    }
    env = CueMemoryEnv(env_config)
    spaces = {"memory": (env.observation_spaces["memory"], env.action_spaces["memory"])}
    configs = {
        "memory": _module_config(algorithm, graph=graph, recurrent=recurrent, seed=seed)
    }
    return (
        _rollout_config(CueMemoryEnv, env_config, spaces, configs, seed),
        configs,
        spaces,
    )
