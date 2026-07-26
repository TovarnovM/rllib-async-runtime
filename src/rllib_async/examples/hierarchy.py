"""A finite hierarchical control example with sparse multi-agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

MANAGER_MODULE_ID = "manager"
WORKER_0_MODULE_ID = "worker_0"
WORKER_1_MODULE_ID = "worker_1"
HIERARCHY_MODULE_IDS = (
    MANAGER_MODULE_ID,
    WORKER_0_MODULE_ID,
    WORKER_1_MODULE_ID,
)


def hierarchy_module_spaces() -> dict[str, tuple[gym.Space, gym.Space]]:
    """Return fresh heterogeneous observation/action spaces for all modules."""

    return {
        MANAGER_MODULE_ID: (
            gym.spaces.Box(-1.0, 1.0, (5,), np.float32),
            gym.spaces.Discrete(2),
        ),
        WORKER_0_MODULE_ID: (
            gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
        ),
        WORKER_1_MODULE_ID: (
            gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
        ),
    }


def hierarchy_policy_mapping_fn(
    agent_id: str,
    *args: object,
    **kwargs: object,
) -> str:
    """Map the three stable agent IDs one-to-one onto their RLModules."""

    del args, kwargs
    if agent_id not in HIERARCHY_MODULE_IDS:
        raise KeyError(f"unknown hierarchy agent_id {agent_id!r}")
    return agent_id


class HierarchySwitchEnv(MultiAgentEnv):
    """Manager-gated continuous control with one active worker per env step.

    The manager has a genuine ``Discrete(2)`` action and appears once every
    ``manager_period`` physical environment steps. Its choice takes effect on
    the following step. Exactly one continuous worker appears on every step;
    the inactive worker is absent from both the observation and action turn.
    """

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        env_config = {} if env_config is None else env_config
        self._episode_length = self._positive_int(
            env_config.get("episode_length", 12),
            name="episode_length",
        )
        self._manager_period = self._positive_int(
            env_config.get("manager_period", 3),
            name="manager_period",
        )
        if self._manager_period > self._episode_length:
            raise ValueError("manager_period cannot exceed episode_length")

        spaces = hierarchy_module_spaces()
        self.possible_agents = list(HIERARCHY_MODULE_IDS)
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {
            module_id: observation_space
            for module_id, (observation_space, _) in spaces.items()
        }
        self.action_spaces = {
            module_id: action_space for module_id, (_, action_space) in spaces.items()
        }
        self._env_t = 0
        self._position = 0.0
        self._velocity = 0.0
        self._active_worker = WORKER_0_MODULE_ID
        self._done = False

    @property
    def manager_period(self) -> int:
        return self._manager_period

    @property
    def episode_length(self) -> int:
        return self._episode_length

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        super().reset(seed=seed)
        del options
        self.agents = list(self.possible_agents)
        self._env_t = 0
        self._position = 0.0
        self._velocity = 0.0
        self._active_worker = WORKER_0_MODULE_ID
        self._done = False
        observations = {
            MANAGER_MODULE_ID: self._manager_observation(),
            self._active_worker: self._worker_observation(),
        }
        return observations, {}

    def step(
        self,
        action_dict: Mapping[str, object],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self._done:
            raise RuntimeError("reset the hierarchy environment after termination")
        expected_agents = {self._active_worker}
        manager_turn = self._env_t % self._manager_period == 0
        if manager_turn:
            expected_agents.add(MANAGER_MODULE_ID)
        if set(action_dict) != expected_agents:
            raise ValueError(
                "actions must match the active action turn exactly: "
                f"expected {sorted(expected_agents)!r}, "
                f"got {sorted(map(str, action_dict))!r}"
            )

        worker_action = np.asarray(
            action_dict[self._active_worker],
            dtype=np.float32,
        )
        if worker_action.shape != (1,) or not np.isfinite(worker_action).all():
            raise ValueError("worker action must be one finite scalar")
        clipped_action = float(np.clip(worker_action[0], -1.0, 1.0))
        next_active_worker = self._active_worker
        if manager_turn:
            manager_action = action_dict[MANAGER_MODULE_ID]
            if not self.action_spaces[MANAGER_MODULE_ID].contains(manager_action):
                raise ValueError("manager action must be 0 or 1")
            next_active_worker = (
                WORKER_0_MODULE_ID if int(manager_action) == 0 else WORKER_1_MODULE_ID
            )

        previous_worker = self._active_worker
        acceleration = (
            abs(clipped_action)
            if previous_worker == WORKER_0_MODULE_ID
            else -abs(clipped_action)
        )
        self._velocity = float(
            np.clip(0.8 * self._velocity + 0.2 * acceleration, -1.0, 1.0)
        )
        self._position = float(
            np.clip(self._position + 0.2 * self._velocity, -1.0, 1.0)
        )
        reward = float(
            1.0 - abs(self._target() - self._position) - 0.05 * clipped_action**2
        )

        self._active_worker = next_active_worker
        self._env_t += 1
        self._done = self._env_t >= self._episode_length
        rewards = {
            MANAGER_MODULE_ID: reward,
            previous_worker: reward,
        }
        if self._done:
            observations = {
                module_id: (
                    self._manager_observation()
                    if module_id == MANAGER_MODULE_ID
                    else self._worker_observation()
                )
                for module_id in HIERARCHY_MODULE_IDS
            }
            terminateds = {"__all__": True}
        else:
            observations = {
                self._active_worker: self._worker_observation(),
            }
            if self._env_t % self._manager_period == 0:
                observations[MANAGER_MODULE_ID] = self._manager_observation()
            terminateds = {"__all__": False}
        return observations, rewards, terminateds, {"__all__": False}, {}

    def _manager_observation(self) -> np.ndarray:
        return np.asarray(
            [
                self._position,
                self._velocity,
                self._target(),
                (-1.0 if self._active_worker == WORKER_0_MODULE_ID else 1.0),
                self._progress(),
            ],
            dtype=np.float32,
        )

    def _worker_observation(self) -> np.ndarray:
        return np.asarray(
            [
                self._position,
                self._velocity,
                self._target(),
                self._progress(),
            ],
            dtype=np.float32,
        )

    def _target(self) -> float:
        return 1.0 if self._env_t < self._episode_length // 2 else -1.0

    def _progress(self) -> float:
        return 2.0 * self._env_t / self._episode_length - 1.0

    @staticmethod
    def _positive_int(value: object, *, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value


def build_hierarchy_sac_config(
    *,
    episode_length: int = 12,
    manager_period: int = 3,
    seed: int = 20260726,
) -> SACConfig:
    """Build the fixed heterogeneous SAC configuration used by Phase 9."""

    spaces = hierarchy_module_spaces()
    policies = {
        module_id: PolicySpec(
            observation_space=observation_space,
            action_space=action_space,
        )
        for module_id, (observation_space, action_space) in spaces.items()
    }
    return (
        SACConfig()
        .environment(
            HierarchySwitchEnv,
            env_config={
                "episode_length": episode_length,
                "manager_period": manager_period,
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
        .learners(num_learners=0, num_gpus_per_learner=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=hierarchy_policy_mapping_fn,
            count_steps_by="agent_steps",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    module_id: RLModuleSpec() for module_id in HIERARCHY_MODULE_IDS
                }
            )
        )
        .training(
            actor_lr=3e-4,
            alpha_lr=3e-4,
            critic_lr=3e-4,
            n_step=1,
            num_steps_sampled_before_learning_starts=0,
            policy_model_config={"fcnet_hiddens": [32, 32]},
            q_model_config={"fcnet_hiddens": [32, 32]},
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
