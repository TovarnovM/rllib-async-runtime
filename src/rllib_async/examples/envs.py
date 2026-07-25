"""Small environments for runtime correctness and throughput checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np


class SyntheticThroughputEnv(gym.Env[np.ndarray, np.ndarray]):
    """Cheap continuous-control episodes for orchestration measurements."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        env_config = {} if env_config is None else env_config
        episode_length = env_config.get("episode_length", 32)
        if (
            not isinstance(episode_length, int)
            or isinstance(episode_length, bool)
            or episode_length < 1
        ):
            raise ValueError("episode_length must be a positive integer")
        self._episode_length = episode_length
        self._step = 0
        self._last_action = 0.0
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._step = 0
        self._last_action = 0.0
        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError("action has the wrong shape")
        self._last_action = float(np.clip(action_array[0], -1.0, 1.0))
        self._step += 1
        reward = 1.0 - self._last_action**2
        terminated = self._step >= self._episode_length
        return self._observation(), reward, terminated, False, {}

    def _observation(self) -> np.ndarray:
        progress = 2.0 * self._step / self._episode_length - 1.0
        return np.asarray(
            [progress, self._last_action, 1.0],
            dtype=np.float32,
        )
