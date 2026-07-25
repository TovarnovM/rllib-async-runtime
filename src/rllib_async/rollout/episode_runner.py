"""One-environment, whole-episode RLlib rollout runner."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO
from ray.tune.registry import ENV_CREATOR, _global_registry

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
)
from rllib_async.protocols.weights import WeightsDescriptor


class EpisodeRunnerError(RuntimeError):
    """An episode runner cannot satisfy the whole-episode contract."""


class WeightVersionError(ValueError):
    """A weight publication is incompatible with the runner's current state."""


@dataclass(frozen=True, slots=True)
class EpisodeRolloutMetrics:
    """Metrics measured while collecting one complete episode."""

    episode_time_s: float
    episode_return: float
    env_steps: int
    agent_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_time_s, int | float)
            or isinstance(self.episode_time_s, bool)
            or not math.isfinite(self.episode_time_s)
            or self.episode_time_s < 0
        ):
            raise ValueError("episode_time_s must be finite and non-negative")
        if (
            not isinstance(self.episode_return, int | float)
            or isinstance(self.episode_return, bool)
            or not math.isfinite(self.episode_return)
        ):
            raise ValueError("episode_return must be finite")
        for name, value in (
            ("env_steps", self.env_steps),
            ("agent_steps", self.agent_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EpisodeRolloutResult:
    """One complete replay envelope and its collection metrics."""

    episode: EpisodeEnvelope
    metrics: EpisodeRolloutMetrics

    def __post_init__(self) -> None:
        if self.metrics.env_steps != self.episode.env_steps:
            raise ValueError("rollout metrics env_steps do not match the episode")
        if self.metrics.agent_steps != self.episode.agent_steps:
            raise ValueError("rollout metrics agent_steps do not match the episode")


@dataclass(frozen=True, slots=True)
class _TimeLimitedEnvCreator:
    creator: Callable[[Mapping[str, Any]], gym.Env]
    max_episode_steps: int

    def __call__(self, env_context: Mapping[str, Any]) -> gym.Env:
        env = self.creator(env_context)
        if not isinstance(env, gym.Env):
            raise TypeError("environment creator must return a gymnasium.Env")
        return gym.wrappers.TimeLimit(
            env,
            max_episode_steps=self.max_episode_steps,
        )


def make_episode_id(
    member_id: str,
    runner_id: str,
    runner_generation: int,
    local_episode_seq: int,
) -> str:
    """Build an unambiguous, retry-stable whole-episode identity."""

    for name, value in (("member_id", member_id), ("runner_id", runner_id)):
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(f"{name} must be a non-empty path segment")
    for name, value in (
        ("runner_generation", runner_generation),
        ("local_episode_seq", local_episode_seq),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return f"{member_id}/{runner_id}/{runner_generation}/{local_episode_seq}"


def _accept_weight_publication(
    current: WeightsDescriptor | None,
    candidate: WeightsDescriptor,
    *,
    member_id: str,
) -> bool:
    if not isinstance(candidate, WeightsDescriptor):
        raise TypeError("weights must be a WeightsDescriptor")
    if candidate.member_id != member_id:
        raise WeightVersionError("weights belong to a different member")
    if set(candidate.module_versions) != {DEFAULT_MODULE_ID}:
        raise WeightVersionError("Phase 5 supports exactly one default_policy module")
    if not isinstance(candidate.state, Mapping) or set(candidate.state) != {
        DEFAULT_MODULE_ID
    }:
        raise WeightVersionError(
            "weight state must contain exactly the default_policy module"
        )
    if current is None:
        return True

    current_version = current.module_versions[DEFAULT_MODULE_ID]
    candidate_version = candidate.module_versions[DEFAULT_MODULE_ID]
    if candidate_version < current_version:
        return False
    if candidate_version == current_version:
        if candidate.learner_updates != current.learner_updates:
            raise WeightVersionError(
                "one module version cannot identify multiple publications"
            )
        return False
    if candidate.learner_updates < current.learner_updates:
        raise WeightVersionError(
            "newer module versions cannot have an older learner update"
        )
    return True


class EpisodeRunner:
    """Collect exactly one complete single-agent episode per call."""

    def __init__(
        self,
        config: AlgorithmConfig,
        codec: FlatEpisodeCodec,
        *,
        member_id: str,
        runner_id: str,
        runner_generation: int,
        max_episode_steps: int,
        initial_weights: WeightsDescriptor,
        worker_index: int,
    ) -> None:
        if not isinstance(config, AlgorithmConfig):
            raise TypeError("config must be an AlgorithmConfig")
        if not isinstance(codec, FlatEpisodeCodec):
            raise TypeError("Phase 5 EpisodeRunner requires FlatEpisodeCodec")
        if (
            not isinstance(max_episode_steps, int)
            or isinstance(max_episode_steps, bool)
            or max_episode_steps < 1
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        if (
            not isinstance(worker_index, int)
            or isinstance(worker_index, bool)
            or worker_index < 0
        ):
            raise ValueError("worker_index must be a non-negative integer")
        make_episode_id(member_id, runner_id, runner_generation, 0)
        if config.num_envs_per_env_runner != 1:
            raise ValueError("EpisodeRunner requires exactly one environment")
        if not config.enable_rl_module_and_learner:
            raise ValueError("EpisodeRunner requires the RLModule API stack")
        if not config.enable_env_runner_and_connector_v2:
            raise ValueError("EpisodeRunner requires ConnectorV2")

        self._codec = codec
        self._member_id = member_id
        self._runner_id = runner_id
        self._runner_generation = runner_generation
        self._max_episode_steps = max_episode_steps
        self._local_episode_seq = 0
        self._closed = False
        self._weights: WeightsDescriptor | None = None

        runner_config = self._runner_config(config, max_episode_steps)
        self._env_runner = SingleAgentEnvRunner(
            config=runner_config,
            worker_index=worker_index,
        )
        try:
            self.install_weights(initial_weights)
        except Exception:
            self._env_runner.stop()
            self._closed = True
            raise

    @property
    def member_id(self) -> str:
        return self._member_id

    @property
    def runner_id(self) -> str:
        return self._runner_id

    @property
    def runner_generation(self) -> int:
        return self._runner_generation

    @property
    def local_episode_seq(self) -> int:
        return self._local_episode_seq

    @property
    def behavior_versions(self) -> FrozenVersions:
        self._require_open()
        assert self._weights is not None
        return FrozenVersions(self._weights.module_versions)

    def install_weights(self, weights: WeightsDescriptor) -> bool:
        """Install a strictly newer publication at an episode boundary."""

        self._require_open()
        if not _accept_weight_publication(
            self._weights,
            weights,
            member_id=self._member_id,
        ):
            return False

        version = weights.module_versions[DEFAULT_MODULE_ID]
        assert isinstance(weights.state, Mapping)
        self._env_runner.set_state(
            {
                COMPONENT_RL_MODULE: copy.deepcopy(dict(weights.state)),
                WEIGHTS_SEQ_NO: version,
            }
        )
        self._weights = copy.deepcopy(weights)
        return True

    def collect_episode(
        self,
        weights: WeightsDescriptor | None = None,
        *,
        explore: bool = True,
    ) -> EpisodeRolloutResult:
        """Apply optional fresh weights, then collect one complete episode."""

        self._require_open()
        if weights is not None:
            self.install_weights(weights)
        assert self._weights is not None

        started = time.monotonic()
        episodes = self._env_runner.sample(num_episodes=1, explore=explore)
        elapsed = time.monotonic() - started
        if len(episodes) != 1:
            raise EpisodeRunnerError(
                f"RLlib returned {len(episodes)} episodes instead of one"
            )
        source = episodes[0]
        step_count = len(source)
        if step_count < 1:
            raise EpisodeRunnerError("RLlib returned an empty episode")
        if step_count > self._max_episode_steps:
            raise EpisodeRunnerError(
                "episode exceeded the configured max_episode_steps"
            )
        if not source.is_done or not (source.is_terminated or source.is_truncated):
            raise EpisodeRunnerError("RLlib returned an incomplete episode")

        expected_version = self._weights.module_versions[DEFAULT_MODULE_ID]
        try:
            observed_versions = np.asarray(
                source.get_extra_model_outputs(WEIGHTS_SEQ_NO),
            )
        except KeyError as error:
            raise EpisodeRunnerError(
                "episode does not contain RLlib weight sequence metadata"
            ) from error
        if (
            observed_versions.shape != (step_count,)
            or observed_versions.dtype.kind not in "iu"
            or np.any(observed_versions != expected_version)
        ):
            raise EpisodeRunnerError(
                "one episode must use exactly one installed behavior version"
            )

        observations = np.asarray(source.get_observations())
        actions = np.asarray(source.get_actions())
        rewards = np.asarray(source.get_rewards())
        if observations.shape[0] != step_count + 1:
            raise EpisodeRunnerError("episode observations are not transition-aligned")
        if actions.shape[0] != step_count or rewards.shape != (step_count,):
            raise EpisodeRunnerError("episode actions or rewards are not aligned")
        if not np.isfinite(rewards).all():
            raise EpisodeRunnerError("episode rewards must be finite")

        transitions = []
        for index in range(step_count):
            is_last = index == step_count - 1
            transitions.append(
                {
                    Columns.OBS: np.array(observations[index], copy=True),
                    Columns.NEXT_OBS: np.array(observations[index + 1], copy=True),
                    Columns.ACTIONS: np.array(actions[index], copy=True),
                    Columns.REWARDS: float(rewards[index]),
                    Columns.TERMINATEDS: bool(is_last and source.is_terminated),
                    Columns.TRUNCATEDS: bool(is_last and source.is_truncated),
                }
            )

        payload = self._codec.encode(transitions)
        episode_id = make_episode_id(
            self._member_id,
            self._runner_id,
            self._runner_generation,
            self._local_episode_seq,
        )
        envelope = EpisodeEnvelope(
            episode_id=episode_id,
            schema_version=self._codec.schema_version,
            producer_member_id=self._member_id,
            runner_id=self._runner_id,
            runner_generation=self._runner_generation,
            local_episode_seq=self._local_episode_seq,
            behavior_versions=FrozenVersions(self._weights.module_versions),
            env_steps=step_count,
            agent_steps=step_count,
            terminated=bool(source.is_terminated),
            truncated=bool(source.is_truncated),
            estimated_bytes=payload.estimated_bytes,
            payload=payload,
        )
        self._codec.validate(envelope)
        metrics = EpisodeRolloutMetrics(
            episode_time_s=elapsed,
            episode_return=float(np.sum(rewards, dtype=np.float64)),
            env_steps=step_count,
            agent_steps=step_count,
        )
        self._local_episode_seq += 1
        return EpisodeRolloutResult(episode=envelope, metrics=metrics)

    def close(self) -> None:
        if self._closed:
            return
        self._env_runner.stop()
        self._closed = True

    def __enter__(self) -> EpisodeRunner:
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise EpisodeRunnerError("episode runner is closed")

    @staticmethod
    def _runner_config(
        config: AlgorithmConfig,
        max_episode_steps: int,
    ) -> AlgorithmConfig:
        runner_config = config.copy(copy_frozen=False)
        runner_config.env_runners(
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            episodes_to_numpy=True,
        )
        env_config = dict(runner_config.env_config)
        env = runner_config.env
        if isinstance(env, str) and _global_registry.contains(ENV_CREATOR, env):
            runner_config.environment(
                env=_TimeLimitedEnvCreator(
                    _global_registry.get(ENV_CREATOR, env),
                    max_episode_steps,
                ),
                env_config=env_config,
            )
        elif callable(env):
            runner_config.environment(
                env=_TimeLimitedEnvCreator(
                    env,
                    max_episode_steps,
                ),
                env_config=env_config,
            )
        else:
            env_config["max_episode_steps"] = max_episode_steps
            runner_config.environment(
                env=env,
                env_config=env_config,
            )
        return runner_config
