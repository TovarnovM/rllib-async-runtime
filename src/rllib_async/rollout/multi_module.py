"""Whole-episode adapter for RLlib sparse multi-agent/module rollouts."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence

import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.core.columns import Columns
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FrozenVersions,
    MultiModuleEpisodeCodec,
    MultiModuleTransition,
)
from rllib_async.protocols.weights import WeightsDescriptor
from rllib_async.rollout.episode_runner import (
    EpisodeRolloutMetrics,
    EpisodeRolloutResult,
    EpisodeRunnerError,
    WeightVersionError,
    accept_weight_publication,
    make_episode_id,
)


class MultiModuleEpisodeRunner:
    """Collect one complete sparse MultiAgentEpisode into the replay contract."""

    def __init__(
        self,
        config: AlgorithmConfig,
        codec: MultiModuleEpisodeCodec,
        *,
        member_id: str,
        runner_id: str,
        runner_generation: int,
        max_episode_steps: int,
        module_ids: Sequence[str],
        initial_weights: WeightsDescriptor,
        worker_index: int,
    ) -> None:
        if not isinstance(config, AlgorithmConfig):
            raise TypeError("config must be an AlgorithmConfig")
        if not isinstance(codec, MultiModuleEpisodeCodec):
            raise TypeError("MultiModuleEpisodeRunner requires MultiModuleEpisodeCodec")
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
        resolved_module_ids = tuple(module_ids)
        if (
            not resolved_module_ids
            or len(resolved_module_ids) != len(set(resolved_module_ids))
            or any(
                not isinstance(module_id, str) or not module_id
                for module_id in resolved_module_ids
            )
        ):
            raise ValueError("module_ids must contain unique non-empty strings")
        make_episode_id(member_id, runner_id, runner_generation, 0)
        if config.num_envs_per_env_runner != 1:
            raise ValueError(
                "MultiModuleEpisodeRunner requires exactly one environment"
            )
        if not config.enable_rl_module_and_learner:
            raise ValueError("MultiModuleEpisodeRunner requires the RLModule API stack")
        if not config.enable_env_runner_and_connector_v2:
            raise ValueError("MultiModuleEpisodeRunner requires ConnectorV2")

        self._codec = codec
        self._member_id = member_id
        self._runner_id = runner_id
        self._runner_generation = runner_generation
        self._max_episode_steps = max_episode_steps
        self._module_ids = resolved_module_ids
        self._module_id_set = set(resolved_module_ids)
        self._local_episode_seq = 0
        self._closed = False
        self._weights: WeightsDescriptor | None = None
        runner_config = config.copy(copy_frozen=False)
        runner_config.env_runners(
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            episodes_to_numpy=True,
        )
        self._env_runner = MultiAgentEnvRunner(
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
    def local_episode_seq(self) -> int:
        return self._local_episode_seq

    @property
    def behavior_versions(self) -> FrozenVersions:
        self._require_open()
        assert self._weights is not None
        return FrozenVersions(self._weights.module_versions)

    def install_weights(self, weights: WeightsDescriptor) -> bool:
        """Install one complete, synchronously versioned module publication."""

        self._require_open()
        if not accept_weight_publication(
            self._weights,
            weights,
            member_id=self._member_id,
            module_ids=self._module_id_set,
        ):
            return False
        versions = set(weights.module_versions.values())
        if len(versions) != 1:
            raise WeightVersionError(
                "RLlib MultiAgentEnvRunner requires one synchronized weight "
                "sequence across all modules"
            )
        assert isinstance(weights.state, Mapping)
        self._env_runner.set_state(
            {
                COMPONENT_RL_MODULE: copy.deepcopy(dict(weights.state)),
                WEIGHTS_SEQ_NO: next(iter(versions)),
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
        env_steps = len(source)
        if env_steps < 1:
            raise EpisodeRunnerError("RLlib returned an empty episode")
        if env_steps > self._max_episode_steps:
            raise EpisodeRunnerError(
                "episode exceeded the configured max_episode_steps"
            )
        if not source.is_done or not (source.is_terminated or source.is_truncated):
            raise EpisodeRunnerError("RLlib returned an incomplete episode")

        actions_by_env_t = source.get_actions(return_list=True)
        if len(actions_by_env_t) != env_steps:
            raise EpisodeRunnerError(
                "multi-agent actions are not environment-step aligned"
            )
        agent_data: dict[str, dict[str, object]] = {}
        participating_modules: set[str] = set()
        total_return = 0.0
        for agent_id, agent_episode in source.agent_episodes.items():
            if not isinstance(agent_id, str) or not agent_id:
                raise EpisodeRunnerError("Phase 9 requires non-empty string agent IDs")
            module_id = source.module_for(agent_id)
            if module_id not in self._module_id_set:
                raise EpisodeRunnerError(
                    f"episode used unexpected module_id {module_id!r}"
                )
            observations = np.asarray(agent_episode.get_observations())
            actions = np.asarray(agent_episode.get_actions())
            rewards = np.asarray(agent_episode.get_rewards())
            count = len(agent_episode)
            if count < 1:
                raise EpisodeRunnerError(
                    f"agent {agent_id!r} did not produce a transition"
                )
            if observations.shape[0] != count + 1:
                raise EpisodeRunnerError(
                    f"agent {agent_id!r} observations are not transition-aligned"
                )
            if actions.shape[0] != count or rewards.shape != (count,):
                raise EpisodeRunnerError(
                    f"agent {agent_id!r} actions or rewards are not aligned"
                )
            if not np.isfinite(rewards).all():
                raise EpisodeRunnerError("episode rewards must be finite")
            try:
                observed_versions = np.asarray(
                    agent_episode.get_extra_model_outputs(WEIGHTS_SEQ_NO),
                )
            except KeyError as error:
                raise EpisodeRunnerError(
                    "episode does not contain RLlib weight sequence metadata"
                ) from error
            expected_version = self._weights.module_versions[module_id]
            if (
                observed_versions.shape != (count,)
                or observed_versions.dtype.kind not in "iu"
                or np.any(observed_versions != expected_version)
            ):
                raise EpisodeRunnerError(
                    "one episode must use one installed version per module"
                )
            agent_data[agent_id] = {
                "module_id": module_id,
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "terminated": bool(agent_episode.is_terminated),
                "truncated": bool(agent_episode.is_truncated),
                "count": count,
            }
            participating_modules.add(module_id)
            total_return += float(np.sum(rewards, dtype=np.float64))

        next_agent_t = {agent_id: 0 for agent_id in agent_data}
        transitions: list[MultiModuleTransition] = []
        for env_t, action_turn in enumerate(actions_by_env_t):
            if not isinstance(action_turn, Mapping) or not action_turn:
                raise EpisodeRunnerError(
                    "every environment step must contain an agent action"
                )
            for agent_id in action_turn:
                data = agent_data.get(agent_id)
                if data is None:
                    raise EpisodeRunnerError(
                        f"action turn references unknown agent_id {agent_id!r}"
                    )
                agent_t = next_agent_t[agent_id]
                count = data["count"]
                assert isinstance(count, int)
                if agent_t >= count:
                    raise EpisodeRunnerError(
                        f"agent {agent_id!r} has more actions than transitions"
                    )
                observations = data["observations"]
                actions = data["actions"]
                rewards = data["rewards"]
                assert isinstance(observations, np.ndarray)
                assert isinstance(actions, np.ndarray)
                assert isinstance(rewards, np.ndarray)
                is_last = agent_t == count - 1
                module_id = data["module_id"]
                assert isinstance(module_id, str)
                transitions.append(
                    MultiModuleTransition(
                        env_t=env_t,
                        agent_t=agent_t,
                        agent_id=agent_id,
                        module_id=module_id,
                        data={
                            Columns.OBS: np.array(
                                observations[agent_t],
                                copy=True,
                            ),
                            Columns.NEXT_OBS: np.array(
                                observations[agent_t + 1],
                                copy=True,
                            ),
                            Columns.ACTIONS: np.array(
                                actions[agent_t],
                                copy=True,
                            ),
                            Columns.REWARDS: float(rewards[agent_t]),
                            Columns.TERMINATEDS: bool(is_last and data["terminated"]),
                            Columns.TRUNCATEDS: bool(is_last and data["truncated"]),
                        },
                    )
                )
                next_agent_t[agent_id] += 1
        if any(
            next_agent_t[agent_id] != data["count"]
            for agent_id, data in agent_data.items()
        ):
            raise EpisodeRunnerError(
                "sparse environment turns do not cover every agent transition"
            )
        agent_steps = len(transitions)
        if agent_steps != source.agent_steps():
            raise EpisodeRunnerError(
                "extracted transition count does not match RLlib agent_steps"
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
            behavior_versions=FrozenVersions(
                {
                    module_id: self._weights.module_versions[module_id]
                    for module_id in participating_modules
                }
            ),
            env_steps=env_steps,
            agent_steps=agent_steps,
            terminated=bool(source.is_terminated),
            truncated=bool(source.is_truncated),
            estimated_bytes=payload.estimated_bytes,
            payload=payload,
        )
        self._codec.validate(envelope)
        metrics = EpisodeRolloutMetrics(
            episode_time_s=elapsed,
            episode_return=total_return,
            env_steps=env_steps,
            agent_steps=agent_steps,
        )
        self._local_episode_seq += 1
        return EpisodeRolloutResult(episode=envelope, metrics=metrics)

    def close(self) -> None:
        if self._closed:
            return
        self._env_runner.stop()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise EpisodeRunnerError("episode runner is closed")
