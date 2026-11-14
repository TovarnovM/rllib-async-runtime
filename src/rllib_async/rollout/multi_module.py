"""Whole-episode adapter for RLlib sparse multi-agent/module rollouts."""

from __future__ import annotations

import copy
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import COMPONENT_RL_MODULE
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule, MultiRLModuleSpec
from ray.rllib.env.env_runner import ENV_STEP_FAILURE
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO
from ray.rllib.utils.spaces.space_utils import normalize_action

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    FrozenVersions,
    MultiModuleEpisodeCodec,
    MultiModuleTransition,
    RawMultiModuleEpisodeCodec,
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


class _DecisionClockMultiRLModule(MultiRLModule):
    """Use each module's real decisions for its exploration schedule."""

    def setup(self) -> None:
        super().setup()
        self.decision_steps: dict[str, int] = {}
        self.publication_seq = 0

    def _with_publication(
        self, module_id: str, output: dict[str, Any]
    ) -> dict[str, Any]:
        # Ray 2.56.1 adds agent provenance only when at least one model-output
        # column remains after ACTIONS is consumed. Stateless DQN otherwise
        # emits actions alone. Carry the actual installed publication explicitly.
        anchor = output.get(Columns.ACTIONS, output.get(Columns.ACTION_DIST_INPUTS))
        if anchor is None:
            raise EpisodeRunnerError(
                "inference must produce actions or distribution inputs"
            )
        shape = anchor.shape[:2] if self[module_id].is_stateful() else anchor.shape[:1]
        output[WEIGHTS_SEQ_NO] = torch.full(
            shape, self.publication_seq, dtype=torch.int64, device=anchor.device
        )
        return output

    def _forward_inference(
        self, batch: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return {
            module_id: self._with_publication(
                module_id, self[module_id].forward_inference(module_batch, **kwargs)
            )
            for module_id, module_batch in batch.items()
            if module_id in self
        }

    def _forward_exploration(
        self, batch: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        kwargs.pop("t", None)
        return {
            module_id: self._with_publication(
                module_id,
                self[module_id].forward_exploration(
                    module_batch,
                    t=self.decision_steps.get(module_id, 0),
                    **kwargs,
                ),
            )
            for module_id, module_batch in batch.items()
            if module_id in self
        }


@dataclass(frozen=True)
class _RawStep:
    world_t: int
    next_world_t: int
    actions: Mapping[str, Any]
    rewards: Mapping[str, float]
    observed_agents: frozenset[str]
    terminateds: Mapping[str, bool]
    truncateds: Mapping[str, bool]
    subtask_terminated: frozenset[str]


class _RawRewardEnvRunner(MultiAgentEnvRunner):
    """Capture the pinned RLlib env-step seam before sparse reward aggregation.

    A temporal environment exposes integer ``world_t`` and advances by zero
    (routing) or one physical tick per call. This contract prevents treating an
    already aggregated reward as a recoverable sequence of physical rewards.
    """

    def _reset_envs(self, *args: Any, **kwargs: Any) -> Any:
        self.raw_steps: list[_RawStep] = []
        return super()._reset_envs(*args, **kwargs)

    def _try_env_reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Any:
        # Complete-episode sampling in Ray 2.56.1 repeatedly marks the next
        # reset as initial. Seed each actual environment once, then let its
        # RNG advance. A recreated environment receives the configured seed.
        if self.env is getattr(self, "_seeded_env", None):
            seed = None
        elif seed is None:
            seed = self._seed
        result = super()._try_env_reset(seed=seed, options=options)
        self._seeded_env = self.env
        return result

    def _world_t(self) -> int:
        value = getattr(self.env.envs[0].unwrapped, "world_t", None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EpisodeRunnerError(
                "raw temporal collection requires env.world_t, a non-negative integer"
            )
        return value

    def _try_env_step(self, actions: Any) -> Any:
        world_t = self._world_t()
        results = super()._try_env_step(actions)
        if results == ENV_STEP_FAILURE:
            return results
        next_world_t = self._world_t()
        if next_world_t - world_t not in (0, 1):
            raise EpisodeRunnerError(
                "one env.step must advance zero or one physical world tick"
            )
        observations, rewards, terminateds, truncateds, infos = results
        step_rewards = {agent: float(value) for agent, value in rewards[0].items()}
        if not all(np.isfinite(value) for value in step_rewards.values()):
            raise EpisodeRunnerError("raw rewards must be finite")
        if next_world_t == world_t and any(step_rewards.values()):
            raise EpisodeRunnerError("a zero-time routing turn cannot carry reward")
        self.raw_steps.append(
            _RawStep(
                world_t=world_t,
                next_world_t=next_world_t,
                actions=copy.deepcopy(actions[0]),
                rewards=step_rewards,
                observed_agents=frozenset(observations[0]),
                terminateds=dict(terminateds[0]),
                truncateds=dict(truncateds[0]),
                subtask_terminated=frozenset(
                    agent
                    for agent, info in infos[0].items()
                    if info.get("subtask_terminated", False)
                ),
            )
        )
        assert isinstance(self.module, _DecisionClockMultiRLModule)
        episode = self._ongoing_episodes[0]
        for agent in actions[0]:
            module_id = episode.module_for(agent)
            self.module.decision_steps[module_id] = (
                self.module.decision_steps.get(module_id, 0) + 1
            )
        return results


def _raw_decisions(steps: Sequence[_RawStep]) -> dict[tuple[str, int], dict[str, Any]]:
    """Close real decisions at the next observation, keeping each physical reward."""

    pending: dict[str, dict[str, Any]] = {}
    next_agent_t: Counter[str] = Counter()
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for env_t, step in enumerate(steps):
        for agent, action in step.actions.items():
            if agent in pending:
                raise EpisodeRunnerError(
                    "agent acted before its previous decision closed"
                )
            pending[agent] = {
                "env_t": env_t,
                "world_t": step.world_t,
                Columns.ACTIONS: action,
                "raw_rewards": [],
            }
        if step.next_world_t > step.world_t:
            for agent, decision in pending.items():
                decision["raw_rewards"].append(step.rewards.get(agent, 0.0))
        for agent in tuple(pending):
            done = bool(
                step.terminateds.get(agent, False)
                or step.terminateds.get("__all__", False)
                or step.truncateds.get(agent, False)
                or step.truncateds.get("__all__", False)
            )
            if agent not in step.observed_agents:
                if done:
                    raise EpisodeRunnerError(
                        "a completed decision requires its real final observation"
                    )
                continue
            decision = pending.pop(agent)
            duration = step.next_world_t - decision["world_t"]
            if duration < 1 or duration != len(decision["raw_rewards"]):
                raise EpisodeRunnerError(
                    "every training decision needs positive duration"
                )
            agent_t = next_agent_t[agent]
            local_terminal = agent in step.subtask_terminated
            decision.update(
                {
                    "duration": duration,
                    "raw_rewards": tuple(decision["raw_rewards"]),
                    "state_reset": agent_t == 0,
                    "subtask_terminated": local_terminal,
                    Columns.TERMINATEDS: bool(
                        step.terminateds.get("__all__", False)
                        or (step.terminateds.get(agent, False) and not local_terminal)
                    ),
                    Columns.TRUNCATEDS: bool(
                        step.truncateds.get("__all__", False)
                        or step.truncateds.get(agent, False)
                    ),
                }
            )
            completed[(agent, agent_t)] = decision
            next_agent_t[agent] += 1
    if pending:
        raise EpisodeRunnerError("whole episode contains unfinished agent decisions")
    return completed


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
        self._temporal = isinstance(codec, RawMultiModuleEpisodeCodec)
        if self._temporal:
            spec = runner_config.rl_module_spec
            if not isinstance(
                spec, MultiRLModuleSpec
            ) or spec.multi_rl_module_class not in (
                MultiRLModule,
                _DecisionClockMultiRLModule,
            ):
                raise ValueError(
                    "temporal collection requires a standard explicit MultiRLModuleSpec"
                )
            spec.multi_rl_module_class = _DecisionClockMultiRLModule
            # Ray 2.56.1 preserves flags on explicit per-module specs even when
            # the EnvRunner requests an inference-only MultiRLModule.
            spec.inference_only = True
            for module_spec in spec.rl_module_specs.values():
                module_spec.inference_only = True
            # AlgorithmConfig.rl_module_spec returns a merged copy, so changes
            # must be persisted through its supported setter before construction.
            runner_config.rl_module(rl_module_spec=spec)
        runner_class = _RawRewardEnvRunner if self._temporal else MultiAgentEnvRunner
        self._env_runner = runner_class(
            config=runner_config,
            worker_index=worker_index,
        )
        self._action_normalization_spaces = {}
        if self._temporal and runner_config.normalize_actions:
            for module_id in self._module_ids:
                action_space = self._env_runner.module[module_id].action_space
                if (
                    isinstance(action_space, gym.spaces.Box)
                    and action_space.dtype in (np.float32, np.float64)
                    and np.all(action_space.bounded_below)
                    and np.all(action_space.bounded_above)
                ):
                    self._action_normalization_spaces[module_id] = action_space
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
        if self._temporal:
            self._env_runner.module.publication_seq = next(iter(versions))
        self._weights = weights
        return True

    def collect_episode(
        self,
        weights: WeightsDescriptor | None = None,
        *,
        explore: bool = True,
        sampled_module_steps: Mapping[str, int] | None = None,
    ) -> EpisodeRolloutResult:
        """Apply optional fresh weights, then collect one complete episode."""

        self._require_open()
        if weights is not None:
            self.install_weights(weights)
        assert self._weights is not None

        if self._temporal and sampled_module_steps is not None:
            module = self._env_runner.module
            assert isinstance(module, _DecisionClockMultiRLModule)
            if any(
                module_id not in self._module_id_set
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for module_id, count in sampled_module_steps.items()
            ):
                raise ValueError(
                    "sampled_module_steps must contain known module counters"
                )
            for module_id, count in sampled_module_steps.items():
                module.decision_steps[module_id] = max(
                    module.decision_steps.get(module_id, 0), count
                )

        started = time.monotonic()
        episodes = self._env_runner.sample(num_episodes=1, explore=explore)
        elapsed = time.monotonic() - started
        if len(episodes) != 1:
            raise EpisodeRunnerError(
                f"RLlib returned {len(episodes)} episodes instead of one"
            )
        source = episodes[0]
        raw_decisions = (
            _raw_decisions(self._env_runner.raw_steps) if self._temporal else None
        )
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
                raise EpisodeRunnerError(
                    "multi-module rollouts require non-empty string agent IDs"
                )
            module_id = source.module_for(agent_id)
            if module_id not in self._module_id_set:
                raise EpisodeRunnerError(
                    f"episode used unexpected module_id {module_id!r}"
                )
            observations = agent_episode.get_observations()
            actions = np.asarray(agent_episode.get_actions())
            rewards = np.asarray(agent_episode.get_rewards())
            count = len(agent_episode)
            if count < 1:
                raise EpisodeRunnerError(
                    f"agent {agent_id!r} did not produce a transition"
                )
            if _batched_tree_length(observations) != count + 1:
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
                            Columns.OBS: _copy_batched_tree_item(
                                observations,
                                agent_t,
                            ),
                            Columns.NEXT_OBS: _copy_batched_tree_item(
                                observations,
                                agent_t + 1,
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
                if raw_decisions is not None:
                    raw = dict(raw_decisions[(agent_id, agent_t)])
                    if raw.pop("env_t") != env_t:
                        raise EpisodeRunnerError(
                            "raw decisions disagree with RLlib action turns"
                        )
                    if not np.isclose(sum(raw["raw_rewards"]), rewards[agent_t]):
                        raise EpisodeRunnerError(
                            "raw reward history disagrees with RLlib aggregation"
                        )
                    executed = raw[Columns.ACTIONS]
                    raw["executed_actions"] = copy.deepcopy(executed)
                    if module_id in self._action_normalization_spaces:
                        # Stock SAC evaluates candidate actions in the policy's
                        # normalized domain. Invert RLlib's official transport
                        # mapping after clipping, retaining the physical facts.
                        raw[Columns.ACTIONS] = normalize_action(
                            executed, self._action_normalization_spaces[module_id]
                        )
                    transitions[-1].data.update(raw)
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
            module_steps=tuple(
                sorted(Counter(item.module_id for item in transitions).items())
            ),
            world_steps=(
                self._env_runner.raw_steps[-1].next_world_t
                - self._env_runner.raw_steps[0].world_t
                if self._temporal
                else None
            ),
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


def _batched_tree_length(value: object) -> int:
    if isinstance(value, Mapping):
        if not value:
            raise EpisodeRunnerError("observation mappings must not be empty")
        lengths = {_batched_tree_length(child) for child in value.values()}
        if len(lengths) != 1:
            raise EpisodeRunnerError(
                "nested observation leaves must share one batch dimension"
            )
        return lengths.pop()
    array = np.asarray(value)
    if array.ndim < 1:
        raise EpisodeRunnerError("observation leaves must contain a batch dimension")
    return len(array)


def _copy_batched_tree_item(value: object, index: int) -> object:
    if isinstance(value, Mapping):
        return {
            key: _copy_batched_tree_item(child, index) for key, child in value.items()
        }
    array = np.asarray(value)
    if index < 0 or index >= len(array):
        raise EpisodeRunnerError("observation index is outside its batch")
    return np.array(array[index], copy=True)
