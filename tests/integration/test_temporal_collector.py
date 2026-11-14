from __future__ import annotations

from collections import Counter

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.policy import PolicySpec

from rllib_async.examples.temporal_hierarchy import (
    CausalHierarchyEnv,
    CueMemoryEnv,
    build_memory_configs,
    build_temporal_hierarchy_configs,
)
from rllib_async.gnn.episodes import ACTION_MASK, NODE_FEATURES
from rllib_async.protocols.episodes import RawMultiModuleEpisodeCodec
from rllib_async.protocols.weights import WeightsDescriptor
from rllib_async.rollout.multi_module import MultiModuleEpisodeRunner


class _ScriptedModule(TorchRLModule):
    """Fixed actions make the environment-to-replay numerical oracle exact."""

    def setup(self):
        self.observed_clocks = []

    def _forward_inference(self, batch, **kwargs):
        del kwargs
        observations = batch[Columns.OBS]
        if isinstance(observations, dict) and ACTION_MASK in observations:
            assert observations[ACTION_MASK].any(dim=-1).all(), (
                "terminal observations must not request another action"
            )
        features = (
            observations[NODE_FEATURES][:, 0]
            if isinstance(observations, dict)
            else observations
        )
        if self.model_config["role"] == "manager":
            actions = (features[:, 2] > 0).long()
        elif isinstance(self.action_space, gym.spaces.Discrete):
            actions = torch.zeros(features.shape[0], dtype=torch.int64)
        else:
            actions = torch.full(
                (features.shape[0], 1),
                self.model_config.get("action", 1.0),
                dtype=torch.float32,
            )
        return {Columns.ACTIONS: actions}

    def _forward_exploration(self, batch, t, **kwargs):
        self.observed_clocks.append(t)
        return self._forward_inference(batch, **kwargs)


class _TerminalMaskHierarchyEnv(CausalHierarchyEnv):
    def __init__(self, config=None):
        super().__init__(config)
        observation_space, action_space = self._spaces["worker_0"]
        self._spaces["worker_0"] = (
            gym.spaces.Dict(
                {
                    **observation_space.spaces,
                    ACTION_MASK: gym.spaces.Box(0, 1, (2,), np.float32),
                }
            ),
            action_space,
        )
        self.observation_spaces = {
            agent: self.get_observation_space(agent) for agent in self.possible_agents
        }

    def _observation(self, agent_id):
        observation = super()._observation(agent_id)
        if agent_id.startswith("worker_0:"):
            observation[ACTION_MASK] = np.ones(2, np.float32)
        return observation

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = super().step(actions)
        for agent, terminal in terminateds.items():
            if terminal and agent.startswith("worker_0:"):
                observations[agent][ACTION_MASK] = np.zeros(2, np.float32)
        return observations, rewards, terminateds, truncateds, infos


@pytest.mark.integration
@pytest.mark.parametrize("time_limit", [False, True])
def test_raw_collector_preserves_actual_step_rewards_and_decision_clocks(time_limit):
    config, _, spaces = build_temporal_hierarchy_configs(
        episode_length=8,
        manager_period=3,
        time_limit=time_limit,
    )
    config.environment(_TerminalMaskHierarchyEnv)
    env = _TerminalMaskHierarchyEnv(config.env_config)
    spaces["worker_0"] = (
        env.get_observation_space("worker_0:0"),
        env.get_action_space("worker_0:0"),
    )
    config.multi_agent(
        policies={
            module_id: PolicySpec(observation_space=obs, action_space=act)
            for module_id, (obs, act) in spaces.items()
        }
    )
    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                module_id: RLModuleSpec(
                    module_class=_ScriptedModule,
                    observation_space=obs,
                    action_space=act,
                    model_config={"role": module_id},
                )
                for module_id, (obs, act) in spaces.items()
            }
        )
    )
    weights = WeightsDescriptor(
        member_id="temporal-oracle",
        module_versions={module_id: 0 for module_id in spaces},
        learner_updates=0,
        published_at_monotonic=0,
        state={module_id: {} for module_id in spaces},
    )
    codec = RawMultiModuleEpisodeCodec()
    runner = MultiModuleEpisodeRunner(
        config,
        codec,
        member_id="temporal-oracle",
        runner_id="runner",
        runner_generation=0,
        max_episode_steps=16,
        module_ids=tuple(spaces),
        initial_weights=weights,
        worker_index=0,
    )
    try:
        result = runner.collect_episode(
            explore=True,
            sampled_module_steps={"manager": 10, "worker_0": 100, "worker_1": 200},
        )
        manager = [
            codec.get_module_transition(result.episode, "manager", index)
            for index in range(codec.module_transition_count(result.episode, "manager"))
        ]
        assert [transition.data["duration"] for transition in manager] == [2, 3, 3]
        raw_steps = runner._env_runner.raw_steps
        for transition in manager:
            data = transition.data
            rewards = tuple(
                step.rewards["manager"]
                for step in raw_steps
                if step.next_world_t > step.world_t
                and data["world_t"]
                < step.next_world_t
                <= data["world_t"] + data["duration"]
            )
            assert data["raw_rewards"] == rewards
            assert data[Columns.REWARDS] == pytest.approx(sum(rewards))
            assert not data["subtask_terminated"]
        assert manager[-1].data[Columns.TRUNCATEDS] is time_limit
        assert manager[-1].data[Columns.TERMINATEDS] is not time_limit
        all_transitions = [
            codec.get_transition(result.episode, index)
            for index in range(result.episode.agent_steps)
        ]
        assert result.metrics.world_steps == 8
        assert result.episode.env_steps == 11
        assert result.episode.agent_steps == 11
        assert dict(result.metrics.module_steps) == {
            "manager": 3,
            "worker_0": 5,
            "worker_1": 3,
        }
        assert Counter(item.agent_id for item in all_transitions) == {
            "manager": 3,
            "worker_0:0": 2,
            "worker_0:1": 3,
            "worker_1:2": 3,
        }
        local_finals = [
            item.data[Columns.NEXT_OBS][ACTION_MASK]
            for item in all_transitions
            if item.module_id == "worker_0" and item.data["subtask_terminated"]
        ]
        assert len(local_finals) == 2
        assert all(not mask.any() for mask in local_finals)
        modules = runner._env_runner.module
        assert all(modules[module_id].inference_only for module_id in spaces)
        assert modules["manager"].observed_clocks == [10, 11, 12]
        assert modules["worker_0"].observed_clocks == list(range(100, 105))
        assert modules["worker_1"].observed_clocks == list(range(200, 203))
        final_obs = manager[-1].data[Columns.NEXT_OBS][NODE_FEATURES]
        assert final_obs[0, 3] == pytest.approx(1.0)
        assert not np.array_equal(
            final_obs, manager[0].data[Columns.OBS][NODE_FEATURES]
        )
    finally:
        runner.close()


class _ScaledActionEnv(CueMemoryEnv):
    def __init__(self, config=None):
        super().__init__({"continuous": True, "episode_length": 3})
        self.action_spaces["memory"] = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.executed = []

    def reset(self, **kwargs):
        self.executed = []
        return super().reset(**kwargs)

    def step(self, actions):
        self.executed.append(np.array(actions["memory"], copy=True))
        return super().step(actions)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("normalize", "proposal", "physical", "training"),
    [(True, 0.25, 0.5, 0.25), (True, 2.0, 2.0, 1.0), (False, 3.0, 2.0, 2.0)],
)
def test_raw_collector_maps_executed_actions_back_to_the_learner_domain(
    normalize, proposal, physical, training
):
    config, _, _ = build_memory_configs(algorithm="sac", recurrent=False)
    observation_space = gym.spaces.Box(-2.0, 2.0, (4,), np.float32)
    action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
    config.environment(_ScaledActionEnv, normalize_actions=normalize, clip_actions=True)
    config.multi_agent(
        policies={
            "memory": PolicySpec(
                observation_space=observation_space, action_space=action_space
            )
        }
    )
    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "memory": RLModuleSpec(
                    module_class=_ScriptedModule,
                    observation_space=observation_space,
                    action_space=action_space,
                    model_config={"role": "memory", "action": proposal},
                )
            }
        )
    )
    codec = RawMultiModuleEpisodeCodec()
    runner = MultiModuleEpisodeRunner(
        config,
        codec,
        member_id="scaled-actions",
        runner_id="runner",
        runner_generation=0,
        max_episode_steps=3,
        module_ids=("memory",),
        initial_weights=WeightsDescriptor(
            member_id="scaled-actions",
            module_versions={"memory": 0},
            learner_updates=0,
            published_at_monotonic=0,
            state={"memory": {}},
        ),
        worker_index=0,
    )
    try:
        result = runner.collect_episode(explore=False)
        env = runner._env_runner.env.envs[0].unwrapped
        assert len(env.executed) == 3
        for index, executed in enumerate(env.executed):
            np.testing.assert_array_equal(executed, [physical])
            data = codec.get_transition(result.episode, index).data
            np.testing.assert_array_equal(data["executed_actions"], [physical])
            np.testing.assert_array_equal(data[Columns.ACTIONS], [training])
    finally:
        runner.close()


@pytest.mark.integration
def test_complete_episode_collection_advances_the_seeded_cue_sequence():
    seed = 20261114
    config, _, spaces = build_memory_configs(
        recurrent=False, episode_length=3, seed=seed
    )
    observation_space, action_space = spaces["memory"]
    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "memory": RLModuleSpec(
                    module_class=_ScriptedModule,
                    observation_space=observation_space,
                    action_space=action_space,
                    model_config={"role": "memory"},
                )
            }
        )
    )
    codec = RawMultiModuleEpisodeCodec()
    observed = []
    for _ in range(2):
        runner = MultiModuleEpisodeRunner(
            config,
            codec,
            member_id="cue-sequence",
            runner_id="runner",
            runner_generation=0,
            max_episode_steps=3,
            module_ids=("memory",),
            initial_weights=WeightsDescriptor(
                member_id="cue-sequence",
                module_versions={"memory": 0},
                learner_updates=0,
                published_at_monotonic=0,
                state={"memory": {}},
            ),
            worker_index=0,
        )
        try:
            cues = []
            for _ in range(12):
                result = runner.collect_episode(explore=False)
                first = codec.get_transition(result.episode, 0)
                cues.append(float(first.data[Columns.OBS][0]))
            observed.append(cues)
        finally:
            runner.close()
    expected_rng = np.random.default_rng(seed)
    expected = [float(expected_rng.choice((-1, 1))) for _ in range(12)]
    assert observed[0] == observed[1] == expected
    assert set(observed[0]) == {-1.0, 1.0}
