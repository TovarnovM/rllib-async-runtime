"""Native MPE semantics, graph visibility and DQN/SAC routing."""

import gymnasium as gym
import numpy as np
import pytest
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig

pytest.importorskip("mpe2")

from mpe2 import simple_spread_v3  # noqa: E402
from pettingzoo.test import parallel_api_test  # noqa: E402

from rllib_async.examples.marl_spread import (  # noqa: E402
    GraphSpreadEnv,
    GraphSpreadParallelEnv,
    build_spread_configs,
)
from rllib_async.runtime import AsyncRuntimeConfig  # noqa: E402
from rllib_async.runtime.team import prepare_team_config  # noqa: E402


@pytest.mark.parametrize("discrete", [0, 2])
def test_spread_parallel_api_and_finite_seeded_graphs(discrete):
    env = GraphSpreadParallelEnv(agents=4, discrete_agents=discrete, max_cycles=4)
    try:
        parallel_api_test(env, num_cycles=12)
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        for agent, graph in first.items():
            assert env.observation_space(agent).contains(graph)
            for key, value in graph.items():
                np.testing.assert_array_equal(value, second[agent][key])
            assert graph["node_count"] == 8
            assert graph["edge_count"] == 14
    finally:
        env.close()


def test_discrete_codebook_matches_native_discrete_physics_and_rewards():
    wrapped = GraphSpreadParallelEnv(agents=3, discrete_agents=2, max_cycles=6)
    native = simple_spread_v3.parallel_env(N=3, max_cycles=6, continuous_actions=False)
    try:
        wrapped.reset(seed=31)
        native.reset(seed=31)
        for action in range(5):
            expected = native.step(dict.fromkeys(native.agents, action))
            actual = wrapped.step({
                "agent_0": action,
                "agent_1": action,
                "agent_2": np.eye(5, dtype=np.float32)[action],
            })
            assert actual[1:4] == expected[1:4]
            for agent, raw in expected[0].items():
                graph = actual[0][agent]["node_features"]
                np.testing.assert_allclose(graph[0, 2:4], raw[:2])
                np.testing.assert_allclose(graph[3:, :2], raw[4:10].reshape(-1, 2))
                np.testing.assert_allclose(graph[1:3, :2], raw[10:14].reshape(-1, 2))
                assert not graph[1:, 2:4].any()  # Other velocities are unobserved.
    finally:
        wrapped.close()
        native.close()


def test_rllib_adapter_preserves_time_limit_and_physical_clock():
    env = GraphSpreadEnv({"agents": 2, "discrete_agents": 1, "max_cycles": 2})
    try:
        env.reset(seed=7)
        for step in (1, 2):
            _, _, terminated, truncated, _ = env.step({
                "agent_0": 0, "agent_1": np.zeros(5, dtype=np.float32)
            })
            assert env.world_t == step
            assert not terminated["__all__"]
            assert truncated["__all__"] == (step == 2)
        env.reset(seed=7)
        assert env.world_t == 0
    finally:
        env.close()


@pytest.mark.parametrize("discrete", [0, 2])
def test_spread_uses_dqn_for_discrete_agents_and_supports_shared_gpu(discrete):
    rollout, modules = build_spread_configs(
        agents=4, discrete_agents=discrete, num_gpus=0.25, hidden_dim=8
    )
    _, prepared, spaces, _, _ = prepare_team_config(
        rollout, modules, AsyncRuntimeConfig(num_gpus_per_learner=0.25)
    )
    assert isinstance(prepared["sac"], SACConfig)
    assert isinstance(spaces["sac"][1], gym.spaces.Box)
    if discrete:
        assert isinstance(prepared["dqn"], DQNConfig)
        assert isinstance(spaces["dqn"][1], gym.spaces.Discrete)
    else:
        assert set(prepared) == {"sac"}
    assert rollout.policy_mapping_fn("agent_0") == ("dqn" if discrete else "sac")
    assert rollout.policy_mapping_fn("agent_3") == "sac"
