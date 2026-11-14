from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ray.rllib.core.columns import Columns

from rllib_async.examples.temporal_hierarchy import (
    CausalHierarchyEnv,
    CueMemoryEnv,
    temporal_policy_mapping_fn,
)
from rllib_async.gnn.episodes import NODE_FEATURES
from rllib_async.rollout.multi_module import _raw_decisions, _RawStep


def _features(observation):
    return (
        observation[NODE_FEATURES][0] if isinstance(observation, dict) else observation
    )


def _run_scripted_environment(env):
    observations, _ = env.reset(seed=7)
    steps = []
    while True:
        active = [agent for agent in observations if agent in env.agents]
        assert len(active) == 1
        agent = active[0]
        role = temporal_policy_mapping_fn(agent)
        action = (
            int(_features(observations[agent])[2] > 0)
            if role == "manager"
            else 0
            if role == "worker_0"
            else np.asarray([1.0], np.float32)
        )
        world_t = env.world_t
        observations, rewards, terminateds, truncateds, infos = env.step(
            {agent: action}
        )
        steps.append(
            _RawStep(
                world_t,
                env.world_t,
                {agent: action},
                rewards,
                frozenset(observations),
                terminateds,
                truncateds,
                frozenset(
                    agent
                    for agent, info in infos.items()
                    if info.get("subtask_terminated")
                ),
            )
        )
        if terminateds["__all__"] or truncateds["__all__"]:
            break
    return steps, _raw_decisions(steps)


def test_causal_hierarchy_manager_routes_before_the_first_worker_action():
    env = CausalHierarchyEnv({"episode_length": 6, "vector_worker": True})
    observations, _ = env.reset(seed=7)
    assert set(observations) == {"manager"}
    observations, rewards, terminateds, _, _ = env.step({"manager": 1})
    assert set(observations) == {"worker_1:0"}
    assert _features(observations["worker_1:0"])[2] == pytest.approx(0.6)
    assert env.world_t == 0
    assert rewards == {}
    assert not terminateds["__all__"]
    with pytest.raises(ValueError, match="currently active"):
        env.step({"worker_0:0": 0})
    assert env.world_t == 0
    env.step({"worker_1:0": np.asarray([0.5], np.float32)})
    assert env.world_t == 1


@pytest.mark.parametrize("variable_duration", [False, True])
def test_raw_manager_intervals_keep_physical_rewards_and_local_boundaries(
    variable_duration,
):
    env = CausalHierarchyEnv(
        {
            "episode_length": 8,
            "manager_period": 3,
            "variable_duration": variable_duration,
        }
    )
    steps, decisions = _run_scripted_environment(env)
    manager = [value for (agent, _), value in decisions.items() if agent == "manager"]
    assert sum(value["duration"] for value in manager) == 8
    assert [value["duration"] for value in manager] == (
        [2, 3, 3] if variable_duration else [3, 3, 2]
    )
    assert all(
        value["state_reset"] == (index == 0) for index, value in enumerate(manager)
    )
    assert all(not value["subtask_terminated"] for value in manager)
    assert all(not value[Columns.TERMINATEDS] for value in manager[:-1])
    assert manager[-1][Columns.TERMINATEDS]
    assert sum(len(step.actions) for step in steps) == len(decisions)
    assert len(steps) == 8 + len(manager)
    for value in manager:
        expected = tuple(
            step.rewards.get("manager", 0.0)
            for step in steps
            if value["world_t"]
            < step.next_world_t
            <= value["world_t"] + value["duration"]
            and step.next_world_t > step.world_t
        )
        assert value["raw_rewards"] == expected
    workers = [(key, value) for key, value in decisions.items() if key[0] != "manager"]
    assert len(workers) == 8
    assert all(value["duration"] == 1 for _, value in workers)
    assert all(value["state_reset"] == (key[1] == 0) for key, value in workers)
    assert any(
        value["subtask_terminated"] and not value[Columns.TERMINATEDS]
        for _, value in workers
    )
    ids = Counter(key[0] for key, _ in workers)
    assert len(ids) == len(manager)


def test_time_limit_keeps_real_final_state_and_manager_bootstrap():
    env = CausalHierarchyEnv(
        {"episode_length": 4, "manager_period": 3, "time_limit": True}
    )
    _, decisions = _run_scripted_environment(env)
    final = decisions[("manager", 1)]
    assert final[Columns.TRUNCATEDS]
    assert not final[Columns.TERMINATEDS]
    assert not final["subtask_terminated"]
    assert final["world_t"] + final["duration"] == 4


@pytest.mark.parametrize(("episode_length", "goal_finished"), [(5, False), (6, True)])
def test_time_limit_distinguishes_interrupted_and_completed_local_goals(
    episode_length, goal_finished
):
    env = CausalHierarchyEnv(
        {
            "episode_length": episode_length,
            "manager_period": 3,
            "variable_duration": False,
            "time_limit": True,
        }
    )
    _, decisions = _run_scripted_environment(env)
    manager = decisions[("manager", 1)]
    worker = next(
        value
        for (agent, _), value in decisions.items()
        if agent != "manager" and value["world_t"] + value["duration"] == episode_length
    )
    assert manager[Columns.TRUNCATEDS]
    assert not manager[Columns.TERMINATEDS]
    assert not manager["subtask_terminated"]
    assert worker[Columns.TRUNCATEDS]
    assert not worker[Columns.TERMINATEDS]
    assert worker["subtask_terminated"] is goal_finished


@pytest.mark.parametrize("graph", [False, True])
def test_delayed_cue_does_not_leak_into_query_observation(graph):
    env = CueMemoryEnv({"graph": graph, "episode_length": 4})
    observations_by_cue = {}
    returns = {}
    for seed in range(20):
        observations, infos = env.reset(seed=seed)
        cue = float(_features(observations["memory"])[0])
        assert infos == {}
        if cue in observations_by_cue:
            continue
        hidden_observations = []
        for step in range(4):
            action = int(cue > 0) if step == 3 else 0
            observations, rewards, terminateds, _, infos = env.step({"memory": action})
            assert infos == {}
            hidden_observations.append(observations["memory"])
        assert terminateds["__all__"]
        observations_by_cue[cue] = hidden_observations
        returns[cue] = rewards["memory"]
    assert set(observations_by_cue) == {-1.0, 1.0}
    assert returns == {-1.0: 1.0, 1.0: 1.0}
    for negative, positive in zip(
        observations_by_cue[-1.0], observations_by_cue[1.0], strict=True
    ):
        if graph:
            for key in negative:
                np.testing.assert_array_equal(negative[key], positive[key])
        else:
            np.testing.assert_array_equal(negative, positive)
