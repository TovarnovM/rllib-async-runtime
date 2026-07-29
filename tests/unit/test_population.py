from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import pytest
import ray

import rllib_async.runtime.population as population_module
from rllib_async.protocols import ReplayCursor, ReplayStats
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    PopulationAsyncSAC,
    PopulationLauncher,
    PopulationMemberSpec,
    PopulationTrainable,
    RuntimeState,
    SingleMemberAsyncSAC,
    make_runtime_member_id,
)
from tests.helpers import make_sac_config


def make_single_trial_specs(
    *,
    size: int = 2,
) -> tuple[PopulationMemberSpec, ...]:
    return tuple(
        PopulationMemberSpec(
            make_sac_config().debugging(seed=100 + index),
            AsyncSACRuntimeConfig(
                member_id=f"member-{index:02d}",
                runner_count=2,
                evaluation_interval_env_steps=0,
                evaluation_num_episodes=0,
                report_interval_s=0.1,
                seed=100 + index,
                num_cpus_per_replay=1,
                num_cpus_per_runner=0.5,
                num_cpus_per_learner=1,
            ),
        )
        for index in range(size)
    )


def test_population_rejects_incompatible_observation_action_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    members = (
        PopulationMemberSpec(
            make_sac_config().environment("Pendulum-v1"),
            AsyncSACRuntimeConfig(member_id="member-0"),
        ),
        PopulationMemberSpec(
            make_sac_config().environment("CartPole-v1"),
            AsyncSACRuntimeConfig(member_id="member-1"),
        ),
    )

    with pytest.raises(ValueError, match="observation/action spaces"):
        PopulationLauncher(
            members,
            replay_actor_name="incompatible-spaces",
            namespace="test",
        )


def test_single_trial_population_requires_two_unique_slots() -> None:
    one_member = make_single_trial_specs(size=1)
    with pytest.raises(ValueError, match="at least two"):
        PopulationTrainable._parse_config({"members": one_member})

    duplicate = (
        make_single_trial_specs()[0],
        make_single_trial_specs()[0],
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        PopulationTrainable._parse_config({"members": duplicate})


def test_single_trial_population_rejects_different_runtime_topology() -> None:
    first, second = make_single_trial_specs()
    incompatible = PopulationMemberSpec(
        second.sac_config,
        replace(second.runtime_config, runner_count=3),
    )

    with pytest.raises(ValueError, match="runtime topology"):
        PopulationTrainable._parse_config({"members": (first, incompatible)})


def test_runtime_member_id_is_generation_specific_and_deterministic() -> None:
    assert (
        make_runtime_member_id("run-test", "member-02", 4) == "run-test-member-02-g0004"
    )
    assert make_runtime_member_id("run-test", "member-02", 4) != (
        make_runtime_member_id("run-test", "member-02", 5)
    )
    with pytest.raises(ValueError, match="generation"):
        make_runtime_member_id("run-test", "member-02", -1)


def test_slash_delimited_metric_extraction_is_strict() -> None:
    report = {"train": {"episode_reward_mean": 3.5}}

    assert (
        PopulationAsyncSAC._extract_metric(
            report,
            "train/episode_reward_mean",
        )
        == 3.5
    )
    assert (
        PopulationAsyncSAC._extract_metric(
            report,
            "train/missing",
        )
        is None
    )
    assert PopulationAsyncSAC._extract_metric(report, "train/") is None


def test_single_trial_resource_request_reserves_whole_population() -> None:
    resources = PopulationTrainable.default_resource_request(
        {"members": make_single_trial_specs()}
    )

    assert resources.bundles == [
        {"CPU": 1.0},
        {"CPU": 1.0},
        {"CPU": 1.0},
        {"CPU": 0.5},
        {"CPU": 0.5},
        {"CPU": 1.0},
        {"CPU": 0.5},
        {"CPU": 0.5},
    ]


def test_population_report_namespaces_members_and_shared_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    spaces = {
        "default_policy": (
            gym.spaces.Box(-1.0, 1.0, shape=(3,)),
            gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        )
    }
    monkeypatch.setattr(
        SingleMemberAsyncSAC,
        "_resolve_spaces",
        staticmethod(lambda _: spaces),
    )
    population = PopulationAsyncSAC(
        make_single_trial_specs(),
        run_id="run-test",
    )
    population._report_index = 1

    def member_report(score: float) -> dict[str, object]:
        return {
            "timesteps_this_iter": 4,
            "episodes_this_iter": 1,
            "train": {
                "episode_reward_mean": score,
                "episode_reward_min": score - 1,
                "episode_reward_max": score + 1,
                "episodes_in_window": 2,
                "episodes_since_metric_reset": 2,
            },
            "controller": {},
            "rollout": {
                "episodes_collected": 2,
                "env_steps_per_s": 5.0,
            },
            "fast_replay": {},
            "batching": {},
            "learner": {
                "learner_updates": 3,
                "samples_per_s": 8.0,
            },
            "evaluation": {"enabled": False},
        }

    replay = ReplayStats(
        cursor=ReplayCursor("shared", 7),
        episode_count=4,
        total_transitions=16,
        total_estimated_bytes=1_024,
        producer_episode_counts=(),
        producer_transition_counts=(),
        oldest_available_mutation_seq=1,
        journal_entries=4,
        deduplication_entries=4,
        commit_attempts=4,
        committed_episodes=4,
        duplicate_commits=0,
        rejected_commits=0,
        conflicting_commits=0,
        evicted_episodes=0,
    )
    result = population._format_report(
        {
            "member-00": member_report(1.0),
            "member-01": member_report(3.0),
        },
        replay,
    )

    assert result["timesteps_this_iter"] == 8
    assert result["population"] == {
        "run_id": "run-test",
        "report_index": 1,
        "size": 2,
        "eligible_members": 2,
        "best_score": 3.0,
        "mean_score": 2.0,
        "worst_score": 1.0,
        "exploit_count": 0,
        "reports_since_perturbation": 1,
        "last_exploit_duration_s": 0.0,
    }
    assert result["members"]["member-00"]["train"]["episode_reward_mean"] == 1.0
    assert result["members"]["member-00"]["learner"]["updates"] == 3
    assert result["members"]["member-01"]["pbt"]["generation"] == 0
    assert result["replay"] == {
        "store_generation": "shared",
        "cursor": 7,
        "episodes": 4,
        "transitions": 16,
        "bytes": 1_024,
        "insert_rate": 10.0,
        "sample_rate": 16.0,
        "committed_episodes": 4,
        "duplicate_commits": 0,
    }


def test_population_pump_rotates_first_member_without_blocking() -> None:
    calls: list[tuple[str, float]] = []

    class FakeMember:
        def __init__(self, slot_id: str) -> None:
            self._slot_id = slot_id

        def pump_once(self, *, timeout_s: float) -> None:
            calls.append((self._slot_id, timeout_s))

    population = object.__new__(PopulationAsyncSAC)
    population._state = RuntimeState.RUNNING
    population._slot_ids = ("member-00", "member-01", "member-02")
    population._members = {
        slot_id: FakeMember(slot_id) for slot_id in population._slot_ids
    }
    population._next_pump_index = 0

    population.pump_once()
    population.pump_once()

    assert calls == [
        ("member-00", 0.0),
        ("member-01", 0.0),
        ("member-02", 0.0),
        ("member-01", 0.0),
        ("member-02", 0.0),
        ("member-00", 0.0),
    ]


def test_partial_population_setup_stops_created_members_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        population_module,
        "_validate_population_members",
        lambda members, *, validate_spaces, validate_structure: tuple(members),
    )
    replay = object()

    class FakeReplayActor:
        @classmethod
        def options(cls, **_: object):
            return cls

        @classmethod
        def remote(cls, *_: object, **__: object) -> object:
            return replay

    created: list[FakeMember] = []

    class FakeMember:
        def __init__(self, *_: object, **__: object) -> None:
            if created:
                raise RuntimeError("second member failed")
            self.stopped = False
            created.append(self)

        def start(self) -> None:
            raise AssertionError("members must all be created before start")

        def stop(self, *, graceful: bool) -> None:
            assert not graceful
            self.stopped = True

    killed: list[object] = []
    monkeypatch.setattr(population_module, "ReplayActor", FakeReplayActor)
    monkeypatch.setattr(population_module, "SingleMemberAsyncSAC", FakeMember)
    monkeypatch.setattr(
        population_module.ray,
        "kill",
        lambda actor, *, no_restart: killed.append(actor),
    )
    population = PopulationAsyncSAC(
        make_single_trial_specs(),
        run_id="run-test",
    )

    with pytest.raises(RuntimeError, match="second member failed"):
        population.start()

    assert created[0].stopped
    assert killed == [replay]
    assert population.state is RuntimeState.FAILED
