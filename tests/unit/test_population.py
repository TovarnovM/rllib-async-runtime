from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import pytest
import ray

import rllib_async.runtime.population as population_module
from rllib_async.protocols import ReplayCursor, ReplayStats
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    FloatMutation,
    PopulationAsyncSAC,
    PopulationError,
    PopulationLauncher,
    PopulationMemberSpec,
    PopulationTrainable,
    RuntimeState,
    SimplePBTConfig,
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


def make_pbt_config(**overrides: object) -> SimplePBTConfig:
    values = {
        "perturbation_interval_reports": 2,
        "min_episodes_after_restart": 2,
        "seed": 20260729,
        "mutations": {
            "actor_lr": FloatMutation(1e-5, 1e-3),
            "critic_lr": FloatMutation(1e-5, 1e-3),
            "alpha_lr": FloatMutation(1e-5, 1e-3),
        },
    }
    values.update(overrides)
    return SimplePBTConfig(**values)


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


def test_pbt_config_rejects_unknown_or_non_changing_mutations() -> None:
    with pytest.raises(ValueError, match="unknown PBT mutation"):
        SimplePBTConfig(
            perturbation_interval_reports=1,
            mutations={"gamma": FloatMutation(0.1, 0.9)},
        )
    with pytest.raises(ValueError, match="able to change"):
        FloatMutation(1e-5, 1e-3, factors=(1.0,))

    with pytest.raises(ValueError, match="unknown PBT mutation"):
        PopulationTrainable._parse_config(
            {
                "members": make_single_trial_specs(),
                "pbt": {
                    "perturbation_interval_reports": 1,
                    "mutations": {
                        "batch_size": {
                            "low": 1,
                            "high": 2,
                        }
                    },
                },
            }
        )


def test_pbt_mutation_is_reproducible_bounded_and_changes_one_hparam() -> None:
    config = make_pbt_config()
    hparams = {
        "actor_lr": 3e-4,
        "critic_lr": 4e-4,
        "alpha_lr": 1e-4,
    }

    first = PopulationAsyncSAC._mutate_hparams(
        hparams,
        config=config,
        exploit_count=3,
    )
    second = PopulationAsyncSAC._mutate_hparams(
        hparams,
        config=config,
        exploit_count=3,
    )

    assert first == second
    mutated, parameter, parameter_index, _, old_value, new_value = first
    assert parameter_index == tuple(sorted(config.mutations)).index(parameter)
    assert old_value != new_value
    assert {name for name in hparams if hparams[name] != mutated[name]} == {parameter}
    bounds = config.mutations[parameter]
    assert bounds.low <= new_value <= bounds.high


def test_pbt_selection_has_deterministic_tie_breaking() -> None:
    scores = {
        "member-02": 3.0,
        "member-01": 3.0,
        "member-00": 1.0,
    }

    assert PopulationAsyncSAC._select_donor_and_target(
        scores,
        mode="max",
    ) == ("member-01", "member-00")
    assert PopulationAsyncSAC._select_donor_and_target(
        scores,
        mode="min",
    ) == ("member-00", "member-01")


def test_pbt_waits_for_fresh_episodes_without_resetting_due_interval() -> None:
    class FakeMember:
        state = RuntimeState.RUNNING

    population = object.__new__(PopulationAsyncSAC)
    population._pbt_config = make_pbt_config(
        perturbation_interval_reports=1,
        min_episodes_after_restart=2,
    )
    population._slot_ids = ("member-00", "member-01")
    population._members = {slot_id: FakeMember() for slot_id in population._slot_ids}
    population._restarting_slot = None
    population._reports_since_perturbation = 1
    population._exploit_count = 0
    population._last_pbt_event = {}

    reports = {
        "member-00": {
            "train": {
                "episode_reward_mean": 3.0,
                "episodes_since_metric_reset": 2,
            }
        },
        "member-01": {
            "train": {
                "episode_reward_mean": 1.0,
                "episodes_since_metric_reset": 1,
            }
        },
    }

    event = population._maybe_run_pbt_step(reports)

    assert event["event_reason"] == "not_enough_eligible_members"
    assert population._reports_since_perturbation == 1
    assert population._exploit_count == 0


def test_pbt_step_restarts_only_deterministic_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMember:
        state = RuntimeState.RUNNING

        def __init__(self, slot_id: str) -> None:
            self.slot_id = slot_id
            self.exports = 0

        def export_pbt_state(self) -> object:
            self.exports += 1
            return object()

    population = object.__new__(PopulationAsyncSAC)
    population._pbt_config = make_pbt_config(perturbation_interval_reports=1)
    population._slot_ids = ("member-00", "member-01", "member-02")
    population._members = {
        slot_id: FakeMember(slot_id) for slot_id in population._slot_ids
    }
    population._restarting_slot = None
    population._reports_since_perturbation = 1
    population._exploit_count = 0
    population._last_exploit_duration_s = 0.0
    population._last_pbt_event = {}
    population._generations = dict.fromkeys(population._slot_ids, 0)
    population._current_hparams = {
        slot_id: {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 1e-4,
        }
        for slot_id in population._slot_ids
    }
    replaced: list[tuple[str, dict[str, float], object]] = []

    def replace_target(
        target_slot: str,
        *,
        new_hparams,
        donor_state: object,
    ) -> str:
        replaced.append((target_slot, dict(new_hparams), donor_state))
        return "run-test-member-00-g0001"

    monkeypatch.setattr(population, "_replace_target", replace_target)
    reports = {
        "member-00": {
            "train": {
                "episode_reward_mean": 1.0,
                "episodes_since_metric_reset": 2,
            }
        },
        "member-01": {
            "train": {
                "episode_reward_mean": 3.0,
                "episodes_since_metric_reset": 2,
            }
        },
        "member-02": {
            "train": {
                "episode_reward_mean": 2.0,
                "episodes_since_metric_reset": 2,
            }
        },
    }

    event = population._maybe_run_pbt_step(reports)

    assert event["event_happened"] == 1
    assert event["donor_slot"] == "member-01"
    assert event["target_slot"] == "member-00"
    assert len(replaced) == 1
    assert population._members["member-01"].exports == 1
    assert population._members["member-00"].exports == 0
    assert population._exploit_count == 1
    assert population._reports_since_perturbation == 0


def test_target_restart_uses_new_identity_shared_replay_and_fails_fast(
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
        pbt_config=make_pbt_config(),
    )
    replay = object()
    population._replay_actor = replay

    class OldMember:
        state = RuntimeState.RUNNING

        def __init__(self) -> None:
            self.stops = 0

        def stop(self, *, graceful: bool = True) -> None:
            assert graceful
            self.stops += 1

    class Replacement:
        def __init__(self) -> None:
            self.starts = 0

        def start(self) -> None:
            self.starts += 1

        def stop(self, *, graceful: bool = True) -> None:
            raise AssertionError(f"unexpected replacement stop {graceful=}")

    donor = OldMember()
    target = OldMember()
    population._members = {
        "member-00": target,
        "member-01": donor,
    }
    replacement = Replacement()
    captured: dict[str, object] = {}

    def from_pbt_state(
        cls,
        sac_config,
        runtime_config,
        pbt_state,
        *,
        replay_actor,
        reward_window_episodes,
    ):
        del cls
        captured.update(
            {
                "sac_config": sac_config,
                "runtime_config": runtime_config,
                "pbt_state": pbt_state,
                "replay_actor": replay_actor,
                "reward_window_episodes": reward_window_episodes,
            }
        )
        return replacement

    monkeypatch.setattr(
        SingleMemberAsyncSAC,
        "from_pbt_state",
        classmethod(from_pbt_state),
    )
    donor_state = object()
    new_hparams = {
        "actor_lr": 2.4e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 1e-4,
    }

    runtime_member_id = population._replace_target(
        "member-00",
        new_hparams=new_hparams,
        donor_state=donor_state,
    )

    assert runtime_member_id == "run-test-member-00-g0001"
    assert target.stops == 1
    assert donor.stops == 0
    assert replacement.starts == 1
    assert population._members["member-00"] is replacement
    assert population._members["member-01"] is donor
    assert captured["pbt_state"] is donor_state
    assert captured["replay_actor"] is replay
    assert captured["reward_window_episodes"] == 100
    runtime_config = captured["runtime_config"]
    assert runtime_config.member_id == runtime_member_id
    assert runtime_config.seed == 101
    sac_config = captured["sac_config"]
    assert sac_config.actor_lr == new_hparams["actor_lr"]
    assert population._generations["member-00"] == 1
    assert population._exploit_count_as_target["member-00"] == 1

    def fail_from_pbt_state(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("constructor failed")

    next_target = OldMember()
    population._members["member-01"] = next_target
    monkeypatch.setattr(
        SingleMemberAsyncSAC,
        "from_pbt_state",
        fail_from_pbt_state,
    )
    with pytest.raises(PopulationError, match="failed to restart"):
        population._replace_target(
            "member-01",
            new_hparams=new_hparams,
            donor_state=donor_state,
        )
    assert next_target.stops == 1


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
    population._reports_since_perturbation = 1

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

    population._pbt_config = make_pbt_config(mode="min")
    minimized = population._format_report(
        {
            "member-00": member_report(1.0),
            "member-01": member_report(3.0),
        },
        replay,
    )
    assert minimized["population"]["best_score"] == 1.0
    assert minimized["population"]["worst_score"] == 3.0


def test_population_report_preserves_scheduled_learning_rates(
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
    schedules = {
        "actor_lr": [[0, 3e-4], [100, 1e-4]],
        "critic_lr": [[0, 4e-4], [100, 2e-4]],
        "alpha_lr": [[0, 5e-4], [100, 3e-4]],
    }
    members = tuple(
        PopulationMemberSpec(
            make_sac_config().training(**schedules).debugging(seed=100 + index),
            spec.runtime_config,
        )
        for index, spec in enumerate(make_single_trial_specs())
    )
    population = PopulationAsyncSAC(members, run_id="run-test")

    report = {
        "train": {},
        "learner": {},
        "rollout": {},
    }
    replay = ReplayStats(
        cursor=ReplayCursor("shared", 0),
        episode_count=0,
        total_transitions=0,
        total_estimated_bytes=0,
        producer_episode_counts=(),
        producer_transition_counts=(),
        oldest_available_mutation_seq=1,
        journal_entries=0,
        deduplication_entries=0,
        commit_attempts=0,
        committed_episodes=0,
        duplicate_commits=0,
        rejected_commits=0,
        conflicting_commits=0,
        evicted_episodes=0,
    )

    result = population._format_report(
        {slot_id: report for slot_id in population.slot_ids}, replay
    )

    assert result["members"]["member-00"]["hparams"] == schedules


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
