from __future__ import annotations

import json
from dataclasses import asdict, replace

import gymnasium as gym
import pytest
import ray

import rllib_async.runtime.population as population_module
from rllib_async.protocols import ReplayCursor, ReplayStats
from rllib_async.replay.reference import EpisodeStore
from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    FloatMutation,
    InvalidPopulationCheckpointError,
    PopulationAsyncSAC,
    PopulationError,
    PopulationLauncher,
    PopulationMemberSpec,
    PopulationTrainable,
    RuntimeCheckpointState,
    RuntimeState,
    SimplePBTConfig,
    SingleMemberAsyncSAC,
    make_runtime_member_id,
)
from rllib_async.runtime.checkpoint import write_population_checkpoint
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


def write_single_trial_checkpoint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    object,
    tuple[PopulationMemberSpec, ...],
    SimplePBTConfig,
    PopulationAsyncSAC,
]:
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
    specs = make_single_trial_specs()
    pbt_config = make_pbt_config()
    population = PopulationAsyncSAC(
        specs,
        run_id="run-checkpoint",
        report_interval_s=0.1,
        pbt_config=pbt_config,
    )
    population._report_index = 7
    population._reports_since_perturbation = 1

    runtime = population._member_specs["member-00"].runtime_config
    store = EpisodeStore(
        population_module.FlatEpisodeCodec(),
        capacity_transitions=runtime.replay_capacity_transitions,
        capacity_bytes=runtime.replay_capacity_bytes,
        journal_capacity=runtime.replay_journal_capacity,
        store_generation="pbt-checkpoint",
    )
    states = {
        spec.runtime_config.member_id: RuntimeCheckpointState(
            state_version=1,
            member_id=spec.runtime_config.member_id,
            runtime_config=asdict(spec.runtime_config),
            replay_file="replay.snapshot",
            replay_cursor=store.cursor,
            learner=f"learner-{slot_id}".encode(),
            rollout={},
            evaluation=None,
            controller={},
        )
        for slot_id, spec in population._member_specs.items()
    }
    checkpoint_dir = tmp_path / "population"
    checkpoint_dir.mkdir()
    write_population_checkpoint(
        checkpoint_dir,
        replay_state=store.export_state(),
        members=states,
        pbt_metadata=population._checkpoint_metadata().to_mapping(),
    )
    return checkpoint_dir, specs, pbt_config, population


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


def test_population_run_modes_require_an_explicit_checkpoint_path() -> None:
    members = make_single_trial_specs()

    with pytest.raises(ValueError, match="run_mode"):
        PopulationTrainable._parse_config(
            {
                "members": members,
                "run_mode": "restore",
            }
        )
    with pytest.raises(ValueError, match="requires checkpoint_path"):
        PopulationTrainable._parse_config(
            {
                "members": members,
                "run_mode": "resume",
            }
        )
    with pytest.raises(ValueError, match="does not accept checkpoint_path"):
        PopulationTrainable._parse_config(
            {
                "members": members,
                "checkpoint_path": "/tmp/checkpoint",
            }
        )

    parsed = PopulationTrainable._parse_config(
        {
            "members": members,
            "run_mode": "warm_start",
            "checkpoint_path": "/tmp/checkpoint",
        }
    )
    assert parsed[-2:] == ("warm_start", "/tmp/checkpoint")


def test_exact_resume_restores_pbt_metadata_before_actor_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, specs, pbt_config, source = write_single_trial_checkpoint(
        tmp_path,
        monkeypatch,
    )

    restored = PopulationAsyncSAC.from_checkpoint(
        specs,
        checkpoint_dir,
        pbt_config=pbt_config,
    )

    assert restored.run_id == source.run_id
    assert restored.runtime_member_ids == source.runtime_member_ids
    assert restored._report_index == 7
    assert restored._reports_since_perturbation == 1
    assert restored._exploit_count == 0
    assert restored._current_hparams == source._current_hparams
    assert restored._checkpoint_replay_state is not None
    assert set(restored._checkpoint_member_states or {}) == {
        "member-00",
        "member-01",
    }

    incompatible = make_pbt_config(seed=pbt_config.seed + 1)
    with pytest.raises(ValueError, match="PBT config"):
        PopulationAsyncSAC.from_checkpoint(
            specs,
            checkpoint_dir,
            pbt_config=incompatible,
        )


def test_exact_resume_rejects_hparam_outside_checkpoint_bounds(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, specs, pbt_config, _ = write_single_trial_checkpoint(
        tmp_path,
        monkeypatch,
    )
    metadata_path = checkpoint_dir / "pbt_state.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["members"]["member-00"]["actor_lr"] = 2e-3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(InvalidPopulationCheckpointError, match="outside"):
        PopulationAsyncSAC.from_checkpoint(
            specs,
            checkpoint_dir,
            pbt_config=pbt_config,
        )


@pytest.mark.parametrize(
    ("changed_config", "message"),
    [
        (lambda config: config.debugging(seed=999), "SAC seed"),
        (lambda config: config.environment("Pendulum-v1"), "SAC env"),
        (
            lambda config: config.environment(env_config={"gravity": 3.0}),
            "SAC env_config",
        ),
    ],
)
def test_exact_resume_rejects_changed_rollout_configuration(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    changed_config,
    message: str,
) -> None:
    checkpoint_dir, specs, pbt_config, _ = write_single_trial_checkpoint(
        tmp_path,
        monkeypatch,
    )
    changed_specs = (
        PopulationMemberSpec(
            changed_config(specs[0].sac_config.copy(copy_frozen=False)),
            specs[0].runtime_config,
        ),
        specs[1],
    )

    with pytest.raises(ValueError, match=message):
        PopulationAsyncSAC.from_checkpoint(
            changed_specs,
            checkpoint_dir,
            pbt_config=pbt_config,
        )


def test_warm_start_resets_identity_and_rejects_ambiguous_hparams(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, specs, _, source = write_single_trial_checkpoint(
        tmp_path,
        monkeypatch,
    )
    warm_config = make_pbt_config(
        seed=1,
        mutations={"actor_lr": FloatMutation(1e-5, 1e-3)},
    )

    warm = PopulationAsyncSAC.from_warm_start_checkpoint(
        specs,
        checkpoint_dir,
        run_id="run-warm",
        pbt_config=warm_config,
    )

    assert warm.run_id == "run-warm"
    assert warm.run_id != source.run_id
    assert warm._generations == {"member-00": 0, "member-01": 0}
    assert warm._exploit_count == 0
    assert warm._report_index == 0
    assert warm._reports_since_perturbation == 0
    assert all(
        runtime_id.startswith("run-warm-")
        for runtime_id in warm.runtime_member_ids.values()
    )
    assert set(warm._warm_start_source_states or {}) == {
        "member-00",
        "member-01",
    }

    out_of_bounds = make_pbt_config(
        mutations={"actor_lr": FloatMutation(5e-4, 1e-3)},
    )
    with pytest.raises(ValueError, match="outside"):
        PopulationAsyncSAC.from_warm_start_checkpoint(
            specs,
            checkpoint_dir,
            run_id="run-invalid",
            pbt_config=out_of_bounds,
        )

    with pytest.raises(ValueError, match="slots"):
        PopulationAsyncSAC.from_warm_start_checkpoint(
            make_single_trial_specs(size=3),
            checkpoint_dir,
            run_id="run-resized",
            pbt_config=warm_config,
        )


def test_exact_resume_requires_pbt_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, specs, pbt_config, _ = write_single_trial_checkpoint(
        tmp_path,
        monkeypatch,
    )
    (checkpoint_dir / "pbt_state.json").unlink()

    with pytest.raises(InvalidPopulationCheckpointError, match="PBT checkpoint"):
        PopulationAsyncSAC.from_checkpoint(
            specs,
            checkpoint_dir,
            pbt_config=pbt_config,
        )


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


@pytest.mark.parametrize(
    ("bound", "factor"),
    ((1e-5, 0.8), (1e-3, 1.2)),
)
def test_pbt_one_sided_mutation_reflects_at_saturated_bound(
    bound: float,
    factor: float,
) -> None:
    config = make_pbt_config(
        mutations={
            name: FloatMutation(1e-5, 1e-3, factors=(factor,))
            for name in ("actor_lr", "critic_lr", "alpha_lr")
        }
    )
    hparams = {name: bound for name in config.mutations}

    mutated, parameter, _, applied_factor, old_value, new_value = (
        PopulationAsyncSAC._mutate_hparams(
            hparams,
            config=config,
            exploit_count=0,
        )
    )

    assert old_value == bound
    assert new_value != old_value
    assert 1e-5 <= new_value <= 1e-3
    assert new_value == pytest.approx(old_value * applied_factor)
    assert {name for name in hparams if hparams[name] != mutated[name]} == {parameter}


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


@pytest.mark.parametrize("failure_stage", ("member", "replay", "writer"))
def test_population_checkpoint_resumes_paused_members_after_failure(
    failure_stage: str,
    tmp_path,
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
        run_id="run-checkpoint-failure",
        pbt_config=make_pbt_config(),
    )
    population._state = RuntimeState.RUNNING
    runtime = population._member_specs["member-00"].runtime_config
    store = EpisodeStore(
        population_module.FlatEpisodeCodec(),
        capacity_transitions=runtime.replay_capacity_transitions,
        capacity_bytes=runtime.replay_capacity_bytes,
        journal_capacity=runtime.replay_journal_capacity,
        store_generation="checkpoint-failure",
    )
    calls: list[tuple[str, str]] = []

    class FakeMember:
        def __init__(self, slot_id: str) -> None:
            self.slot_id = slot_id
            self.state = RuntimeState.RUNNING

        def pause(self, *, timeout_s: float) -> None:
            assert timeout_s > 0
            calls.append(("pause", self.slot_id))
            self.state = RuntimeState.PAUSED

        def drain(self, *, timeout_s: float) -> None:
            assert timeout_s > 0
            calls.append(("drain", self.slot_id))

        def get_member_checkpoint_state(
            self,
            *,
            timeout_s: float,
        ) -> RuntimeCheckpointState:
            assert timeout_s > 0
            calls.append(("snapshot", self.slot_id))
            if failure_stage == "member" and self.slot_id == "member-01":
                raise RuntimeError("member snapshot failed")
            spec = population._member_specs[self.slot_id]
            return RuntimeCheckpointState(
                state_version=1,
                member_id=spec.runtime_config.member_id,
                runtime_config=asdict(spec.runtime_config),
                replay_file="replay.snapshot",
                replay_cursor=store.cursor,
                learner=b"learner",
                rollout={},
                evaluation=None,
                controller={},
            )

        def resume(self) -> None:
            calls.append(("resume", self.slot_id))
            self.state = RuntimeState.RUNNING

    replay_ref = object()

    class RemoteMethod:
        @staticmethod
        def remote() -> object:
            calls.append(("snapshot", "replay"))
            return replay_ref

    class FakeReplay:
        get_checkpoint_state = RemoteMethod()

    def fake_ray_get(ref: object, *, timeout: float):
        assert ref is replay_ref
        assert timeout > 0
        if failure_stage == "replay":
            raise RuntimeError("replay snapshot failed")
        return store.export_state()

    def fake_writer(*args, **kwargs):
        del args, kwargs
        calls.append(("write", "population"))
        if failure_stage == "writer":
            raise RuntimeError("writer failed")
        raise AssertionError("writer must only be reached in its failure case")

    population._members = {
        slot_id: FakeMember(slot_id) for slot_id in population.slot_ids
    }
    population._replay_actor = FakeReplay()
    monkeypatch.setattr(population_module.ray, "get", fake_ray_get)
    monkeypatch.setattr(
        population_module,
        "write_population_checkpoint",
        fake_writer,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(RuntimeError, match="failed"):
        population.save_checkpoint(checkpoint_dir)

    assert calls[:2] == [
        ("pause", "member-00"),
        ("pause", "member-01"),
    ]
    assert calls[2:4] == [
        ("drain", "member-00"),
        ("drain", "member-01"),
    ]
    assert calls[-2:] == [
        ("resume", "member-00"),
        ("resume", "member-01"),
    ]
    assert all(
        member.state is RuntimeState.RUNNING for member in population._members.values()
    )


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
