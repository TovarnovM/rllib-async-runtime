from __future__ import annotations

import struct
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import ray
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.exceptions import RayActorError
from ray.rllib.algorithms.sac import SACConfig
from ray.tune import Stopper
from tensorboardX.proto.event_pb2 import Event

from rllib_async.examples import SyntheticThroughputEnv
from rllib_async.runtime import (
    PBT_STATE_FILENAME,
    AsyncSACRuntimeConfig,
    FloatMutation,
    PopulationAsyncSAC,
    PopulationLauncher,
    PopulationMemberSpec,
    PopulationTrainable,
    SimplePBTConfig,
    SingleMemberAsyncSAC,
    read_population_checkpoint_bundle,
)


def make_population_specs(
    *,
    num_gpus_per_learner: int = 0,
) -> tuple[PopulationMemberSpec, PopulationMemberSpec]:
    members: list[PopulationMemberSpec] = []
    for index in range(2):
        seed = 20260726 + index
        sac_config = (
            SACConfig()
            .environment(
                SyntheticThroughputEnv,
                env_config={"episode_length": 4},
            )
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .env_runners(
                num_env_runners=0,
                create_local_env_runner=True,
                num_envs_per_env_runner=1,
                batch_mode="complete_episodes",
                episodes_to_numpy=True,
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=num_gpus_per_learner,
            )
            .training(
                num_steps_sampled_before_learning_starts=8,
                policy_model_config={"fcnet_hiddens": [16]},
                q_model_config={"fcnet_hiddens": [16]},
                target_network_update_freq=1,
                train_batch_size_per_learner=8,
                twin_q=True,
            )
            .debugging(seed=seed)
        )
        runtime_config = AsyncSACRuntimeConfig(
            member_id=f"member-{index}",
            runner_count=4,
            max_episode_steps=4,
            replay_capacity_transitions=2_048,
            replay_capacity_bytes=20_000_000,
            replay_journal_capacity=512,
            replay_sync_max_bytes=1_000_000,
            pending_commit_high_watermark=8,
            pending_commit_low_watermark=3,
            batch_size=8,
            batch_queue_capacity=2,
            learner_updates_per_tick=2,
            publication_interval_updates=1,
            evaluation_interval_env_steps=0,
            evaluation_num_episodes=0,
            report_interval_s=0.1,
            event_poll_timeout_s=0.005,
            shutdown_timeout_s=20,
            seed=seed,
            num_cpus_per_replay=0,
            num_cpus_per_runner=0,
            num_cpus_per_evaluation_runner=0,
            num_cpus_per_learner=0,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        members.append(PopulationMemberSpec(sac_config, runtime_config))
    return members[0], members[1]


def population_actor_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def make_single_trial_population_specs() -> tuple[PopulationMemberSpec, ...]:
    return tuple(
        PopulationMemberSpec(
            member.sac_config,
            replace(
                member.runtime_config,
                member_id=f"member-{index:02d}",
            ),
        )
        for index, member in enumerate(make_population_specs())
    )


def make_test_pbt_config() -> SimplePBTConfig:
    return SimplePBTConfig(
        perturbation_interval_reports=10_000,
        reward_window_episodes=8,
        min_episodes_after_restart=1,
        seed=20260729,
        mutations={
            "actor_lr": FloatMutation(
                low=1e-5,
                high=1e-3,
                factors=(0.8,),
            )
        },
    )


class PopulationReadyStopper(Stopper):
    """Wait for both population data and learner progress, with a hard bound."""

    def __init__(self, *, max_iterations: int = 100) -> None:
        self._max_iterations = max_iterations

    def __call__(self, trial_id: str, result: dict) -> bool:
        del trial_id
        producers = set(
            result.get("fast_replay", {}).get(
                "active_producer_episode_counts",
                {},
            )
        )
        learner_updates = result.get("learner", {}).get("learner_updates", 0)
        ready = learner_updates >= 1 and {"member-0", "member-1"}.issubset(producers)
        return ready or result.get("training_iteration", 0) >= self._max_iterations

    def stop_all(self) -> bool:
        return False


class SingleTrialPopulationReadyStopper(Stopper):
    def __init__(self, *, max_iterations: int = 50) -> None:
        self._max_iterations = max_iterations

    def __call__(self, trial_id: str, result: dict) -> bool:
        del trial_id
        members = result.get("members", {})
        ready = len(members) >= 2 and all(
            member.get("train", {}).get("episodes_in_window", 0) >= 1
            and member.get("learner", {}).get("learner_updates", 0) >= 1
            for member in members.values()
        )
        return ready or result.get("training_iteration", 0) >= self._max_iterations

    def stop_all(self) -> bool:
        return False


def read_tensorboard_tags(path: Path) -> set[str]:
    """Read TensorBoard's TFRecord framing without adding TensorFlow."""

    tags: set[str] = set()
    with path.open("rb") as stream:
        while length_bytes := stream.read(8):
            if len(length_bytes) != 8:
                raise AssertionError("truncated TensorBoard event length")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise AssertionError("truncated TensorBoard length checksum")
            payload = stream.read(length)
            if len(payload) != length:
                raise AssertionError("truncated TensorBoard event")
            if len(stream.read(4)) != 4:
                raise AssertionError("truncated TensorBoard event checksum")
            event = Event()
            event.ParseFromString(payload)
            if event.HasField("summary"):
                tags.update(value.tag for value in event.summary.value)
    return tags


def test_population_ready_stopper_requires_data_and_learner_progress() -> None:
    stopper = PopulationReadyStopper(max_iterations=3)
    one_producer = {
        "training_iteration": 1,
        "learner": {"learner_updates": 1},
        "fast_replay": {
            "active_producer_episode_counts": {"member-0": 1},
        },
    }
    no_updates = {
        "training_iteration": 2,
        "learner": {"learner_updates": 0},
        "fast_replay": {
            "active_producer_episode_counts": {
                "member-0": 1,
                "member-1": 1,
            },
        },
    }
    ready = {
        **no_updates,
        "learner": {"learner_updates": 1},
    }

    assert not stopper("member-0", one_producer)
    assert not stopper("member-0", no_updates)
    assert stopper("member-0", ready)
    assert stopper("member-0", {"training_iteration": 3})
    assert not stopper.stop_all()


@pytest.mark.integration
def test_single_trial_population_runs_all_members_on_one_replay(
    ray_runtime: None,
) -> None:
    population = PopulationAsyncSAC(
        make_single_trial_population_specs(),
        run_id="run-integration",
        report_interval_s=0.1,
    )
    replay = None
    actor_probes: list[tuple[object, str]] = []
    try:
        population.start()
        replay = population.replay_actor
        members = population.members
        assert set(members) == {"member-00", "member-01"}
        assert all(member.state.value == "running" for member in members.values())
        assert (
            len({member._learner_actor._actor_id for member in members.values()}) == 2
        )
        assert all(
            member._replay_actor._actor_id == replay._actor_id
            for member in members.values()
        )
        actor_probes = [
            (replay, "get_stats"),
            *((member._learner_actor, "get_stats") for member in members.values()),
            *(
                (runner, "close")
                for member in members.values()
                for runner in member._rollout_group._runners.values()
            ),
        ]

        expected_runtime_ids = set(population.runtime_member_ids.values())
        report = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            report = population.run_for_report_interval(0.1)
            member_reports = report["members"].values()
            if all(
                member["learner"]["learner_updates"] >= 1
                and member["rollout"]["env_steps"] >= 1
                and member["train"]["episodes_in_window"] >= 1
                and expected_runtime_ids.issubset(
                    member["fast_replay"]["active_producer_episode_counts"]
                )
                for member in member_reports
            ):
                break

        assert report is not None
        assert report["population"]["size"] == 2
        assert report["population"]["eligible_members"] == 2
        assert report["replay"]["transitions"] >= 1
        assert all(
            member["learner"]["learner_updates"] >= 1
            for member in report["members"].values()
        )
        assert all(
            member["train"]["episodes_since_metric_reset"] >= 1
            for member in report["members"].values()
        )
        active_intervals = [
            (
                member["controller"]["started_at_monotonic"],
                member["controller"]["reported_at_monotonic"],
            )
            for member in report["members"].values()
        ]
        assert max(start for start, _ in active_intervals) < min(
            end for _, end in active_intervals
        )
    finally:
        population.stop(graceful=False)

    assert population.state.value == "stopped"
    for actor, method_name in actor_probes:
        deadline = time.monotonic() + 5
        while True:
            try:
                ray.get(getattr(actor, method_name).remote(), timeout=1)
            except RayActorError:
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
            if time.monotonic() >= deadline:
                pytest.fail(f"{method_name} actor remained alive after stop")
            time.sleep(0.05)


@pytest.mark.integration
def test_pbt_restarts_only_target_and_resumes_learning_on_shared_replay(
    ray_runtime: None,
) -> None:
    population = PopulationAsyncSAC(
        make_single_trial_population_specs(),
        run_id="run-pbt-integration",
        report_interval_s=0.1,
        pbt_config=make_test_pbt_config(),
    )
    try:
        population.start()
        report = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            report = population.run_for_report_interval(0.1)
            if all(
                member["learner"]["learner_updates"] >= 1
                and member["train"]["episodes_since_metric_reset"] >= 1
                for member in report["members"].values()
            ):
                break
        assert report is not None
        assert all(
            member["learner"]["learner_updates"] >= 1
            and member["train"]["episodes_since_metric_reset"] >= 1
            for member in report["members"].values()
        )

        members_before = population.members
        donor_before = members_before["member-01"]
        target_before = members_before["member-00"]
        donor_learner_id = donor_before._learner_actor._actor_id
        target_learner_id = target_before._learner_actor._actor_id
        donor_updates_before = report["members"]["member-01"]["learner"][
            "learner_updates"
        ]
        replay_id = population.replay_actor._actor_id
        replay_before = ray.get(population.replay_actor.get_stats.remote())
        runtime_ids_before = population.runtime_member_ids
        donor_hparams = dict(population._current_hparams["member-01"])

        population._reports_since_perturbation = 10_000
        event = population._maybe_run_pbt_step(
            {
                "member-00": {
                    "train": {
                        "episode_reward_mean": 1.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
                "member-01": {
                    "train": {
                        "episode_reward_mean": 3.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
            }
        )

        assert event["event_happened"] == 1
        assert event["donor_slot"] == "member-01"
        assert event["target_slot"] == "member-00"
        assert event["mutated_parameter"] == "actor_lr"
        assert population._generations == {"member-00": 1, "member-01": 0}
        assert donor_before is population.members["member-01"]
        assert donor_before._learner_actor._actor_id == donor_learner_id
        target_after = population.members["member-00"]
        assert target_after is not target_before
        assert target_after._learner_actor._actor_id != target_learner_id
        assert target_before.state.value == "stopped"
        assert target_after._replay_actor._actor_id == replay_id
        assert (
            population.runtime_member_ids["member-01"]
            == (runtime_ids_before["member-01"])
        )
        assert (
            population.runtime_member_ids["member-00"]
            != (runtime_ids_before["member-00"])
        )
        assert population._current_hparams["member-00"] == {
            **donor_hparams,
            "actor_lr": pytest.approx(donor_hparams["actor_lr"] * 0.8),
        }
        imported_weights = ray.get(
            target_after._learner_actor.get_published_weights.remote()
        )
        assert imported_weights.member_id == population.runtime_member_ids["member-00"]
        assert imported_weights.learner_updates == 0
        assert set(imported_weights.module_versions.values()) == {1}

        fresh_target_report = target_after.get_report()
        assert fresh_target_report["train"]["episodes_since_metric_reset"] == 0
        assert fresh_target_report["learner"]["learner_updates"] == 0
        assert (
            fresh_target_report["controller"]["budget_sampled_origin"]
            == target_after._sac_config.num_steps_sampled_before_learning_starts
        )
        assert "member-00" not in population._eligible_member_scores(
            {
                "member-00": fresh_target_report,
                "member-01": {
                    "train": {
                        "episode_reward_mean": 3.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
            }
        )

        resumed_report = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            resumed_report = population.run_for_report_interval(0.1)
            target_metrics = resumed_report["members"]["member-00"]
            if (
                target_metrics["rollout"]["env_steps"] >= 1
                and target_metrics["learner"]["learner_updates"] >= 1
            ):
                break
        assert resumed_report is not None
        target_metrics = resumed_report["members"]["member-00"]
        assert target_metrics["rollout"]["env_steps"] >= 1
        assert target_metrics["learner"]["learner_updates"] >= 1
        assert (
            resumed_report["members"]["member-01"]["learner"]["learner_updates"]
            > donor_updates_before
        )
        replay_after = ray.get(population.replay_actor.get_stats.remote())
        assert replay_after.cursor.mutation_seq > replay_before.cursor.mutation_seq
    finally:
        population.stop(graceful=False)


@pytest.mark.integration
def test_population_checkpoint_exact_resume_and_warm_start(
    ray_runtime: None,
    tmp_path: Path,
) -> None:
    specs = make_single_trial_population_specs()
    pbt_config = make_test_pbt_config()
    source = PopulationAsyncSAC(
        specs,
        run_id="run-stage3-source",
        report_interval_s=0.1,
        pbt_config=pbt_config,
    )
    checkpoint_dir = tmp_path / "stage3-checkpoint"
    checkpoint_dir.mkdir()
    source_runtime_ids: dict[str, str] = {}
    source_hparams: dict[str, dict[str, float]] = {}
    source_generations: dict[str, int] = {}
    source_report_index = 0
    source_exploit_count = 0
    replay_cursor = None
    try:
        source.start()
        deadline = time.monotonic() + 30
        report = None
        while time.monotonic() < deadline:
            report = source.run_for_report_interval(0.1)
            if all(
                member["learner"]["learner_updates"] >= 1
                and member["train"]["episodes_since_metric_reset"] >= 1
                for member in report["members"].values()
            ):
                break
        assert report is not None
        assert all(
            member["learner"]["learner_updates"] >= 1
            for member in report["members"].values()
        )

        source._reports_since_perturbation = 10_000
        event = source._maybe_run_pbt_step(
            {
                "member-00": {
                    "train": {
                        "episode_reward_mean": 1.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
                "member-01": {
                    "train": {
                        "episode_reward_mean": 3.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
            }
        )
        assert event["event_happened"] == 1

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            report = source.run_for_report_interval(0.1)
            if all(
                member["learner"]["learner_updates"] >= 1
                and member["train"]["episodes_since_metric_reset"] >= 1
                for member in report["members"].values()
            ):
                break
        assert all(
            member["learner"]["learner_updates"] >= 1
            for member in report["members"].values()
        )

        source_runtime_ids = source.runtime_member_ids
        source_hparams = {
            slot_id: dict(values) for slot_id, values in source._current_hparams.items()
        }
        source_generations = dict(source._generations)
        source_report_index = source._report_index
        source_exploit_count = source._exploit_count
        replay_cursor = ray.get(source.replay_actor.get_stats.remote()).cursor

        source.save_checkpoint(checkpoint_dir)
        assert all(
            member.state.value == "running" for member in source.members.values()
        )
    finally:
        source.stop(graceful=False)

    _, replay_state, checkpoint_members = read_population_checkpoint_bundle(
        checkpoint_dir
    )
    assert replay_cursor is not None
    assert replay_state.mutation_seq >= replay_cursor.mutation_seq
    checkpoint_updates = {
        slot_id: checkpoint_members[runtime_id].controller["learner_updates_completed"]
        for slot_id, runtime_id in source_runtime_ids.items()
    }

    resumed = PopulationAsyncSAC.from_checkpoint(
        specs,
        checkpoint_dir,
        pbt_config=pbt_config,
    )
    assert resumed.run_id == "run-stage3-source"
    assert resumed.runtime_member_ids == source_runtime_ids
    assert resumed._generations == source_generations
    assert resumed._current_hparams == source_hparams
    assert resumed._report_index == source_report_index
    assert resumed._exploit_count == source_exploit_count
    try:
        resumed.start()
        restored_replay = ray.get(resumed.replay_actor.get_stats.remote())
        assert restored_replay.cursor.mutation_seq == replay_state.mutation_seq
        immediate = {
            slot_id: member.get_report(include_authoritative_replay=False)
            for slot_id, member in resumed.members.items()
        }
        assert all(
            report["controller"]["restore_count"] == 1
            and report["learner"]["learner_updates"] >= checkpoint_updates[slot_id]
            and report["train"]["episodes_since_metric_reset"]
            == report["episodes_this_iter"]
            for slot_id, report in immediate.items()
        )

        deadline = time.monotonic() + 30
        resumed_report = None
        while time.monotonic() < deadline:
            resumed_report = resumed.run_for_report_interval(0.1)
            if all(
                member["train"]["episodes_since_metric_reset"] >= 1
                and member["learner"]["learner_updates"] > checkpoint_updates[slot_id]
                for slot_id, member in resumed_report["members"].items()
            ):
                break
        assert resumed_report is not None
        assert all(
            member["train"]["episodes_since_metric_reset"] >= 1
            for member in resumed_report["members"].values()
        )
    finally:
        resumed.stop(graceful=False)

    warm_specs = tuple(
        PopulationMemberSpec(
            member.sac_config.copy(copy_frozen=False).training(
                actor_lr=2e-4,
                critic_lr=2.5e-4,
                alpha_lr=8e-5,
            ),
            member.runtime_config,
        )
        for member in specs
    )
    warm_pbt_config = SimplePBTConfig(
        perturbation_interval_reports=10_000,
        reward_window_episodes=4,
        min_episodes_after_restart=1,
        seed=7,
        mutations={
            "actor_lr": FloatMutation(
                low=1e-4,
                high=4e-4,
                factors=(1.2,),
            )
        },
    )
    warm = PopulationAsyncSAC.from_warm_start_checkpoint(
        warm_specs,
        checkpoint_dir,
        run_id="run-stage3-warm",
        pbt_config=warm_pbt_config,
    )
    assert warm.run_id == "run-stage3-warm"
    assert warm._generations == {"member-00": 0, "member-01": 0}
    assert warm._exploit_count == 0
    assert warm._report_index == 0
    assert all(
        runtime_id.startswith("run-stage3-warm-")
        for runtime_id in warm.runtime_member_ids.values()
    )
    try:
        warm.start()
        warm_replay = ray.get(warm.replay_actor.get_stats.remote())
        assert warm_replay.cursor.mutation_seq == replay_state.mutation_seq
        deadline = time.monotonic() + 30
        warm_report = None
        while time.monotonic() < deadline:
            warm_report = warm.run_for_report_interval(0.1)
            if all(
                member["train"]["episodes_since_metric_reset"] >= 1
                and member["learner"]["learner_updates"] >= 1
                for member in warm_report["members"].values()
            ):
                break
        assert warm_report is not None
        assert all(
            member["controller"]["restore_count"] == 0
            and member["controller"]["budget_sampled_origin"] > 0
            and member["learner"]["learner_updates"] >= 1
            for member in warm_report["members"].values()
        )
        assert warm_report["members"]["member-00"]["hparams"] == {
            "actor_lr": 2e-4,
            "critic_lr": 2.5e-4,
            "alpha_lr": 8e-5,
        }

        warm._reports_since_perturbation = 10_000
        event = warm._maybe_run_pbt_step(
            {
                "member-00": {
                    "train": {
                        "episode_reward_mean": 1.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
                "member-01": {
                    "train": {
                        "episode_reward_mean": 3.0,
                        "episodes_since_metric_reset": 1,
                    }
                },
            }
        )
        assert event["event_happened"] == 1
        assert event["new_value"] == pytest.approx(2e-4 * 1.2)
        assert 1e-4 <= event["new_value"] <= 4e-4
    finally:
        warm.stop(graceful=False)


@pytest.mark.integration
def test_single_trial_population_writes_expected_tensorboard_tags(
    ray_runtime: None,
    tmp_path: Path,
) -> None:
    results = tune.Tuner(
        PopulationTrainable,
        param_space={
            "members": make_single_trial_population_specs(),
            "run_id": "run-tensorboard",
            "report_interval_s": 0.1,
        },
        run_config=RunConfig(
            name="single-trial-population-tensorboard",
            storage_path=str(tmp_path),
            stop=SingleTrialPopulationReadyStopper(),
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
            verbose=0,
        ),
    ).fit()

    result_list = list(results)
    assert len(result_list) == 1
    result = result_list[0]
    assert result.error is None
    assert result.checkpoint is not None
    assert result.metrics["population"]["size"] == 2
    with result.checkpoint.as_directory() as checkpoint_directory:
        checkpoint_path = Path(checkpoint_directory)
        assert (checkpoint_path / "population.snapshot").is_file()
        assert (checkpoint_path / "replay.snapshot").is_file()
        assert (checkpoint_path / PBT_STATE_FILENAME).is_file()
        assert len(tuple((checkpoint_path / "members").glob("*/member.snapshot"))) == 2
    event_files = tuple(Path(result.path).glob("events.out.tfevents.*"))
    assert event_files
    tags = set().union(*(read_tensorboard_tags(path) for path in event_files))
    assert {
        "ray/tune/population/report_index",
        "ray/tune/members/member-00/train/episode_reward_mean",
        "ray/tune/members/member-00/hparams/actor_lr",
        "ray/tune/members/member-01/train/episode_reward_mean",
        "ray/tune/replay/transitions",
    }.issubset(tags)


@pytest.mark.integration
def test_stopping_one_member_preserves_other_member_and_shared_replay(
    ray_runtime: None,
) -> None:
    specs = make_population_specs()
    launcher = PopulationLauncher(
        specs,
        replay_actor_name=population_actor_name("lifecycle-replay"),
    )
    member_zero = None
    member_one = None
    try:
        launcher.start()
        named_replay = ray.get_actor(
            launcher.descriptor.actor_name,
            namespace=launcher.descriptor.namespace,
        )
        assert named_replay._actor_id == launcher.replay_actor._actor_id

        member_zero = SingleMemberAsyncSAC(
            specs[0].sac_config,
            specs[0].runtime_config,
            replay_actor=launcher.replay_actor,
        )
        member_one = SingleMemberAsyncSAC(
            specs[1].sac_config,
            specs[1].runtime_config,
            replay_actor=launcher.replay_actor,
        )
        assert (
            member_zero._learner_actor._actor_id != member_one._learner_actor._actor_id
        )
        member_zero.start()
        member_one.start()

        reports = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            reports = (
                member_zero.run_for(0.1),
                member_one.run_for(0.1),
            )
            if all(
                report["learner"]["learner_updates"] >= 1
                and set(report["fast_replay"]["active_producer_episode_counts"])
                == {"member-0", "member-1"}
                for report in reports
            ):
                break
        assert reports is not None
        assert all(report["learner"]["learner_updates"] >= 1 for report in reports)
        assert all(
            set(report["fast_replay"]["active_producer_episode_counts"])
            == {"member-0", "member-1"}
            for report in reports
        )
        weights = ray.get(
            [
                member_zero._learner_actor.get_published_weights.remote(),
                member_one._learner_actor.get_published_weights.remote(),
            ]
        )
        assert [weights[0].member_id, weights[1].member_id] == [
            "member-0",
            "member-1",
        ]

        member_zero.stop(timeout_s=20)
        before = member_one.get_report()
        replay_before = launcher.get_replay_stats()
        deadline = time.monotonic() + 20
        after = before
        while time.monotonic() < deadline:
            after = member_one.run_for(0.1)
            if after["rollout"]["env_steps"] > before["rollout"]["env_steps"]:
                break
        assert after["rollout"]["env_steps"] > before["rollout"]["env_steps"]
        replay_after = launcher.get_replay_stats()
        assert replay_after.cursor.mutation_seq > replay_before.cursor.mutation_seq
    finally:
        if member_zero is not None and member_zero.state.value != "stopped":
            member_zero.stop(graceful=False)
        if member_one is not None and member_one.state.value != "stopped":
            member_one.stop(graceful=False)
        launcher.close()


@pytest.mark.integration
def test_tune_population_checkpoint_restores_shared_replay_once(
    ray_runtime: None,
    tmp_path,
) -> None:
    specs = make_population_specs()
    checkpoint_dir = tmp_path / "population-checkpoint"
    checkpoint_dir.mkdir()
    launcher = PopulationLauncher(
        specs,
        replay_actor_name=population_actor_name("tune-replay"),
    )
    try:
        results = launcher.fit(
            run_config=RunConfig(
                name="phase-8-population",
                storage_path=str(tmp_path / "ray-results"),
                stop=PopulationReadyStopper(),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
                verbose=0,
            )
        )
        result_list = list(results)
        assert all(
            result.metrics["learner"]["learner_updates"] >= 1 for result in result_list
        )
        assert all(
            set(result.metrics["fast_replay"]["active_producer_episode_counts"])
            == {"member-0", "member-1"}
            for result in result_list
        )
        assert all(
            result.metrics["controller"]["owns_replay_actor"] is False
            for result in result_list
        )
        for result in result_list:
            assert result.checkpoint is not None
            with result.checkpoint.as_directory() as checkpoint_directory:
                assert not (Path(checkpoint_directory) / "replay.snapshot").exists()
        active_intervals = [
            (
                result.metrics["controller"]["started_at_monotonic"],
                result.metrics["controller"]["reported_at_monotonic"],
            )
            for result in result_list
        ]
        assert max(start for start, _ in active_intervals) < min(
            end for _, end in active_intervals
        )

        checkpoint = launcher.save_checkpoint(checkpoint_dir, results=results)
        manifest, replay_state, member_states = read_population_checkpoint_bundle(
            checkpoint_dir
        )
        assert checkpoint.member_ids == ("member-0", "member-1")
        assert manifest.replay_cursor.mutation_seq > 0
        assert replay_state.mutation_seq == manifest.replay_cursor.mutation_seq
        assert set(member_states) == {"member-0", "member-1"}
    finally:
        launcher.close()

    restored = PopulationLauncher.from_checkpoint(
        specs,
        checkpoint_dir,
        replay_actor_name=population_actor_name("restored-replay"),
    )
    try:
        assert restored.get_replay_stats().cursor == manifest.replay_cursor
        resumed = restored.fit(
            run_config=RunConfig(
                name="phase-8-population-restored",
                storage_path=str(tmp_path / "ray-results-restored"),
                stop={"training_iteration": 1},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
                verbose=0,
            )
        )
        assert all(
            result.metrics["controller"]["restore_count"] == 1 for result in resumed
        )
    finally:
        restored.close()
