from __future__ import annotations

import json
import time

import pytest
import ray
import torch
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE

from rllib_async.learner import SAC_TEMPERATURE_STATE
from rllib_async.runtime import (
    PopulationAsyncSAC,
    read_population_checkpoint_bundle,
)
from rllib_async.runtime.learner_host import decode_learner_host_checkpoint
from tests.helpers import assert_learner_parameter_devices, assert_tree_close
from tests.mixed_population_helpers import mixed_pbt, mixed_specs


def _progress(population, minimum=1):
    deadline = time.monotonic() + 40
    report = None
    while time.monotonic() < deadline:
        report = population.run_for_report_interval(0.1)
        if all(
            member["train"]["episodes_since_metric_reset"] >= 1
            and member["evaluation"]["rounds_completed"] >= 1
            and member["evaluation"]["latest_module_version"] > 1
            and all(
                dict(counters)["optimizer_updates"] >= minimum
                for _, counters in member["learner"]["module_counters"]
            )
            for member in report["members"].values()
        ):
            return report
    pytest.fail(f"Both mixed teams did not make progress: {report!r}")


def _inspect(member, device_type):
    def inspect(actor):
        host = actor._host
        devices = {}
        learners = {}
        for mid, adapter in host._adapters.items():
            assert adapter._learner_group._learner is not None
            devices[mid] = assert_learner_parameter_devices(
                adapter._learner_group, device_type
            )
            learners[mid] = adapter.get_state()["learner_group"]["learner"]
        return {
            "learners": learners,
            "counters": host._module_counters,
            "rng": {mid: rng.getstate() for mid, rng in host._samplers.items()},
            "priorities": {
                mid: sampler.get_state()
                for mid, sampler in host._prioritized_samplers.items()
            },
            "devices": devices,
            "gpu_ids": ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
        }

    return ray.get(member._learner_actor.__ray_call__.remote(inspect), timeout=15)


def _placement_evidence(population, request, prefix, num_gpus):
    snapshots = {
        slot: _inspect(member, "cuda" if num_gpus else "cpu")
        for slot, member in population.members.items()
    }
    devices = {slot: value["devices"] for slot, value in snapshots.items()}
    gpu_ids = {slot: value["gpu_ids"] for slot, value in snapshots.items()}
    assert all(len(ids) == num_gpus for ids in gpu_ids.values())
    if num_gpus:
        assert len({ids[0] for ids in gpu_ids.values()}) == 2
    request.node.user_properties.extend(
        [
            (f"{prefix}_parameter_devices", json.dumps(devices)),
            (f"{prefix}_accelerator_ids", json.dumps(gpu_ids)),
        ]
    )
    return snapshots


@pytest.mark.integration
@pytest.mark.parametrize(
    "num_gpus",
    [pytest.param(0, id="cpu"), pytest.param(1, marks=pytest.mark.gpu, id="two-gpu")],
)
def test_mixed_population_replacement_checkpoint_and_resume(
    ray_runtime, tmp_path, request, num_gpus,
) -> None:
    if num_gpus and torch.cuda.device_count() < 2:
        pytest.skip("This placement acceptance case requires two physical GPUs")
    assert ray.available_resources().get("GPU", 0) >= 2 * num_gpus
    specs = mixed_specs(num_gpus=num_gpus)
    population = PopulationAsyncSAC(specs, run_id="mixed", pbt_config=mixed_pbt())
    restored = None
    warm = None
    directory = tmp_path / "population"
    directory.mkdir()
    try:
        population.start()
        _progress(population)
        original = _placement_evidence(population, request, "original", num_gpus)
        assert all(
            any(
                optimizer["state"]["state"]
                for optimizer in learner[COMPONENT_OPTIMIZER].values()
            )
            for snapshot in original.values()
            for learner in snapshot["learners"].values()
        )
        replay_id = population.replay_actor._actor_id
        assert all(
            member._replay_actor._actor_id == replay_id
            for member in population.members.values()
        )
        donor = population.members["team-1"]
        old_target = population.members["team-0"]
        donor_models = _inspect(donor, "cuda" if num_gpus else "cpu")["learners"]
        population._reports_since_perturbation = 10_000
        event = population._maybe_run_pbt_step(
            {
                slot: {"train": {
                    "episode_reward_mean": score,
                    "episodes_since_metric_reset": 1,
                }}
                for slot, score in (("team-0", 1.0), ("team-1", 2.0))
            }
        )
        assert event["event_happened"] == 1
        assert event["mutated_parameter"] == "manager/lr"
        assert population._generations == {"team-0": 1, "team-1": 0}
        assert population.members["team-1"] is donor
        assert old_target.state.value == "stopped"
        target = population.members["team-0"]
        assert target._replay_actor._actor_id == replay_id
        received = _inspect(target, "cuda" if num_gpus else "cpu")
        for mid, learner in received["learners"].items():
            assert_tree_close(
                learner[COMPONENT_RL_MODULE], donor_models[mid][COMPONENT_RL_MODULE]
            )
            if SAC_TEMPERATURE_STATE in learner:
                assert_tree_close(
                    learner[SAC_TEMPERATURE_STATE],
                    donor_models[mid][SAC_TEMPERATURE_STATE],
                )
            assert all(
                not opt["state"]["state"]
                for opt in learner[COMPONENT_OPTIMIZER].values()
            )
            assert received["counters"][mid]["budget"] == 0
            assert received["counters"][mid]["optimizer_updates"] == 0
            priorities = received["priorities"][mid]
            assert priorities["max_priority"] == 1.0
        _progress(population)
        ids = population.runtime_member_ids
        hparams = dict(population._current_hparams)
        population.save_checkpoint(directory, timeout_s=30)
        population.stop(graceful=False)
        _, replay_state, member_states = read_population_checkpoint_bundle(directory)
        assert len({state.replay_file for state in member_states.values()}) == 1
        saved = {
            slot: decode_learner_host_checkpoint(member_states[mid].learner)
            for slot, mid in ids.items()
        }

        restored = PopulationAsyncSAC.from_checkpoint(
            specs, directory, pbt_config=mixed_pbt()
        )
        assert restored.runtime_member_ids == ids
        assert restored._current_hparams == hparams
        restored.start()
        assert (
            ray.get(restored.replay_actor.get_stats.remote()).cursor.mutation_seq
            == replay_state.mutation_seq
        )
        snapshots = _placement_evidence(restored, request, "restored", num_gpus)
        for slot, snapshot in snapshots.items():
            assert snapshot["counters"] == saved[slot]["module_counters"]
            assert snapshot["rng"] == saved[slot]["sampler_rng_states"]
            assert_tree_close(snapshot["priorities"], saved[slot]["priority_states"])
            for mid, learner in snapshot["learners"].items():
                assert_tree_close(
                    learner, saved[slot]["adapters"][mid]["learner_group"]["learner"]
                )
        _progress(restored)
        restored.stop(graceful=False)

        if not num_gpus:
            warm = PopulationAsyncSAC.from_warm_start_checkpoint(
                mixed_specs(seed=700, manager_lr=5e-4), directory,
                run_id="warm", pbt_config=mixed_pbt(),
            )
            warm.start()
            assert warm.runtime_member_ids != ids
            assert warm._generations == {"team-0": 0, "team-1": 0}
            assert warm._current_hparams["team-0"]["manager/lr"] == 5e-4
            for slot, member in warm.members.items():
                snapshot = _inspect(member, "cpu")
                for mid, learner in snapshot["learners"].items():
                    previous = saved[slot]["adapters"][mid]["learner_group"]["learner"]
                    assert_tree_close(
                        learner[COMPONENT_RL_MODULE], previous[COMPONENT_RL_MODULE]
                    )
                    assert all(
                        not opt["state"]["state"]
                        for opt in learner[COMPONENT_OPTIMIZER].values()
                    )
                    assert snapshot["counters"][mid]["budget"] == 0
            _progress(warm)
    finally:
        for runtime in (population, restored, warm):
            if runtime is not None:
                runtime.stop(graceful=False)
