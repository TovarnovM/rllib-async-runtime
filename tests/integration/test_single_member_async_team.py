from __future__ import annotations

import copy
import json
import time

import pytest
import ray
import torch
from ray.exceptions import GetTimeoutError, RayActorError

from rllib_async.examples.temporal_hierarchy import build_temporal_hierarchy_configs
from rllib_async.replay.sequences import SequenceConfig
from rllib_async.runtime import (
    AsyncRuntimeConfig,
    RuntimeState,
    SingleMemberAsyncTeam,
    read_runtime_checkpoint,
)
from tests.helpers import (
    assert_learner_parameter_devices,
    assert_tree_close,
    flattened_key_paths,
)


def _settings(*, num_gpus: int = 0) -> AsyncRuntimeConfig:
    return AsyncRuntimeConfig(
        runner_count=1,
        max_episode_steps=16,
        replay_capacity_transitions=512,
        replay_capacity_bytes=4_000_000,
        pending_commit_high_watermark=2,
        pending_commit_low_watermark=0,
        batch_size=2,
        batch_queue_capacity=0,
        learner_updates_per_tick=3,
        publication_interval_updates=1,
        evaluation_interval_env_steps=8,
        evaluation_num_episodes=1,
        report_interval_s=0.1,
        shutdown_timeout_s=15,
        num_cpus_per_replay=1,
        num_cpus_per_runner=0,
        num_cpus_per_evaluation_runner=0,
        num_cpus_per_learner=0,
        num_gpus_per_learner=num_gpus,
        validate_startup=True,
    )


def _counters(runtime):
    stats = ray.get(runtime._learner_actor.get_stats.remote(), timeout=10)
    return {module_id: dict(values) for module_id, values in stats.module_counters}


def _train_until(runtime, minimum_updates, *, minimum_env_steps=0):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        report = runtime.run_for(0.1)
        counters = _counters(runtime)
        if (
            all(
                counters[module_id]["optimizer_updates"] >= count
                for module_id, count in minimum_updates.items()
            )
            and report["evaluation"]["rounds_completed"] >= 1
            and report["rollout"]["env_steps"] >= minimum_env_steps
        ):
            return report
    pytest.fail(f"Mixed team did not make progress: {report!r}, {counters!r}")


def _owned_actors(runtime):
    return [
        runtime._replay_actor,
        runtime._learner_actor,
        *runtime._rollout_group._runners.values(),
        *runtime._evaluation_group._actors.values(),
    ]


def _assert_stopped(actors):
    for actor in actors:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                ray.get(actor.__ray_ready__.remote(), timeout=1)
            except RayActorError:
                break
            except GetTimeoutError:
                continue
        else:
            pytest.fail(f"Actor remained alive after mixed-team stop: {actor}")


def _inspect_devices(runtime, device_type):
    def inspect(actor):
        host = actor._host
        devices = {}
        for module_id, adapter in host._adapters.items():
            # A local LearnerGroup executes in this host, taking no second Ray GPU.
            assert adapter._learner_group._learner is not None
            devices[module_id] = assert_learner_parameter_devices(
                adapter._learner_group, device_type
            )
            assert set(devices[module_id]) == {module_id}
        return devices, ray.get_runtime_context().get_accelerator_ids().get("GPU", [])

    return ray.get(runtime._learner_actor.__ray_call__.remote(inspect), timeout=10)


def _assert_bundle(weights, module_ids):
    assert set(weights.module_versions) == set(weights.state) == set(module_ids)
    assert len(set(weights.module_versions.values())) == 1
    paths = flattened_key_paths(weights.state)
    assert not any("target_" in path or "optimizer" in path for path in paths)
    assert not any(
        "qf" in path for path in flattened_key_paths(weights.state["worker_1"])
    )


def _frozen_learner_states(runtime):
    return ray.get(
        runtime._learner_actor.__ray_call__.remote(
            lambda actor: {
                module_id: actor._host._adapters[module_id].get_state()[
                    "learner_group"
                ]
                for module_id in ("worker_0", "worker_1")
            }
        ),
        timeout=10,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "num_gpus",
    [pytest.param(0, id="cpu"), pytest.param(1, marks=pytest.mark.gpu, id="gpu")],
)
def test_mixed_recurrent_runtime_publishes_evaluates_and_resumes(
    ray_runtime, tmp_path, request, num_gpus
) -> None:
    if num_gpus and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this container")
    rollout, modules, _ = build_temporal_hierarchy_configs(
        episode_length=8,
        manager_period=3,
        variable_duration=True,
        vector_worker=True,
        recurrent=True,
    )
    for config in modules.values():
        config.learners(num_gpus_per_learner=num_gpus)
        config.training(
            n_step=2,
            train_batch_size_per_learner=2,
            num_steps_sampled_before_learning_starts=0,
            training_intensity=1.0,
            target_network_update_freq=1,
        )
    sequences = {module_id: SequenceConfig(learning_length=2) for module_id in modules}
    settings = _settings(num_gpus=num_gpus)
    assert ray.available_resources().get("GPU", 0) >= num_gpus
    device_type = "cuda" if num_gpus else "cpu"
    runtime = SingleMemberAsyncTeam(
        rollout, modules, settings, sequence_configs=sequences
    )
    restored = None
    actors = _owned_actors(runtime)
    try:
        runtime.start()
        report = _train_until(runtime, dict.fromkeys(modules, 1))
        assert report["controller"]["algorithm"] == "team"
        assert report["learner"]["weight_publications"] > 0
        assert report["authoritative_replay"]["total_transitions"] > 0
        assert set(report["fast_replay"]["active_module_transition_counts"]) == set(
            modules
        )
        assert report["batching"]["queue_capacity"] == 0

        runtime.pause(timeout_s=10)
        checkpoint_dir = tmp_path / "mixed-team"
        checkpoint_dir.mkdir()
        checkpoint = runtime.save_checkpoint(checkpoint_dir, timeout_s=10)
        saved = read_runtime_checkpoint(checkpoint_dir)
        assert saved.controller["algorithm"] == "team"
        source_counters = _counters(runtime)
        source_stats = ray.get(runtime._learner_actor.get_stats.remote(), timeout=10)
        source_weights = ray.get(
            runtime._learner_actor.get_published_weights.remote(), timeout=10
        )
        _assert_bundle(source_weights, modules)
        devices, assigned_gpus = _inspect_devices(runtime, device_type)
        assert len(assigned_gpus) == len(source_stats.accelerator_ids) == num_gpus
        request.node.user_properties.append(
            ("source_parameter_devices", json.dumps(devices))
        )
        assert (
            source_stats.fast_replay.cursor
            == checkpoint.replay_cursor
            == saved.replay_cursor
        )
        assert source_stats.fast_replay.active_cursor == saved.replay_cursor
        assert all(
            values["learned_positions"] > 0 for values in source_counters.values()
        )
        runtime.stop(graceful=False)
        _assert_stopped(actors)

        restored = SingleMemberAsyncTeam.from_checkpoint(
            rollout, modules, settings, checkpoint_dir, sequence_configs=sequences
        )
        actors.extend(_owned_actors(restored))
        assert _counters(restored) == source_counters
        restored_stats = ray.get(restored._learner_actor.get_stats.remote(), timeout=10)
        assert restored_stats.learner_updates == source_stats.learner_updates
        assert (
            restored_stats.latest_module_version == source_stats.latest_module_version
        )
        restored_weights = ray.get(
            restored._learner_actor.get_published_weights.remote(), timeout=10
        )
        _assert_bundle(restored_weights, modules)
        assert_tree_close(source_weights.state, restored_weights.state)
        assert restored_stats.fast_replay.cursor == saved.replay_cursor
        devices, assigned_gpus = _inspect_devices(restored, device_type)
        assert len(assigned_gpus) == num_gpus
        request.node.user_properties.append(
            ("restored_parameter_devices", json.dumps(devices))
        )
        assert dict(restored._rollout_group.get_stats().runner_generations) == {
            runner_id: generation + 1
            for runner_id, generation in saved.rollout["runner_generations"].items()
        }

        restored.start()
        continued = _train_until(
            restored,
            {
                module_id: values["optimizer_updates"] + 1
                for module_id, values in source_counters.items()
            },
            minimum_env_steps=saved.rollout["env_steps"] + 1,
        )
        assert continued["controller"]["restore_count"] == 1
        assert (
            continued["learner"]["latest_module_version"]
            > source_stats.latest_module_version
        )
        restored.drain(timeout_s=10)
        final_weights = ray.get(
            restored._learner_actor.get_published_weights.remote(), timeout=10
        )
        _assert_bundle(final_weights, modules)
        final_counters = _counters(restored)
        assert all(
            final_counters[module_id]["ingested_decisions"]
            > source_counters[module_id]["ingested_decisions"]
            for module_id in modules
        )
    finally:
        try:
            if runtime.state is not RuntimeState.STOPPED:
                runtime.stop(graceful=False)
        finally:
            if restored is not None:
                restored.stop(graceful=False)
            _assert_stopped(actors)


@pytest.mark.integration
def test_frozen_workers_keep_weights_while_manager_learns(ray_runtime) -> None:
    rollout, modules, _ = build_temporal_hierarchy_configs(
        episode_length=6, manager_period=2, variable_duration=False
    )
    modules["manager"].training(
        train_batch_size_per_learner=2,
        num_steps_sampled_before_learning_starts=0,
        training_intensity=1.0,
    )
    runtime = SingleMemberAsyncTeam(
        rollout, modules, _settings(), trainable_modules=("manager",)
    )
    actors = _owned_actors(runtime)
    try:
        initial = copy.deepcopy(
            ray.get(
                runtime._learner_actor.get_published_weights.remote(), timeout=10
            ).state
        )
        frozen_states = _frozen_learner_states(runtime)
        runtime.start()
        _train_until(runtime, {"manager": 2})
        runtime.drain(timeout_s=10)
        counters = _counters(runtime)
        final = ray.get(
            runtime._learner_actor.get_published_weights.remote(), timeout=10
        )
        _assert_bundle(final, modules)
        assert counters["manager"]["optimizer_updates"] >= 2
        for module_id in ("worker_0", "worker_1"):
            assert counters[module_id]["optimizer_updates"] == 0
            assert counters[module_id]["learned_positions"] == 0
            assert_tree_close(initial[module_id], final.state[module_id])
        assert_tree_close(frozen_states, _frozen_learner_states(runtime))
    finally:
        runtime.stop(graceful=False)
        _assert_stopped(actors)
