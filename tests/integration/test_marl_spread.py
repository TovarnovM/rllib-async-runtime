"""Small real MPE2 lifecycle gate; cluster scale stays outside ordinary CI."""

import gzip
import json
import time

import pytest
import ray
import torch

pytest.importorskip("mpe2")

from rllib_async.examples.marl_spread import build_spread_configs
from rllib_async.runtime import (
    AsyncRuntimeConfig,
    FloatMutation,
    PopulationAsyncSAC,
    PopulationMemberSpec,
    SimplePBTConfig,
)
from tests.helpers import assert_learner_parameter_devices


@pytest.mark.integration
def test_external_monitor_collects_proc_without_runtime_callbacks(ray_runtime):
    from benchmarks.marl_cluster import NodeMonitor

    monitor = NodeMonitor.remote()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            info = ray.get(monitor.describe.remote())
            assert info["sampler_exit_code"] is None
            if info["sample_written"]:
                break
            time.sleep(0.1)
        else:
            pytest.fail("External sampler did not write evidence")
        result = ray.get(monitor.stop.remote())
        assert result["exit_code"] == 0, result["stderr"]
        records = [
            json.loads(line) for line in gzip.decompress(result["data"]).splitlines()
        ]
        assert records
        assert "cpu" in records[0]["cpu_ticks"]
        assert records[0]["network"]
        assert records[0]["processes"]
        assert records[0]["memory"]["MemTotal_bytes"] > 0
    finally:
        ray.kill(monitor, no_restart=True)


def _progress(population):
    deadline = time.monotonic() + 60
    report = None
    while time.monotonic() < deadline:
        report = population.run_for_report_interval(0.1)
        if all(
            member["rollout"]["episodes_committed"] > 0
            and all(
                dict(counters)["optimizer_updates"] > 0
                for _, counters in member["learner"]["module_counters"]
            )
            for member in report["members"].values()
        ):
            return report
    pytest.fail(f"MPE population stalled: {report}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "discrete,gpu",
    [
        (0, 0),
        (1, 0),
        pytest.param(1, 0.5, marks=pytest.mark.gpu, id="shared-gpu-dqn-sac"),
    ],
)
def test_marl_population_learns_replaces_and_restores(
    ray_runtime, tmp_path, discrete, gpu
):
    if gpu and not torch.cuda.is_available():
        pytest.skip("Fractional placement acceptance requires a physical GPU")
    specs = []
    for index in range(2):
        rollout, modules = build_spread_configs(
            agents=2,
            discrete_agents=discrete,
            max_cycles=4,
            hidden_dim=8,
            batch_size=4,
            learning_starts=0,
            num_gpus=gpu,
            seed=30 + index,
        )
        runtime = AsyncRuntimeConfig(
            member_id=f"team-{index}",
            runner_count=1,
            max_episode_steps=4,
            batch_size=4,
            learner_updates_per_tick=2,
            publication_interval_updates=1,
            replay_capacity_transitions=256,
            replay_capacity_bytes=2_000_000,
            pending_commit_high_watermark=2,
            pending_commit_low_watermark=0,
            evaluation_interval_env_steps=0,
            evaluation_num_episodes=0,
            num_cpus_per_runner=0,
            num_cpus_per_learner=0,
            num_cpus_per_replay=0,
            num_gpus_per_learner=gpu,
            seed=30 + index,
        )
        specs.append(PopulationMemberSpec.for_team(rollout, modules, runtime))
    pbt = SimplePBTConfig(
        perturbation_interval_reports=10_000,
        min_episodes_after_restart=1,
        mutations={"sac/actor_lr": FloatMutation(1e-5, 1e-3)},
    )
    population = PopulationAsyncSAC(specs, pbt_config=pbt)
    restored = None
    try:
        population.start()
        _progress(population)

        def inspect(actor):
            return {
                "devices": {
                    mid: assert_learner_parameter_devices(
                        adapter._learner_group, "cuda" if gpu else "cpu"
                    )
                    for mid, adapter in actor._host._adapters.items()
                },
                "gpu_ids": ray.get_runtime_context()
                .get_accelerator_ids()
                .get("GPU", []),
            }

        before = [
            ray.get(member._learner_actor.__ray_call__.remote(inspect))
            for member in population.members.values()
        ]
        if gpu:
            assert len(before[0]["gpu_ids"]) == 1
            assert before[0]["gpu_ids"] == before[1]["gpu_ids"]
        old_ids = population.runtime_member_ids
        pumps = {
            slot: member._pump_iterations for slot, member in population.members.items()
        }
        population._reports_since_perturbation = 10_000
        event = population.get_report()["pbt"]
        assert event["event_happened"]
        target = event["target_slot"]
        donor = event["donor_slot"]
        assert population.runtime_member_ids[target] != old_ids[target]
        assert population.members[donor]._pump_iterations > pumps[donor]
        _progress(population)
        directory = tmp_path / "checkpoint"
        directory.mkdir()
        population.save_checkpoint(directory)
        ids = population.runtime_member_ids
        population.stop(graceful=False)
        restored = PopulationAsyncSAC.from_checkpoint(specs, directory, pbt_config=pbt)
        restored.start()
        assert restored.runtime_member_ids == ids
        _progress(restored)
        after = [
            ray.get(member._learner_actor.__ray_call__.remote(inspect))
            for member in restored.members.values()
        ]
        if gpu:
            assert after[0]["gpu_ids"] == after[1]["gpu_ids"]
    finally:
        for runtime in (population, restored):
            if runtime is not None:
                runtime.stop(graceful=False)
