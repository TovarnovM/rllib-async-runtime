from __future__ import annotations

import time
from dataclasses import replace

import pytest
import ray
import torch

from rllib_async.runtime import (
    FloatMutation,
    PopulationAsyncSAC,
    PopulationMemberSpec,
    SimplePBTConfig,
)
from tests.integration.test_population import make_population_specs


def _gpu_specs() -> tuple[PopulationMemberSpec, ...]:
    return tuple(
        PopulationMemberSpec(
            member.sac_config,
            replace(member.runtime_config, member_id=f"member-{index:02d}"),
        )
        for index, member in enumerate(make_population_specs(num_gpus_per_learner=1))
    )


def _pbt_config(*, seed: int) -> SimplePBTConfig:
    return SimplePBTConfig(
        perturbation_interval_reports=10_000,
        reward_window_episodes=8,
        min_episodes_after_restart=1,
        seed=seed,
        mutations={
            "actor_lr": FloatMutation(
                low=1e-5,
                high=1e-3,
                factors=(0.8,),
            )
        },
    )


def _wait_for_learning(population: PopulationAsyncSAC) -> dict:
    deadline = time.monotonic() + 60
    report = None
    while time.monotonic() < deadline:
        report = population.run_for_report_interval(0.1)
        if all(
            member["learner"]["learner_updates"] >= 1
            and member["train"]["episodes_since_metric_reset"] >= 1
            for member in report["members"].values()
        ):
            return report
    raise AssertionError(f"population did not become ready: {report}")


@pytest.mark.gpu
def test_two_gpu_pbt_checkpoint_resume_and_warm_start(
    ray_runtime: None,
    tmp_path,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for the PBT recovery gate")

    specs = _gpu_specs()
    pbt_config = _pbt_config(seed=20260729)
    checkpoint_dir = tmp_path / "pbt-checkpoint"
    checkpoint_dir.mkdir()
    source = PopulationAsyncSAC(
        specs,
        run_id="run-gpu-source",
        pbt_config=pbt_config,
    )
    try:
        source.start()
        report = _wait_for_learning(source)
        accelerator_ids = [
            tuple(member["learner"]["accelerator_ids"])
            for member in report["members"].values()
        ]
        assert all(len(ids) == 1 for ids in accelerator_ids)
        assert accelerator_ids[0] != accelerator_ids[1]

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
        _wait_for_learning(source)
        source.save_checkpoint(checkpoint_dir)
    finally:
        source.stop(graceful=False)

    resumed = PopulationAsyncSAC.from_checkpoint(
        specs,
        checkpoint_dir,
        pbt_config=pbt_config,
    )
    try:
        resumed.start()
        resumed_report = _wait_for_learning(resumed)
        assert resumed._exploit_count == 1
        assert all(
            member["controller"]["restore_count"] == 1
            for member in resumed_report["members"].values()
        )
    finally:
        resumed.stop(graceful=False)

    warm = PopulationAsyncSAC.from_warm_start_checkpoint(
        specs,
        checkpoint_dir,
        run_id="run-gpu-warm",
        pbt_config=_pbt_config(seed=7),
    )
    try:
        warm.start()
        warm_report = _wait_for_learning(warm)
        assert warm._exploit_count == 0
        assert warm._generations == {"member-00": 0, "member-01": 0}
        assert all(
            member["controller"]["restore_count"] == 0
            for member in warm_report["members"].values()
        )
        assert ray.get(warm.replay_actor.get_stats.remote()).cursor.mutation_seq > 0
    finally:
        warm.stop(graceful=False)
