from __future__ import annotations

import pytest
import torch
from ray.air import CheckpointConfig, RunConfig

from rllib_async.runtime import PopulationLauncher
from tests.integration.test_population import (
    PopulationReadyStopper,
    make_population_specs,
    population_actor_name,
)


@pytest.mark.gpu
def test_two_members_update_concurrently_on_distinct_gpus(
    ray_runtime: None,
    tmp_path,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for the Phase 8 GPU gate")

    launcher = PopulationLauncher(
        make_population_specs(num_gpus_per_learner=1),
        replay_actor_name=population_actor_name("two-gpu-replay"),
    )
    try:
        results = launcher.fit(
            run_config=RunConfig(
                name="phase-8-two-gpu",
                storage_path=str(tmp_path / "ray-results"),
                stop=PopulationReadyStopper(),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_at_end=True,
                ),
                verbose=0,
            )
        )
        metrics = [result.metrics for result in results]
        node_ids = [result["learner"]["node_id"] for result in metrics]
        accelerator_ids = [
            tuple(result["learner"]["accelerator_ids"]) for result in metrics
        ]
        assert all(node_ids)
        assert len(set(node_ids)) == 1
        assert all(len(ids) == 1 for ids in accelerator_ids)
        assert accelerator_ids[0] != accelerator_ids[1]
        assert all(result["learner"]["learner_updates"] >= 1 for result in metrics)
        assert all(
            set(result["fast_replay"]["active_producer_episode_counts"])
            == {"member-0", "member-1"}
            for result in metrics
        )
        intervals = [
            (
                result["controller"]["started_at_monotonic"],
                result["controller"]["reported_at_monotonic"],
            )
            for result in metrics
        ]
        assert max(start for start, _ in intervals) < min(end for _, end in intervals)
    finally:
        launcher.close()
