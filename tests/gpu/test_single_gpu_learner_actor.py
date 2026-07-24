from __future__ import annotations

import os

import numpy as np
import pytest
import ray
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

from rllib_async.learner import SACLearnerAdapter
from tests.helpers import make_sac_config


@pytest.mark.gpu
def test_ray_actor_exposes_exactly_one_gpu_to_local_learner(
    ray_runtime: None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this container")

    @ray.remote(num_gpus=1)
    def build_gpu_learner() -> dict[str, object]:
        config = make_sac_config(num_gpus_per_learner=1)
        runner = SingleAgentEnvRunner(config=config, worker_index=0)
        adapter = SACLearnerAdapter(
            config,
            spaces=runner.get_spaces(),
            member_id="member-0",
            publication_interval_updates=1,
        )
        try:
            batch = {
                Columns.OBS: np.zeros((32, 3), dtype=np.float32),
                Columns.NEXT_OBS: np.ones((32, 3), dtype=np.float32),
                Columns.ACTIONS: np.zeros((32, 1), dtype=np.float32),
                Columns.REWARDS: np.zeros(32, dtype=np.float32),
                Columns.TERMINATEDS: np.zeros(32, dtype=np.bool_),
                Columns.TRUNCATEDS: np.zeros(32, dtype=np.bool_),
            }
            update = adapter.update(batch, sampled_env_steps=32)
            checkpoint = adapter.get_state()
            adapter.set_state(checkpoint)
            return {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_count": torch.cuda.device_count(),
                "update_performed": update.performed,
                "published_version": dict(
                    update.published_weights.module_versions
                    if update.published_weights is not None
                    else {}
                ),
                "cuda_rng_states": len(checkpoint["torch_cuda_rng_states"]),
            }
        finally:
            adapter.close()
            runner.stop()

    result = ray.get(build_gpu_learner.remote())
    assert result["device_count"] == 1
    assert result["update_performed"] is True
    assert result["published_version"] == {"default_policy": 1}
    assert result["cuda_rng_states"] == 1
    assert result["cuda_visible_devices"] not in (None, "")
