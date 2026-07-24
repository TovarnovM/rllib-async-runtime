from __future__ import annotations

import os

import gymnasium as gym
import pytest
import ray
import torch

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
        env = gym.make("Pendulum-v1")
        learner_group = config.build_learner_group(env=env)
        try:
            return {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_count": torch.cuda.device_count(),
                "is_local": learner_group.is_local,
            }
        finally:
            learner_group.shutdown()
            env.close()

    result = ray.get(build_gpu_learner.remote())
    assert result["device_count"] == 1
    assert result["is_local"] is True
    assert result["cuda_visible_devices"] not in (None, "")
