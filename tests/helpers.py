from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import numpy as np
import torch
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core import (
    COMPONENT_OPTIMIZER,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from rllib_async.learner import SAC_TEMPERATURE_STATE


def make_sac_config(
    *,
    learner_class: type[SACTorchLearner] = SACTorchLearner,
    num_gpus_per_learner: int = 0,
) -> SACConfig:
    return (
        SACConfig()
        .environment("Pendulum-v1")
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
            learner_class=learner_class,
        )
        .training(
            actor_lr=3e-4,
            alpha_lr=3e-4,
            critic_lr=3e-4,
            n_step=1,
            num_steps_sampled_before_learning_starts=0,
            policy_model_config={"fcnet_hiddens": [32, 32]},
            q_model_config={"fcnet_hiddens": [32, 32]},
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 1_024,
            },
            train_batch_size_per_learner=32,
            twin_q=True,
        )
    )


def update_learner(learner_group: Any, episodes: Sequence[Any]) -> list[dict[str, Any]]:
    steps = sum(len(episode) for episode in episodes)
    return learner_group.update(
        episodes=copy.deepcopy(list(episodes)),
        timesteps={
            NUM_ENV_STEPS_SAMPLED_LIFETIME: steps,
            NUM_AGENT_STEPS_SAMPLED_LIFETIME: steps,
        },
    )


def assert_finite_losses(results: Any) -> None:
    losses: list[float] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if hasattr(value, "peek") and callable(value.peek):
            visit(value.peek(), path)
        elif isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, (*path, str(key)))
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))
        elif isinstance(value, np.ndarray):
            if any("loss" in key.lower() for key in path):
                losses.extend(np.asarray(value, dtype=float).reshape(-1).tolist())
        elif torch.is_tensor(value):
            if any("loss" in key.lower() for key in path):
                losses.extend(value.detach().cpu().reshape(-1).tolist())
        elif isinstance(value, Number) and any("loss" in key.lower() for key in path):
            losses.append(float(value))

    visit(results, ())
    assert losses, f"No loss metrics found in learner result: {results!r}"
    assert all(math.isfinite(value) for value in losses), losses


def assert_tree_close(left: Any, right: Any, *, path: str = "root") -> None:
    if torch.is_tensor(left):
        left = left.detach().cpu().numpy()
    if torch.is_tensor(right):
        right = right.detach().cpu().numpy()

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(left),
            np.asarray(right),
            rtol=1e-6,
            atol=1e-7,
            err_msg=path,
        )
        return

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        assert set(left) == set(right), path
        for key in left:
            assert_tree_close(left[key], right[key], path=f"{path}.{key}")
        return

    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, str | bytes)
        and not isinstance(right, str | bytes)
    ):
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            assert_tree_close(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(left, Number) and isinstance(right, Number):
        assert math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7), (
            path,
            left,
            right,
        )
        return

    assert left == right, (path, left, right)


def learner_checkpoint_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    learner = state["learner"]
    return {
        COMPONENT_RL_MODULE: learner[COMPONENT_RL_MODULE],
        COMPONENT_OPTIMIZER: learner[COMPONENT_OPTIMIZER],
        SAC_TEMPERATURE_STATE: learner[SAC_TEMPERATURE_STATE],
    }


def flattened_key_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return [prefix]
    paths: list[str] = []
    for key, nested in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        paths.extend(flattened_key_paths(nested, child))
    return paths
