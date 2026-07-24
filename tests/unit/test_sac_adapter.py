from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from ray.rllib.core import (
    COMPONENT_OPTIMIZER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from rllib_async.learner import (
    SAC_TARGET_UPDATE_STATE,
    SAC_TEMPERATURE_STATE,
    SACBatchError,
    SACLearnerAdapter,
    build_rllib_sac_batch,
)
from tests.helpers import (
    assert_finite_losses,
    assert_tree_close,
    make_sac_config,
)


def make_fixed_batch(size: int = 32) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260724)
    return {
        Columns.OBS: rng.normal(size=(size, 3)).astype(np.float32),
        Columns.NEXT_OBS: rng.normal(size=(size, 3)).astype(np.float32),
        Columns.ACTIONS: rng.uniform(-2, 2, size=(size, 1)).astype(np.float32),
        Columns.REWARDS: rng.normal(size=size).astype(np.float64),
        Columns.TERMINATEDS: np.zeros(size, dtype=np.bool_),
        Columns.TRUNCATEDS: np.zeros(size, dtype=np.bool_),
    }


def learner_payload(state: Mapping[str, object]) -> dict[str, object]:
    learner = state["learner"]
    assert isinstance(learner, Mapping)
    return {
        COMPONENT_RL_MODULE: learner[COMPONENT_RL_MODULE],
        COMPONENT_OPTIMIZER: learner[COMPONENT_OPTIMIZER],
        SAC_TEMPERATURE_STATE: learner[SAC_TEMPERATURE_STATE],
        SAC_TARGET_UPDATE_STATE: learner[SAC_TARGET_UPDATE_STATE],
    }


def loss_values(results: object) -> dict[str, object]:
    losses: dict[str, object] = {}

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, (*path, str(key)))
        elif isinstance(value, list | tuple):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))
        elif any("loss" in key.lower() for key in path):
            if hasattr(value, "peek") and callable(value.peek):
                value = value.peek()
            losses[".".join(path)] = value

    visit(results, ())
    assert losses
    return losses


def test_build_rllib_sac_batch_normalizes_exact_columns() -> None:
    flat = make_fixed_batch(4)

    batch = build_rllib_sac_batch(flat)

    assert batch.env_steps() == 4
    assert set(batch.policy_batches) == {DEFAULT_MODULE_ID}
    sample_batch = batch.policy_batches[DEFAULT_MODULE_ID]
    assert tuple(sample_batch) == (
        Columns.OBS,
        Columns.NEXT_OBS,
        Columns.ACTIONS,
        Columns.REWARDS,
        Columns.TERMINATEDS,
        Columns.TRUNCATEDS,
        "n_step",
        "weights",
    )
    assert sample_batch[Columns.REWARDS].dtype == np.float32
    assert sample_batch[Columns.TERMINATEDS].dtype == np.bool_
    assert sample_batch[Columns.TRUNCATEDS].dtype == np.bool_
    np.testing.assert_array_equal(
        sample_batch["n_step"],
        np.ones(4, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        sample_batch["weights"],
        np.ones(4, dtype=np.float32),
    )
    assert all(np.asarray(value).flags.c_contiguous for value in sample_batch.values())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch: batch.pop(Columns.NEXT_OBS),
            "missing columns",
        ),
        (
            lambda batch: batch.__setitem__("unknown", np.ones(4)),
            "unsupported columns",
        ),
        (
            lambda batch: batch.__setitem__(
                Columns.REWARDS,
                np.ones((4, 1)),
            ),
            "one-dimensional",
        ),
        (
            lambda batch: batch.__setitem__(
                Columns.TERMINATEDS,
                np.zeros(4, dtype=np.int64),
            ),
            "must be boolean",
        ),
        (
            lambda batch: batch.__setitem__("n_step", np.zeros(4)),
            "positive integers",
        ),
        (
            lambda batch: batch.__setitem__(
                "weights",
                np.array([1.0, 1.0, np.nan, 1.0]),
            ),
            "finite non-negative",
        ),
        (
            lambda batch: batch.__setitem__(
                Columns.OBS,
                np.full((4, 3), np.nan),
            ),
            "finite values",
        ),
        (
            lambda batch: batch.__setitem__(
                Columns.NEXT_OBS,
                np.ones((4, 2)),
            ),
            "identical shapes",
        ),
    ],
)
def test_build_rllib_sac_batch_rejects_ambiguous_columns(
    mutate,
    message: str,
) -> None:
    batch = make_fixed_batch(4)
    mutate(batch)

    with pytest.raises(SACBatchError, match=message):
        build_rllib_sac_batch(batch)


def test_adapter_matches_stock_rllib_fixed_batch_and_target_schedule() -> None:
    config = make_sac_config()
    config.training(
        num_steps_sampled_before_learning_starts=0,
        target_network_update_freq=10,
        tau=1.0,
    )
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    baseline = adapter = None

    try:
        adapter = SACLearnerAdapter(
            config,
            spaces=runner.get_spaces(),
            member_id="member-0",
            publication_interval_updates=2,
        )
        baseline = config.build_learner_group(spaces=runner.get_spaces())
        adapter_state = adapter.get_state()
        baseline.set_state(copy.deepcopy(adapter_state["learner_group"]))
        flat_batch = make_fixed_batch()
        rllib_batch = build_rllib_sac_batch(flat_batch)

        initial = adapter.get_published_weights()
        assert dict(initial.module_versions) == {DEFAULT_MODULE_ID: 0}
        assert initial.learner_updates == 0

        for sampled_steps, expected_target_ts in ((9, 0), (10, 10)):
            timesteps = {
                NUM_ENV_STEPS_SAMPLED_LIFETIME: sampled_steps,
                NUM_AGENT_STEPS_SAMPLED_LIFETIME: sampled_steps,
            }
            torch.manual_seed(20260724 + sampled_steps)
            baseline_result = baseline.update(
                batch=copy.deepcopy(rllib_batch),
                timesteps=timesteps,
            )
            torch.manual_seed(20260724 + sampled_steps)
            adapter_result = adapter.update(
                flat_batch,
                sampled_env_steps=sampled_steps,
            )

            assert adapter_result.performed
            assert_finite_losses(baseline_result)
            assert_finite_losses(adapter_result.learner_results)
            assert_tree_close(
                loss_values(baseline_result),
                loss_values(adapter_result.learner_results),
            )
            current = adapter.get_state()["learner_group"]
            assert_tree_close(
                learner_payload(current)[COMPONENT_RL_MODULE],
                baseline.get_state()["learner"][COMPONENT_RL_MODULE],
            )
            assert_tree_close(
                learner_payload(current)[COMPONENT_OPTIMIZER],
                baseline.get_state()["learner"][COMPONENT_OPTIMIZER],
            )
            alpha = baseline._learner.curr_log_alpha[DEFAULT_MODULE_ID]
            np.testing.assert_allclose(
                learner_payload(current)[SAC_TEMPERATURE_STATE][DEFAULT_MODULE_ID],
                alpha.detach().cpu().numpy(),
                rtol=1e-6,
                atol=1e-7,
            )
            assert (
                learner_payload(current)[SAC_TARGET_UPDATE_STATE][DEFAULT_MODULE_ID]
                == expected_target_ts
            )

        assert adapter_result.published_weights is not None
        assert dict(adapter_result.published_weights.module_versions) == {
            DEFAULT_MODULE_ID: 1
        }
        assert adapter_result.published_weights.learner_updates == 2
    finally:
        if baseline is not None:
            baseline.shutdown()
        if adapter is not None:
            adapter.close()
        runner.stop()


def test_learning_start_and_full_checkpoint_restore_next_update() -> None:
    config = make_sac_config()
    config.training(
        num_steps_sampled_before_learning_starts=5,
        target_network_update_freq=10,
        tau=1.0,
    )
    runner = SingleAgentEnvRunner(config=config, worker_index=0)
    restored_runner = SingleAgentEnvRunner(config=config, worker_index=1)
    source = restored = None

    try:
        source = SACLearnerAdapter(
            config,
            spaces=runner.get_spaces(),
            member_id="member-0",
            publication_interval_updates=2,
        )
        flat_batch = make_fixed_batch()
        initial_checkpoint = source.get_state()
        assert (
            learner_payload(initial_checkpoint["learner_group"])[
                SAC_TARGET_UPDATE_STATE
            ][DEFAULT_MODULE_ID]
            == 0
        )
        source.set_state(copy.deepcopy(initial_checkpoint))

        skipped = source.update(flat_batch, sampled_env_steps=4)
        assert not skipped.performed
        assert skipped.learner_updates == 0

        torch.manual_seed(11)
        first = source.update(flat_batch, sampled_env_steps=5)
        assert first.performed
        assert first.published_weights is None
        torch.manual_seed(12)
        second = source.update(flat_batch, sampled_env_steps=10)
        assert second.performed
        assert second.published_weights is not None

        checkpoint = source.get_state()
        checkpoint = pickle.loads(pickle.dumps(checkpoint))
        payload = learner_payload(checkpoint["learner_group"])
        assert payload[SAC_TARGET_UPDATE_STATE][DEFAULT_MODULE_ID] == 10
        assert not np.allclose(
            payload[SAC_TEMPERATURE_STATE][DEFAULT_MODULE_ID],
            0.0,
        )
        assert payload[COMPONENT_OPTIMIZER]
        assert checkpoint["learner_updates"] == 2
        assert checkpoint["sampled_env_steps"] == 10
        assert checkpoint["last_published_update"] == 2
        assert dict(checkpoint["latest_weights"].module_versions) == {
            DEFAULT_MODULE_ID: 1
        }
        assert checkpoint["torch_rng_state"].dtype == torch.uint8

        restored = SACLearnerAdapter(
            config,
            spaces=restored_runner.get_spaces(),
            member_id="member-0",
            publication_interval_updates=2,
        )
        restored.set_state(copy.deepcopy(checkpoint))
        assert_tree_close(
            learner_payload(source.get_state()["learner_group"]),
            learner_payload(restored.get_state()["learner_group"]),
        )
        assert_tree_close(
            restored.get_published_weights().state,
            checkpoint["latest_weights"].state,
        )
        incompatible = copy.deepcopy(checkpoint)
        incompatible["config_contract"]["tau"] = 0.5
        with pytest.raises(ValueError, match="configuration"):
            restored.set_state(incompatible)
        assert_tree_close(
            learner_payload(source.get_state()["learner_group"]),
            learner_payload(restored.get_state()["learner_group"]),
        )

        for sampled_steps, expected_target_ts in (
            (19, 10),
            (20, 20),
        ):
            step_checkpoint = source.get_state()
            source_result = source.update(
                flat_batch,
                sampled_env_steps=sampled_steps,
            )
            restored.set_state(copy.deepcopy(step_checkpoint))
            restored_result = restored.update(
                flat_batch,
                sampled_env_steps=sampled_steps,
            )
            assert_tree_close(
                loss_values(source_result.learner_results),
                loss_values(restored_result.learner_results),
            )
            assert bool(source_result.published_weights) == bool(
                restored_result.published_weights
            )
            assert_tree_close(
                learner_payload(source.get_state()["learner_group"]),
                learner_payload(restored.get_state()["learner_group"]),
            )
            assert (
                learner_payload(source.get_state()["learner_group"])[
                    SAC_TARGET_UPDATE_STATE
                ][DEFAULT_MODULE_ID]
                == expected_target_ts
            )

        assert source_result.published_weights is not None
        assert dict(source_result.published_weights.module_versions) == {
            DEFAULT_MODULE_ID: 2
        }
        assert source_result.published_weights.learner_updates == 4
    finally:
        if source is not None:
            source.close()
        if restored is not None:
            restored.close()
        runner.stop()
        restored_runner.stop()
