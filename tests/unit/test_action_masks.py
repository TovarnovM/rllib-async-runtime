from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_PROBS,
    QF_TARGET_NEXT_PREDS,
)
from ray.rllib.core.columns import Columns

from rllib_async.learner.action_masks import (
    ActionMaskedDQNTorchRLModule,
    epsilon_greedy_actions,
    mask_action_values,
    mask_dqn_train_output,
    masked_action_probabilities,
)


def test_epsilon_sampling_includes_zero_q_and_excludes_invalid_actions() -> None:
    values = torch.tensor([[0.0, 1_000.0, 0.0]]).expand(1024, -1)
    observation = {"action_mask": torch.tensor([[1, 0, 1]]).expand_as(values)}
    with torch.random.fork_rng():
        torch.manual_seed(12)
        actions = epsilon_greedy_actions(values, observation, epsilon=1.0)
    assert set(actions.tolist()) == {0, 2}


def test_epsilon_sampling_with_unmasked_zero_initialized_q_head() -> None:
    values = torch.zeros(1024, 3)
    with torch.random.fork_rng():
        torch.manual_seed(12)
        actions = epsilon_greedy_actions(values, torch.zeros(1024, 4), epsilon=1.0)
    assert set(actions.tolist()) == {0, 1, 2}


def test_masked_greedy_action_ignores_high_invalid_q() -> None:
    values = torch.tensor([[-4.0, 100.0, -2.0]])
    observation = {"action_mask": torch.tensor([[1, 0, 1]])}
    assert mask_action_values(values, observation).argmax(-1).item() == 2
    assert epsilon_greedy_actions(values, observation, epsilon=0.0).item() == 2


@pytest.mark.parametrize("helper", [mask_action_values, masked_action_probabilities])
def test_empty_nonterminal_mask_and_shape_mismatch_are_errors(helper) -> None:
    values = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="nonterminal.*no legal action"):
        helper(
            values,
            {"action_mask": torch.tensor([[0, 0, 0], [1, 1, 1]])},
            terminateds=torch.tensor([False, True]),
        )
    with pytest.raises(ValueError, match="same shape"):
        helper(values, {"action_mask": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="terminateds"):
        helper(
            values,
            {"action_mask": torch.ones_like(values)},
            terminateds=torch.ones(2, 1),
        )


def test_double_dqn_masks_selection_and_evaluation_and_keeps_regression_raw() -> None:
    current_q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    current_probs = torch.full((2, 3, 5), 0.2)
    output = {
        QF_PREDS: current_q,
        QF_PROBS: current_probs,
        QF_NEXT_PREDS: torch.tensor([[3.0, 1_000.0, 1.0], [5.0, 6.0, 7.0]]),
        QF_TARGET_NEXT_PREDS: torch.tensor([[4.0, 2_000.0, 9.0], [8.0, 9.0, 10.0]]),
    }
    batch = {
        Columns.OBS: {"action_mask": torch.ones(2, 3)},
        Columns.NEXT_OBS: {"action_mask": torch.tensor([[1, 0, 1], [0, 0, 0]])},
        Columns.TERMINATEDS: torch.tensor([False, True]),
    }

    masked = mask_dqn_train_output(output, batch)

    assert masked[QF_PREDS] is current_q
    assert masked[QF_PROBS] is current_probs
    selected = masked[QF_NEXT_PREDS].argmax(-1, keepdim=True)
    double_targets = masked[QF_TARGET_NEXT_PREDS].gather(-1, selected).squeeze(-1)
    torch.testing.assert_close(double_targets, torch.tensor([4.0, 0.0]))
    torch.testing.assert_close(
        masked[QF_TARGET_NEXT_PREDS].max(-1).values, torch.tensor([9.0, 0.0])
    )
    bellman_targets = (
        torch.tensor([1.0, 2.0])
        + 0.9 * (1.0 - batch[Columns.TERMINATEDS].float()) * double_targets
    )
    torch.testing.assert_close(bellman_targets, torch.tensor([4.6, 2.0]))


def test_dqn_current_empty_mask_is_invalid_even_on_terminal_transition() -> None:
    with pytest.raises(ValueError, match="nonterminal.*no legal action"):
        mask_dqn_train_output(
            {QF_PREDS: torch.ones(1, 2)},
            {
                Columns.OBS: {"action_mask": torch.zeros(1, 2)},
                Columns.TERMINATEDS: torch.ones(1, dtype=torch.bool),
            },
        )


def test_dqn_module_masks_inference_and_preserves_exploration_state() -> None:
    updates = []
    state = torch.tensor([[7.0]])
    module = SimpleNamespace(
        compute_q_values=lambda batch: {
            QF_PREDS: torch.tensor([[0.0, 10.0, 0.0]]),
            Columns.STATE_OUT: state,
        },
        epsilon_schedule=SimpleNamespace(
            update=updates.append, get_current_value=lambda: 1.0
        ),
    )
    batch = {Columns.OBS: {"action_mask": torch.tensor([[1, 0, 0]])}}
    inference = ActionMaskedDQNTorchRLModule._forward_inference(module, batch)
    exploration = ActionMaskedDQNTorchRLModule._forward_exploration(module, batch, 17)
    assert inference[Columns.ACTIONS].item() == 0
    assert exploration[Columns.ACTIONS].item() == 0
    assert exploration[Columns.STATE_OUT] is state
    assert updates == [17]


def test_sac_masks_soft_target_and_entropy_without_nan() -> None:
    # log(3) gives probabilities [1/4, 0, 3/4] over legal actions.
    logits = torch.tensor([[0.0, 100.0, math.log(3.0)]], requires_grad=True)
    probs, log_probs = masked_action_probabilities(
        logits, {"action_mask": torch.tensor([[1, 0, 1]])}
    )
    torch.testing.assert_close(probs, torch.tensor([[0.25, 0.0, 0.75]]))
    assert log_probs[0, 1] == 0.0
    q_values = torch.tensor([[2.0, 1_000_000.0, 6.0]])
    expected_q = (probs * q_values).sum(-1)
    torch.testing.assert_close(expected_q, torch.tensor([5.0]))
    entropy = -(probs * log_probs).sum(-1)
    expected_entropy = -(0.25 * torch.log(torch.tensor(0.25))) - (
        0.75 * torch.log(torch.tensor(0.75))
    )
    torch.testing.assert_close(entropy, expected_entropy.reshape(1))
    (expected_q + entropy).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0.0


def test_sac_single_legal_action_and_terminal_empty_mask_have_finite_losses() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    probs, log_probs = masked_action_probabilities(
        logits,
        {"action_mask": torch.tensor([[0, 1, 0], [0, 0, 0]])},
        terminateds=torch.tensor([False, True]),
    )
    torch.testing.assert_close(probs[0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(log_probs[0], torch.zeros(3))
    soft_targets = (probs * (torch.ones(2, 3) - 0.2 * log_probs)).sum(-1)
    torch.testing.assert_close(
        soft_targets * torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    )
    loss = (probs * (0.2 * log_probs - torch.ones(2, 3))).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_sac_log_probabilities_remain_finite_when_softmax_underflows() -> None:
    logits = torch.tensor([[0.0, -1_000.0]], requires_grad=True)
    probs, log_probs = masked_action_probabilities(logits, {})
    loss = (probs * log_probs).sum()
    loss.backward()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
