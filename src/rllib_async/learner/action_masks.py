"""Discrete action masks at RLlib's module boundary, with stock losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    QF_LOGITS,
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_PROBS,
    QF_TARGET_NEXT_PREDS,
)
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
    DefaultDQNTorchRLModule,
)
from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    QF_TARGET_NEXT,
    QF_TWIN_PREDS,
)
from ray.rllib.algorithms.sac.sac_learner import (
    QF_PREDS as SAC_QF_PREDS,
)
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT


def _legal_actions(
    values: torch.Tensor,
    observation: Any,
    *,
    terminateds: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Read a binary mask already validated at observation ingress.

    Shape and empty-row checks remain here because they are required to make
    sampling and reductions well-defined. There are no repeated feature scans.
    Terminal next states may have no action: their bootstrap is disabled by the
    learner, so use a nonempty distribution for intermediate computations.
    """

    if not isinstance(observation, Mapping) or "action_mask" not in observation:
        return None
    legal = torch.as_tensor(observation["action_mask"], device=values.device).bool()
    if legal.shape != values.shape:
        raise ValueError("action_mask must have the same shape as action values")
    if terminateds is not None:
        terminal = torch.as_tensor(terminateds, device=values.device).bool()
        if terminal.shape != values.shape[:-1]:
            raise ValueError("terminateds must match the action batch shape")
        legal = legal | terminal.unsqueeze(-1)
    if not torch.all(legal.any(dim=-1)):
        raise ValueError("a nonterminal observation has no legal action")
    return legal


def mask_action_values(
    values: torch.Tensor,
    observation: Any,
    *,
    terminateds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask expected Q or categorical logits; zero terminal bootstrap values."""

    legal = _legal_actions(values, observation, terminateds=terminateds)
    if legal is None:
        return values
    masked = values.masked_fill(~legal, -torch.inf)
    if terminateds is not None:
        terminal = torch.as_tensor(terminateds, device=values.device).bool()
        masked = torch.where(terminal.unsqueeze(-1), 0.0, masked)
    return masked


def epsilon_greedy_actions(
    values: torch.Tensor,
    observation: Any,
    epsilon: float,
) -> torch.Tensor:
    """Sample uniformly among legal actions, including those with Q == 0."""

    legal = _legal_actions(values, observation)
    if legal is None:
        legal = torch.ones_like(values, dtype=torch.bool)
    greedy = values.masked_fill(~legal, -torch.inf).argmax(dim=-1)
    random_actions = torch.multinomial(
        legal.reshape(-1, values.shape[-1]).float(), num_samples=1
    ).reshape(values.shape[:-1])
    explore = torch.rand(values.shape[:-1], device=values.device) < epsilon
    return torch.where(explore, random_actions, greedy)


def mask_dqn_train_output(
    output: dict[str, Any], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Mask next-action selection without altering Q regression or C51 atoms."""

    # A current observation always represents an actual decision, even when
    # the transition leads to a terminal next observation.
    _legal_actions(output[QF_PREDS], batch[Columns.OBS])
    for key in (QF_NEXT_PREDS, QF_TARGET_NEXT_PREDS):
        if key in output:
            output[key] = mask_action_values(
                output[key],
                batch[Columns.NEXT_OBS],
                terminateds=batch[Columns.TERMINATEDS],
            )
    return output


def masked_action_probabilities(
    logits: torch.Tensor,
    observation: Any,
    *,
    terminateds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return categorical probabilities and finite logs for SAC expectations.

    Invalid actions have exactly zero probability and a zero log term, making
    the entropy convention 0 log(0) = 0 explicit for the unmodified SAC loss.
    log_softmax also avoids log(softmax) underflow for legal actions.
    """

    legal = _legal_actions(logits, observation, terminateds=terminateds)
    if legal is not None:
        logits = logits.masked_fill(~legal, -torch.inf)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    if legal is not None:
        log_probs = log_probs.masked_fill(~legal, 0.0)
    return probs, log_probs


class ActionMaskedDQNTorchRLModule(DefaultDQNTorchRLModule):
    """Default DQN with semantic masks and value-independent random actions."""

    def _forward_inference(self, batch: dict[str, Any]) -> dict[str, Any]:
        qf_outs = self.compute_q_values(batch)
        output = {
            Columns.ACTIONS: mask_action_values(
                qf_outs[QF_PREDS], batch[Columns.OBS]
            ).argmax(dim=-1)
        }
        if Columns.STATE_OUT in qf_outs:
            output[Columns.STATE_OUT] = qf_outs[Columns.STATE_OUT]
        return output

    def _forward_exploration(self, batch: dict[str, Any], t: int) -> dict[str, Any]:
        qf_outs = self.compute_q_values(batch)
        self.epsilon_schedule.update(t)
        output = {
            Columns.ACTIONS: epsilon_greedy_actions(
                qf_outs[QF_PREDS],
                batch[Columns.OBS],
                self.epsilon_schedule.get_current_value(),
            )
        }
        if Columns.STATE_OUT in qf_outs:
            output[Columns.STATE_OUT] = qf_outs[Columns.STATE_OUT]
        return output

    def _forward_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        output = super()._forward_train(batch)
        if self.uses_double_q and self.num_atoms > 1:
            # Ray 2.56.1 concatenates current/next observations but only splits
            # expected Q-values. C51 regression needs current-state logits and
            # probabilities, not the concatenated 2B distributions.
            batch_size = output[QF_PREDS].shape[0]
            for key in (QF_LOGITS, QF_PROBS):
                output[key] = output[key][:batch_size]
        return mask_dqn_train_output(output, batch)


class ActionMaskedSACTorchRLModule(DefaultSACTorchRLModule):
    """Masked discrete SAC forwards; continuous SAC retains RLlib's forward."""

    def _forward_inference(self, batch: dict[str, Any]) -> dict[str, Any]:
        output = super()._forward_inference(batch)
        if isinstance(self.action_space, gym.spaces.Discrete):
            output[Columns.ACTION_DIST_INPUTS] = mask_action_values(
                output[Columns.ACTION_DIST_INPUTS], batch[Columns.OBS]
            )
        return output

    def _forward_train_discrete(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Preserve the stock critic/target computations; only the categorical
        # probabilities and entropy inputs differ from RLlib's discrete forward.
        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}
        logits_next = self.pi(self.pi_encoder(batch_next)[ENCODER_OUT])
        probs_next, log_probs_next = masked_action_probabilities(
            logits_next,
            batch[Columns.NEXT_OBS],
            terminateds=batch[Columns.TERMINATEDS],
        )
        output = {
            ACTION_PROBS_NEXT: probs_next,
            ACTION_LOG_PROBS_NEXT: log_probs_next,
            QF_TARGET_NEXT: self.forward_target(batch_next, squeeze=False),
            SAC_QF_PREDS: self._qf_forward_train_helper(
                batch_curr, self.qf_encoder, self.qf, squeeze=False
            ),
        }
        if self.twin_q:
            output[QF_TWIN_PREDS] = self._qf_forward_train_helper(
                batch_curr, self.qf_twin_encoder, self.qf_twin, squeeze=False
            )
        logits = self.pi(self.pi_encoder(batch_curr)[ENCODER_OUT])
        output[ACTION_PROBS], output[ACTION_LOG_PROBS] = masked_action_probabilities(
            logits, batch[Columns.OBS]
        )
        return output
