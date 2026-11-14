"""Observation-history GRUs with RLlib's unmodified DQN and SAC losses.

The replay collator supplies ragged raw histories and flat learning indices.
Each network reconstructs its own prefix with its current weights. Only valid
learning positions reach the stock losses; neither prefix nor padding is a
training example. Rollout state contains the inference network's memory only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    ATOMS,
    QF_LOGITS,
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_PROBS,
    QF_TARGET_NEXT_PREDS,
    QF_TARGET_NEXT_PROBS,
)
from ray.rllib.algorithms.dqn.dqn_catalog import DQNCatalog
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.sac_learner import (
    ACTION_DIST_INPUTS_NEXT,
    ACTION_LOG_PROBS,
    ACTION_LOG_PROBS_NEXT,
    ACTION_PROBS,
    ACTION_PROBS_NEXT,
    QF_TARGET_NEXT,
    QF_TWIN_PREDS,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel

from rllib_async.gnn.encoder import (
    graph_encoder_config,
    graph_model_config_for_role,
)
from rllib_async.gnn.episodes import (
    ACTION_MASK,
    CONTROLLED_NODE,
    EDGE_FEATURES,
    EDGE_INDEX,
    NODE_FEATURES,
)
from rllib_async.learner.action_masks import (
    ActionMaskedDQNTorchRLModule,
    ActionMaskedSACTorchRLModule,
    mask_action_values,
    mask_dqn_train_output,
    masked_action_probabilities,
)

RECURRENT = "recurrent"


def _slice_observations(observations: Any, start: int, stop: int) -> Any:
    """Slice contiguous observations without joining packed graph index spaces."""
    if not isinstance(observations, Mapping):
        return observations[start:stop]
    if "graph_ptr" not in observations:
        return {key: value[start:stop] for key, value in observations.items()}
    pointers = observations["graph_ptr"]
    first, last = int(pointers[start]), int(pointers[stop])
    edges = observations[EDGE_INDEX]
    selected = (edges[0] >= first) & (edges[0] < last)
    result = {
        NODE_FEATURES: observations[NODE_FEATURES][first:last],
        EDGE_INDEX: edges[:, selected] - first,
        "graph_ptr": pointers[start : stop + 1] - first,
    }
    if CONTROLLED_NODE in observations:
        result[CONTROLLED_NODE] = observations[CONTROLLED_NODE][start:stop] - first
    if EDGE_FEATURES in observations:
        result[EDGE_FEATURES] = observations[EDGE_FEATURES][selected]
    if ACTION_MASK in observations:
        result[ACTION_MASK] = observations[ACTION_MASK][start:stop]
    return result


@dataclass
class HistoryEncoderConfig(ModelConfig):
    """A standard observation encoder followed by one agent-level GRU."""

    observation_encoder: ModelConfig | None = None
    hidden_dim: int = 32

    @property
    def output_dims(self) -> tuple[int, ...]:
        return (self.hidden_dim,)

    def build(self, framework: str = "torch") -> Encoder:
        if framework != "torch":
            raise ValueError("history encoders require PyTorch")
        if self.observation_encoder is None:
            raise ValueError("history encoder requires an observation encoder")
        if (
            not isinstance(self.hidden_dim, int)
            or isinstance(self.hidden_dim, bool)
            or self.hidden_dim < 1
        ):
            raise ValueError("recurrent_hidden_dim must be a positive integer")
        return TorchHistoryEncoder(self)


class TorchHistoryEncoder(TorchModel, Encoder):
    """Keep observation memory independent of the action evaluated by a critic."""

    def __init__(self, config: HistoryEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        assert config.observation_encoder is not None
        self.observation_encoder = config.observation_encoder.build(framework="torch")
        self.gru = torch.nn.GRU(
            config.observation_encoder.output_dims[0],
            config.hidden_dim,
            batch_first=True,
        )

    def get_initial_state(self) -> dict[str, torch.Tensor]:
        return {"h": self.gru.weight_ih_l0.new_zeros(self.config.hidden_dim)}

    def _forward(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        observations = inputs[Columns.OBS]
        if isinstance(observations, Mapping):
            nodes = observations[NODE_FEATURES]
            time_axis = "graph_ptr" not in observations and nodes.ndim == 4
            if time_axis:
                batch_size, time_size = nodes.shape[:2]
                observations = {
                    key: value.reshape(batch_size * time_size, *value.shape[2:])
                    for key, value in observations.items()
                }
        else:
            time_axis = observations.ndim == 3
            if time_axis:
                batch_size, time_size = observations.shape[:2]
                observations = observations.reshape(batch_size * time_size, -1)
        features = self.observation_encoder({Columns.OBS: observations})[ENCODER_OUT]
        if time_axis:
            features = features.reshape(batch_size, time_size, -1)
        else:
            features = features.unsqueeze(1)
        state = inputs.get(Columns.STATE_IN)
        hidden = None if state is None else state["h"].unsqueeze(0)
        outputs, hidden = self.gru(features, hidden)
        return {
            ENCODER_OUT: outputs if time_axis else outputs[:, 0],
            Columns.STATE_OUT: {"h": hidden[0]},
        }

    def encode_history(self, history: Mapping[str, Any]) -> torch.Tensor:
        """Reconstruct each history; detach only its explicit burn-in prefix.

        Histories are ragged, so a short sequence never consumes another
        sequence's observations or advances through synthetic padded positions.
        Bounded burn-in uses the same zero initial state as full-prefix mode;
        the collator makes its missing earlier history explicit.
        """
        observations = history[Columns.OBS]
        lengths = history["seq_lens"].tolist()
        prefixes = history["prefix_lens"].tolist()
        parts = []
        offset = 0
        for length, prefix in zip(lengths, prefixes, strict=True):
            hidden = None
            if prefix:
                with torch.no_grad():
                    features = self.observation_encoder(
                        {Columns.OBS: _slice_observations(observations, offset, offset + prefix)}
                    )[ENCODER_OUT]
                    encoded, hidden = self.gru(features.unsqueeze(0), hidden)
                parts.append(encoded[0])
                hidden = hidden.detach()
            features = self.observation_encoder(
                {
                    Columns.OBS: _slice_observations(
                        observations, offset + prefix, offset + length
                    )
                }
            )[ENCODER_OUT]
            encoded, _ = self.gru(features.unsqueeze(0), hidden)
            parts.append(encoded[0])
            offset += length
        return torch.cat(parts, dim=0)


def _history_encoder_config(
    observation_space: gym.Space, model_config: Mapping[str, Any]
) -> HistoryEncoderConfig:
    if model_config.get("use_lstm") or model_config.get("use_attention"):
        raise ValueError("recurrent modules use an observation-history GRU")
    if isinstance(observation_space, gym.spaces.Dict):
        base = graph_encoder_config(observation_space, model_config)
    elif isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1:
        base = Catalog._get_encoder_config(observation_space, dict(model_config))
    else:
        raise ValueError("recurrent observations require a vector Box or graph Dict")
    return HistoryEncoderConfig(
        input_dims=base.input_dims,
        observation_encoder=base,
        hidden_dim=model_config.get("recurrent_hidden_dim", 32),
    )


class RecurrentDQNCatalog(DQNCatalog):
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict[str, Any],
        action_space: gym.Space | None = None,
    ) -> ModelConfig:
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("recurrent DQN requires a Discrete action space")
        if model_config_dict.get("graph_encoders"):
            raise ValueError("recurrent DQN uses graph_encoder, not SAC graph_encoders")
        return _history_encoder_config(observation_space, model_config_dict)


class RecurrentSACCatalog(SACCatalog):
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict[str, Any],
        action_space: gym.Space | None = None,
    ) -> ModelConfig:
        del action_space
        return _history_encoder_config(observation_space, model_config_dict)


def _role_config(model_config: Mapping[str, Any], role: str) -> dict[str, Any]:
    result = graph_model_config_for_role(model_config, role)
    hidden = result.get("recurrent_hidden_dim", 32)
    if isinstance(hidden, Mapping):
        if set(hidden) - {"actor", "q1", "q2"} or role not in hidden:
            raise ValueError("recurrent_hidden_dim must specify each actor/q1/q2 role")
        result["recurrent_hidden_dim"] = hidden[role]
    return result


def _encoded(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse RLlib's DQN heads after independently reconstructing memory."""
    return {ENCODER_OUT: inputs[Columns.OBS]}


class RecurrentDQNTorchRLModule(ActionMaskedDQNTorchRLModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("catalog_class") is None:
            kwargs["catalog_class"] = RecurrentDQNCatalog
        super().__init__(*args, **kwargs)

    def _forward_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.inference_only:
            raise RuntimeError("recurrent DQN training requires inference_only=False")
        if RECURRENT not in batch:
            raise ValueError("recurrent DQN requires raw replay history metadata")
        history = batch[RECURRENT]
        current_indices = history["current_indices"].long()
        next_indices = history["next_indices"].long()
        encoded = self.encoder.encode_history(history)
        heads = {"af": self.af, "vf": self.vf} if self.uses_dueling else self.af
        current = self._qf_forward_helper(
            {Columns.OBS: encoded[current_indices]}, _encoded, heads
        )
        with torch.no_grad():
            target_encoded = self._target_encoder.encode_history(history)
            target_heads = (
                {"af": self._target_af, "vf": self._target_vf}
                if self.uses_dueling
                else self._target_af
            )
            target = self._qf_forward_helper(
                {Columns.OBS: target_encoded[next_indices]}, _encoded, target_heads
            )
            next_values = (
                self._qf_forward_helper(
                    {Columns.OBS: encoded[next_indices]}, _encoded, heads
                )[QF_PREDS]
                if self.uses_double_q
                else None
            )
        output = {QF_PREDS: current[QF_PREDS], QF_TARGET_NEXT_PREDS: target[QF_PREDS]}
        if self.uses_double_q:
            output[QF_NEXT_PREDS] = next_values
        if self.num_atoms > 1:
            output.update(
                {
                    ATOMS: target[ATOMS],
                    QF_LOGITS: current[QF_LOGITS],
                    QF_PROBS: current[QF_PROBS],
                    QF_TARGET_NEXT_PROBS: target[QF_PROBS],
                }
            )
        return mask_dqn_train_output(output, batch)


class RecurrentSACTorchRLModule(ActionMaskedSACTorchRLModule):
    """Independent actor/critic GRUs; candidate actions never alter memory."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("catalog_class") is None:
            kwargs["catalog_class"] = RecurrentSACCatalog
        model_config = kwargs.get("model_config")
        if model_config is not None:
            self._role_model_config = dict(model_config)
            kwargs["model_config"] = _role_config(model_config, "actor")
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        self.twin_q = self.model_config["twin_q"]
        if isinstance(self.action_space, gym.spaces.Discrete) and not self.twin_q:
            raise ValueError("Ray 2.56.1 discrete SAC requires twin_q=True")
        if isinstance(self.action_space, gym.spaces.Box) and len(self.action_space.shape) != 1:
            raise ValueError("recurrent continuous SAC requires a vector Box action space")
        if (
            isinstance(self.action_space, gym.spaces.Box)
            and isinstance(self.observation_space, gym.spaces.Dict)
            and ACTION_MASK in self.observation_space.spaces
        ):
            raise ValueError("action_mask requires a Discrete action space")
        self.pi_encoder = self.catalog.build_encoder(framework=self.framework)
        self.pi = self.catalog.build_pi_head(framework=self.framework)
        if not self.inference_only:
            self.qf_encoder, self.qf = self._build_critic("q1")
            if self.twin_q:
                self.qf_twin_encoder, self.qf_twin = self._build_critic("q2")

    def _build_critic(self, role: str) -> tuple[Encoder, torch.nn.Module]:
        config = _role_config(getattr(self, "_role_model_config", self.model_config), role)
        catalog = RecurrentSACCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=config,
        )
        encoder = catalog.build_encoder(framework=self.framework)
        if isinstance(self.action_space, gym.spaces.Box):
            catalog.qf_head_config.input_dims = (
                catalog.latent_dims[0] + self.action_space.shape[0],
            )
        return encoder, catalog.build_qf_head(framework=self.framework)

    def get_initial_state(self) -> dict[str, torch.Tensor]:
        return self.pi_encoder.get_initial_state()

    def _forward_inference(self, batch: dict[str, Any]) -> dict[str, Any]:
        encoded = self.pi_encoder(batch)
        logits = self.pi(encoded[ENCODER_OUT])
        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = mask_action_values(logits, batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: encoded[Columns.STATE_OUT],
        }

    def _qf_forward_train_helper(
        self,
        batch: dict[str, Any],
        encoder: Encoder,
        head: torch.nn.Module,
        squeeze: bool = True,
    ) -> torch.Tensor:
        encoded = encoder(batch)[ENCODER_OUT]
        if isinstance(self.action_space, gym.spaces.Box):
            encoded = torch.cat((encoded, batch[Columns.ACTIONS]), dim=-1)
        output = head(encoded)
        return output.squeeze(-1) if squeeze else output

    def _forward_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.inference_only:
            raise RuntimeError("recurrent SAC training requires inference_only=False")
        if RECURRENT not in batch:
            raise ValueError("recurrent SAC requires raw replay history metadata")
        history = batch[RECURRENT]
        current = history["current_indices"].long()
        following = history["next_indices"].long()
        actor = self.pi_encoder.encode_history(history)
        q1 = self.qf_encoder.encode_history(history)[current]
        q2 = self.qf_twin_encoder.encode_history(history)[current] if self.twin_q else None
        logits = self.pi(actor[current])
        with torch.no_grad():
            logits_next = self.pi(actor[following])
            target_q1 = self.target_qf_encoder.encode_history(history)[following]
            target_q2 = (
                self.target_qf_twin_encoder.encode_history(history)[following]
                if self.twin_q
                else None
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self._discrete_outputs(batch, logits, logits_next, q1, q2, target_q1, target_q2)
        return self._continuous_outputs(batch, logits, logits_next, q1, q2, target_q1, target_q2)

    def _discrete_outputs(
        self, batch: dict[str, Any], logits: torch.Tensor, logits_next: torch.Tensor,
        q1: torch.Tensor, q2: torch.Tensor | None,
        target_q1: torch.Tensor, target_q2: torch.Tensor | None,
    ) -> dict[str, Any]:
        probs, logs = masked_action_probabilities(logits, batch[Columns.OBS])
        next_probs, next_logs = masked_action_probabilities(
            logits_next, batch[Columns.NEXT_OBS], terminateds=batch[Columns.TERMINATEDS]
        )
        with torch.no_grad():
            target = self.target_qf(target_q1)
            if self.twin_q:
                target = torch.minimum(target, self.target_qf_twin(target_q2))
        output = {
            ACTION_PROBS: probs, ACTION_LOG_PROBS: logs,
            ACTION_PROBS_NEXT: next_probs, ACTION_LOG_PROBS_NEXT: next_logs,
            QF_TARGET_NEXT: target, QF_PREDS: self.qf(q1),
        }
        if self.twin_q:
            output[QF_TWIN_PREDS] = self.qf_twin(q2)
        return output

    def _continuous_outputs(
        self, batch: dict[str, Any], logits: torch.Tensor, logits_next: torch.Tensor,
        q1: torch.Tensor, q2: torch.Tensor | None,
        target_q1: torch.Tensor, target_q2: torch.Tensor | None,
    ) -> dict[str, Any]:
        dist_class = self.get_train_action_dist_cls()
        distribution = dist_class.from_logits(logits)
        next_distribution = dist_class.from_logits(logits_next)
        actions = distribution.rsample()
        next_actions = next_distribution.sample().detach()
        output = {
            Columns.ACTION_DIST_INPUTS: logits,
            ACTION_DIST_INPUTS_NEXT: logits_next,
            "logp_resampled": distribution.logp(actions),
            "logp_next_resampled": next_distribution.logp(next_actions).detach(),
            QF_PREDS: self.qf(torch.cat((q1, batch[Columns.ACTIONS]), dim=-1)).squeeze(-1),
        }
        if self.twin_q:
            output[QF_TWIN_PREDS] = self.qf_twin(
                torch.cat((q2, batch[Columns.ACTIONS]), dim=-1)
            ).squeeze(-1)
        # Detached observations keep the critic's history fixed. Freezing only
        # head parameters still permits dQ/da for the reparameterized actor.
        parameters = list(self.qf.parameters())
        if self.twin_q:
            parameters += list(self.qf_twin.parameters())
        requires_grad = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            q_curr = self.qf(torch.cat((q1.detach(), actions), dim=-1)).squeeze(-1)
            if self.twin_q:
                q_curr = torch.minimum(
                    q_curr,
                    self.qf_twin(torch.cat((q2.detach(), actions), dim=-1)).squeeze(-1),
                )
            output["q_curr"] = q_curr
        finally:
            for parameter, enabled in zip(parameters, requires_grad, strict=True):
                parameter.requires_grad_(enabled)
        with torch.no_grad():
            target = self.target_qf(torch.cat((target_q1, next_actions), dim=-1)).squeeze(-1)
            if self.twin_q:
                target = torch.minimum(
                    target,
                    self.target_qf_twin(torch.cat((target_q2, next_actions), dim=-1)).squeeze(-1),
                )
            output["q_target_next"] = target
        return output
