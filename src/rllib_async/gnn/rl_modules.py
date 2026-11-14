"""Graph-specific construction and batching, retaining RLlib's learner losses."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    ATOMS,
    QF_LOGITS,
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_PROBS,
    QF_TARGET_NEXT_PREDS,
    QF_TARGET_NEXT_PROBS,
)
from ray.rllib.core.columns import Columns

from rllib_async.gnn.encoder import (
    GraphDQNCatalog,
    SharedGraphSACCatalog,
    graph_model_config_for_role,
)
from rllib_async.learner.action_masks import (
    ActionMaskedDQNTorchRLModule,
    ActionMaskedSACTorchRLModule,
    mask_dqn_train_output,
)


class GraphDQNTorchRLModule(ActionMaskedDQNTorchRLModule):
    """Run current and next graphs separately, preserving their index spaces."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("catalog_class") is None:
            kwargs["catalog_class"] = GraphDQNCatalog
        super().__init__(*args, **kwargs)

    def _forward_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.inference_only:
            raise RuntimeError("graph DQN training requires inference_only=False")
        if Columns.STATE_IN in batch or Columns.NEXT_STATE_IN in batch:
            raise ValueError("graph DQN does not support recurrent state yet")
        current = self.compute_q_values({Columns.OBS: batch[Columns.OBS]})
        next_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}
        target_next = self.forward_target(next_batch)
        output = {
            QF_PREDS: current[QF_PREDS],
            QF_TARGET_NEXT_PREDS: target_next[QF_PREDS],
        }
        if self.uses_double_q:
            output[QF_NEXT_PREDS] = self.compute_q_values(next_batch)[QF_PREDS]
        if self.num_atoms > 1:
            output.update(
                {
                    ATOMS: target_next[ATOMS],
                    QF_LOGITS: current[QF_LOGITS],
                    QF_PROBS: current[QF_PROBS],
                    QF_TARGET_NEXT_PROBS: target_next[QF_PROBS],
                }
            )
        return mask_dqn_train_output(output, batch)


class GraphSACTorchRLModule(ActionMaskedSACTorchRLModule):
    """Build actor and twin critics from explicit independent graph specs.

    The continuous critics use SACCatalog's observation/action multi-stream
    encoder. RLlib still owns forward losses, optimizers, and target updates.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("catalog_class") is None:
            kwargs["catalog_class"] = SharedGraphSACCatalog
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        self.twin_q = self.model_config["twin_q"]
        if isinstance(self.action_space, gym.spaces.Discrete) and not self.twin_q:
            raise ValueError("Ray 2.56.1 discrete SAC requires twin_q=True")
        if not self.twin_q and "q2" in self.model_config.get("graph_encoders", {}):
            raise ValueError("a q2 encoder spec requires twin_q=True")
        self.pi_encoder = self.catalog.build_encoder(framework=self.framework)
        self.pi = self.catalog.build_pi_head(framework=self.framework)
        if not self.inference_only:
            q1_catalog = self._critic_catalog("q1")
            self.qf_encoder = q1_catalog.build_qf_encoder(framework=self.framework)
            self.qf = q1_catalog.build_qf_head(framework=self.framework)
            if self.twin_q:
                q2_catalog = self._critic_catalog("q2")
                self.qf_twin_encoder = q2_catalog.build_qf_encoder(
                    framework=self.framework
                )
                self.qf_twin = q2_catalog.build_qf_head(framework=self.framework)

    def _critic_catalog(self, role: str) -> SharedGraphSACCatalog:
        return SharedGraphSACCatalog(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=graph_model_config_for_role(self.model_config, role),
        )
