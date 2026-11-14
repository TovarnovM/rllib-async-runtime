"""Graph algorithm seams against the pinned RLlib modules."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    QF_LOGITS,
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_PROBS,
    QF_TARGET_NEXT_PREDS,
    QF_TARGET_NEXT_PROBS,
)
from ray.rllib.core import COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib_async.gnn.encoder import GraphEncoderConfig
from rllib_async.gnn.graph_batch import pack_graph_observations
from rllib_async.gnn.rl_modules import GraphDQNTorchRLModule, GraphSACTorchRLModule
from rllib_async.learner.dqn_adapter import DQNLearnerAdapter
from rllib_async.learner.sac_adapter import SACLearnerAdapter
from tests.helpers import assert_finite_losses, make_dqn_config, make_sac_config


def _space(*, controlled: bool = True, masked: bool = False) -> gym.spaces.Dict:
    fields = {
        "node_features": gym.spaces.Box(-10.0, 10.0, (5, 2), dtype=np.float32),
        "edge_index": gym.spaces.Box(0, 4, (2, 4), dtype=np.int64),
        "node_count": gym.spaces.Box(1, 5, (), dtype=np.int64),
        "edge_count": gym.spaces.Box(0, 4, (), dtype=np.int64),
    }
    if controlled:
        fields["controlled_node"] = gym.spaces.Box(0, 4, (), dtype=np.int64)
    if masked:
        fields["action_mask"] = gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32)
    return gym.spaces.Dict(fields)


def _graph(node_count: int) -> dict[str, object]:
    return {
        "node_features": np.arange(2 * node_count, dtype=np.float32).reshape(-1, 2)
        / 10,
        "edge_index": np.array(
            [list(range(node_count - 1)), list(range(1, node_count))],
            dtype=np.int64,
        ),
        "controlled_node": 0,
    }


def _observations(*sizes: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value)
        for name, value in pack_graph_observations(
            [_graph(size) for size in sizes]
        ).items()
    }


def _batch() -> dict[str, object]:
    return {
        Columns.OBS: _observations(1, 3),
        Columns.NEXT_OBS: _observations(4, 1),
        Columns.ACTIONS: torch.tensor([0, 1]),
        Columns.REWARDS: torch.tensor([1.0, 0.0]),
        Columns.TERMINATEDS: torch.tensor([False, True]),
        Columns.TRUNCATEDS: torch.tensor([False, False]),
    }


def _dqn(*, double_q: bool = True, num_atoms: int = 1) -> GraphDQNTorchRLModule:
    module = GraphDQNTorchRLModule(
        observation_space=_space(),
        action_space=gym.spaces.Discrete(3),
        model_config={
            "fcnet_hiddens": [8],
            "head_fcnet_hiddens": [8],
            "double_q": double_q,
            "dueling": True,
            "num_atoms": num_atoms,
            "v_min": -10.0,
            "v_max": 10.0,
            "epsilon": 1.0,
            "validate_startup": True,
        },
    )
    module.make_target_networks()
    return module


@pytest.mark.parametrize("double_q", [False, True])
@pytest.mark.parametrize("num_atoms", [1, 5])
def test_graph_dqn_keeps_current_and_next_packed_indices_separate(
    double_q: bool, num_atoms: int
) -> None:
    module = _dqn(double_q=double_q, num_atoms=num_atoms)
    batch = _batch()
    output = module.forward_train(batch)
    current = module.compute_q_values({Columns.OBS: batch[Columns.OBS]})
    target = module.forward_target({Columns.OBS: batch[Columns.NEXT_OBS]})
    torch.testing.assert_close(output[QF_PREDS], current[QF_PREDS])
    torch.testing.assert_close(output[QF_TARGET_NEXT_PREDS], target[QF_PREDS])
    assert output[QF_PREDS].shape == (2, 3)
    if double_q:
        online_next = module.compute_q_values({Columns.OBS: batch[Columns.NEXT_OBS]})
        torch.testing.assert_close(output[QF_NEXT_PREDS], online_next[QF_PREDS])
    else:
        assert QF_NEXT_PREDS not in output
    if num_atoms > 1:
        for name in (QF_LOGITS, QF_PROBS, QF_TARGET_NEXT_PROBS):
            assert output[name].shape == (2, 3, num_atoms)
        torch.testing.assert_close(output[QF_PROBS].sum(-1), torch.ones(2, 3))


def test_graph_dqn_packed_batch_matches_individual_graph_predictions() -> None:
    module = _dqn()
    expected = torch.cat(
        [
            module.compute_q_values({Columns.OBS: _observations(n)})[QF_PREDS]
            for n in (1, 3, 5)
        ],
        dim=0,
    )
    actual = module.compute_q_values({Columns.OBS: _observations(1, 3, 5)})[QF_PREDS]
    torch.testing.assert_close(actual, expected)


def _sac(*, inference_only: bool = False) -> GraphSACTorchRLModule:
    return GraphSACTorchRLModule(
        observation_space=_space(),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        inference_only=inference_only,
        model_config={
            "twin_q": True,
            "fcnet_hiddens": [8],
            "head_fcnet_hiddens": [8],
            "graph_encoders": {
                "actor": {"hidden_dim": 5, "output_dim": 7},
                "q1": {"hidden_dim": 9, "output_dim": 11, "message_layers": 2},
                "q2": {"hidden_dim": 13, "output_dim": 15},
            },
        },
    )


def test_continuous_graph_sac_independent_encoders_and_action_gradient() -> None:
    torch.manual_seed(71)
    module = _sac()
    observations = _observations(1, 3)
    actions = torch.tensor([[0.1, -0.2], [0.4, -0.5]], requires_grad=True)
    batch = {Columns.OBS: observations, Columns.ACTIONS: actions}
    assert module.pi_encoder({Columns.OBS: observations})[ENCODER_OUT].shape == (2, 7)
    assert module.qf_encoder(batch)[ENCODER_OUT].shape == (2, 11)
    assert module.qf_twin_encoder(batch)[ENCODER_OUT].shape == (2, 15)
    groups = [
        {id(p) for p in encoder.parameters()}
        for encoder in (module.pi_encoder, module.qf_encoder, module.qf_twin_encoder)
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    for encoder, head in (
        (module.qf_encoder, module.qf),
        (module.qf_twin_encoder, module.qf_twin),
    ):
        q = module._qf_forward_train_helper(batch, encoder, head)
        gradient = torch.autograd.grad(q.sum(), actions)[0]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
        changed = module._qf_forward_train_helper(
            {Columns.OBS: observations, Columns.ACTIONS: actions.detach() + 0.5},
            encoder,
            head,
        )
        assert not torch.allclose(q, changed)


def test_graph_sac_actor_gradient_and_target_ownership() -> None:
    torch.manual_seed(72)
    module = _sac()
    module.make_target_networks()
    for online, target in module.get_target_network_pairs():
        for online_param, target_param in zip(
            online.parameters(), target.parameters(), strict=True
        ):
            assert online_param is not target_param
            assert not target_param.requires_grad
            torch.testing.assert_close(online_param, target_param)
    batch = _batch()
    batch[Columns.ACTIONS] = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    output = module.forward_train(batch)
    output["q_curr"].sum().backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in module.pi_encoder.parameters()
    )
    assert all(p.grad is None for p in module.qf_encoder.parameters())
    assert all(p.grad is None for p in module.qf_twin_encoder.parameters())
    assert all(p.grad is None for p in module.target_qf_encoder.parameters())
    assert torch.isfinite(output["q_target_next"]).all()
    inference = _sac(inference_only=True)
    inference.set_state(module.get_state(inference_only=True))
    assert not hasattr(inference, "qf_encoder")
    assert not hasattr(inference, "qf_twin_encoder")
    expected = module.forward_inference({Columns.OBS: batch[Columns.OBS]})
    actual = inference.forward_inference({Columns.OBS: batch[Columns.OBS]})
    torch.testing.assert_close(
        actual[Columns.ACTION_DIST_INPUTS], expected[Columns.ACTION_DIST_INPUTS]
    )


def test_global_mean_readout_has_no_controlled_node_dependency() -> None:
    encoder = GraphEncoderConfig(
        node_feature_dim=2,
        hidden_dim=5,
        output_dim=7,
        readout="mean",
        validate_startup=True,
    ).build()
    observations = _observations(1, 3)
    observations.pop("controlled_node")
    assert encoder({Columns.OBS: observations})[ENCODER_OUT].shape == (2, 7)
    module = GraphSACTorchRLModule(
        observation_space=_space(controlled=False),
        action_space=gym.spaces.Discrete(3),
        inference_only=True,
        model_config={
            "twin_q": True,
            "fcnet_hiddens": [5],
            "graph_encoder": {"readout": "mean"},
        },
    )
    assert module.forward_inference({Columns.OBS: observations})[
        Columns.ACTION_DIST_INPUTS
    ].shape == (2, 3)


@pytest.mark.parametrize(
    "config, message",
    [
        ({"use_lstm": True}, "stateless"),
        ({"graph_encoder": {"layers": 3}}, "unknown graph_encoder"),
        ({"graph_encoders": {"critic": {}}}, "only actor, q1, and q2"),
        ({"twin_q": False}, "discrete SAC requires twin_q=True"),
    ],
)
def test_graph_sac_rejects_unsupported_specs(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GraphSACTorchRLModule(
            observation_space=_space(),
            action_space=gym.spaces.Discrete(3),
            model_config={"twin_q": True, "fcnet_hiddens": [8], **config},
        )


@pytest.mark.parametrize("algorithm", ["dqn", "c51", "continuous_sac", "discrete_sac"])
@pytest.mark.parametrize(
    "num_gpus_per_learner",
    [pytest.param(0, id="cpu"), pytest.param(1, marks=pytest.mark.gpu, id="gpu")],
)
def test_graph_learner_update_has_finite_loss_and_updates_encoders(
    algorithm: str,
    num_gpus_per_learner: int,
) -> None:
    if num_gpus_per_learner and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this container")
    torch.manual_seed(71)
    discrete = algorithm != "continuous_sac"
    observation_space = _space(masked=discrete)
    action_space = (
        gym.spaces.Discrete(3)
        if discrete
        else gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    )
    graph_specs = {
        "actor": {"hidden_dim": 5, "output_dim": 7},
        "q1": {"hidden_dim": 9, "output_dim": 11},
        "q2": {"hidden_dim": 13, "output_dim": 15},
    }
    if algorithm in ("dqn", "c51"):
        config = make_dqn_config().training(num_atoms=5 if algorithm == "c51" else 1)
        module_class = GraphDQNTorchRLModule
        adapter_class = DQNLearnerAdapter
        encoder_prefixes = ("encoder.",)
        model_config = {"fcnet_hiddens": [8], "head_fcnet_hiddens": [8]}
    else:
        config = make_sac_config().training(target_entropy=-0.5)
        module_class = GraphSACTorchRLModule
        adapter_class = SACLearnerAdapter
        encoder_prefixes = ("pi_encoder.", "qf_encoder.", "qf_twin_encoder.")
        model_config = {
            "fcnet_hiddens": [8],
            "head_fcnet_hiddens": [8],
            "graph_encoders": graph_specs,
        }
    config.rl_module(
        model_config=model_config,
        rl_module_spec=RLModuleSpec(module_class=module_class),
    )
    config.learners(num_gpus_per_learner=num_gpus_per_learner)
    batch = {
        name: (
            {key: value.numpy() for key, value in column.items()}
            if isinstance(column, dict)
            else column.numpy()
        )
        for name, column in _batch().items()
    }
    if discrete:
        batch[Columns.OBS]["action_mask"] = np.array(
            [[1, 0, 1], [0, 1, 0]], dtype=np.float32
        )
        batch[Columns.NEXT_OBS]["action_mask"] = np.array(
            [[0, 0, 1], [0, 0, 0]], dtype=np.float32
        )
    else:
        batch[Columns.ACTIONS] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    adapter = adapter_class(
        config,
        spaces={DEFAULT_MODULE_ID: (observation_space, action_space)},
        member_id="graph",
        publication_interval_updates=1,
        validate_startup=True,
    )
    try:
        before = adapter.get_state()["learner_group"]["learner"][COMPONENT_RL_MODULE][
            DEFAULT_MODULE_ID
        ]
        before = {key: np.array(value, copy=True) for key, value in before.items()}
        result = adapter.update(batch, sampled_env_steps=2)
        assert result.performed
        assert result.published_weights is not None
        assert_finite_losses(result.learner_results)
        after = adapter.get_state()["learner_group"]["learner"][COMPONENT_RL_MODULE][
            DEFAULT_MODULE_ID
        ]
        for prefix in encoder_prefixes:
            assert any(
                not np.array_equal(value, after[key])
                for key, value in before.items()
                if key.startswith(prefix)
            ), prefix
    finally:
        adapter.close()
