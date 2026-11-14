"""Numerical history oracles and the pinned RLlib learner compatibility gate."""

from __future__ import annotations

import copy
import json
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_TARGET_NEXT_PREDS,
)
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib_async.gnn.graph_batch import pack_graph_observations
from rllib_async.learner.action_masks import ActionMaskedSACTorchRLModule
from rllib_async.learner.dqn_adapter import DQNLearnerAdapter
from rllib_async.learner.recurrent import (
    RECURRENT,
    RecurrentDQNTorchRLModule,
    RecurrentSACTorchRLModule,
    TorchHistoryEncoder,
)
from rllib_async.learner.sac_adapter import SACLearnerAdapter
from tests.helpers import (
    assert_finite_losses,
    assert_learner_parameter_devices,
    assert_tree_close,
    make_dqn_config,
    make_sac_config,
)


def _space(graph: bool) -> gym.Space:
    if not graph:
        return gym.spaces.Box(-10.0, 10.0, (3,), dtype=np.float32)
    return gym.spaces.Dict(
        {
            "node_features": gym.spaces.Box(-10.0, 10.0, (4, 2), dtype=np.float32),
            "edge_index": gym.spaces.Box(0, 3, (2, 3), dtype=np.int64),
            "node_count": gym.spaces.Box(1, 4, (), dtype=np.int64),
            "edge_count": gym.spaces.Box(0, 3, (), dtype=np.int64),
            "controlled_node": gym.spaces.Box(0, 3, (), dtype=np.int64),
        }
    )


def _graph(size: int, index: int) -> dict[str, Any]:
    return {
        "node_features": np.arange(size * 2, dtype=np.float32).reshape(-1, 2) / 10 + index / 20,
        "edge_index": np.asarray([range(size - 1), range(1, size)], dtype=np.int64),
        "controlled_node": 0,
    }


def _tensors(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tensors(item) for key, item in value.items()}
    return torch.as_tensor(value)


def _arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _arrays(item) for key, item in value.items()}
    return value.detach().cpu().numpy()


def _batch(*, graph: bool = False, continuous: bool = False) -> dict[str, Any]:
    current = [2, 3, 5]
    following = [3, 4, 7]
    if graph:
        graphs = [_graph(size, index) for index, size in enumerate((1, 3, 2, 4, 1, 2, 1, 3))]
        observations = pack_graph_observations(graphs)
        obs = pack_graph_observations([graphs[index] for index in current])
        next_obs = pack_graph_observations([graphs[index] for index in following])
    else:
        observations = np.arange(24, dtype=np.float32).reshape(8, 3) / 20
        obs, next_obs = observations[current], observations[following]
    return _tensors(
        {
            Columns.OBS: obs,
            Columns.NEXT_OBS: next_obs,
            Columns.ACTIONS: np.asarray([[0.2], [-0.3], [0.4]] if continuous else [0, 1, 2], dtype=np.float32 if continuous else np.int64),
            Columns.REWARDS: np.asarray([0.4, -0.2, 0.7], dtype=np.float32),
            Columns.TERMINATEDS: np.asarray([False, True, False]),
            Columns.TRUNCATEDS: np.asarray([False, False, True]),
            "weights": np.ones(3, dtype=np.float32),
            "n_step": np.asarray([1, 1, 2], dtype=np.int64),
            RECURRENT: {
                Columns.OBS: observations,
                "seq_lens": np.asarray([5, 3], dtype=np.int64),
                "prefix_lens": np.asarray([2, 0], dtype=np.int64),
                "current_indices": np.asarray(current, dtype=np.int64),
                "next_indices": np.asarray(following, dtype=np.int64),
            },
        }
    )


def _dqn(*, graph: bool = False, num_atoms: int = 1) -> RecurrentDQNTorchRLModule:
    module = RecurrentDQNTorchRLModule(
        observation_space=_space(graph),
        action_space=gym.spaces.Discrete(3),
        model_config={
            "fcnet_hiddens": [8],
            "head_fcnet_hiddens": [8],
            "recurrent_hidden_dim": 7,
            "double_q": True,
            "dueling": True,
            "num_atoms": num_atoms,
            "v_min": -10.0,
            "v_max": 10.0,
            "epsilon": 0.0,
        },
    )
    module.make_target_networks()
    return module


def _sac(*, graph: bool = False, continuous: bool = False, inference_only: bool = False) -> RecurrentSACTorchRLModule:
    module = RecurrentSACTorchRLModule(
        observation_space=_space(graph),
        action_space=gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32) if continuous else gym.spaces.Discrete(3),
        inference_only=inference_only,
        model_config={
            "fcnet_hiddens": [8],
            "head_fcnet_hiddens": [8],
            "recurrent_hidden_dim": {"actor": 5, "q1": 7, "q2": 9},
            "graph_encoders": {
                "actor": {"hidden_dim": 6},
                "q1": {"hidden_dim": 8},
                "q2": {"hidden_dim": 10},
            } if graph else {},
            "twin_q": True,
        },
    )
    if not inference_only:
        module.make_target_networks()
    return module


def _oracle_history(encoder: TorchHistoryEncoder, history: dict[str, Any]) -> torch.Tensor:
    """Encode the complete history in one pass, without prefix slicing logic."""
    features = encoder.observation_encoder({Columns.OBS: history[Columns.OBS]})[ENCODER_OUT]
    return torch.cat(
        [encoder.gru(part.unsqueeze(0))[0][0] for part in features.split(history["seq_lens"].tolist())]
    )


@pytest.mark.parametrize("graph", [False, True])
def test_prefix_matches_full_history_and_reencodes_current_weights(graph: bool) -> None:
    torch.manual_seed(31)
    encoder = _dqn(graph=graph).encoder
    history = _batch(graph=graph)[RECURRENT]
    actual = encoder.encode_history(history)
    torch.testing.assert_close(actual, _oracle_history(encoder, history))
    payload = copy.deepcopy(history)
    with torch.no_grad():
        encoder.gru.bias_ih_l0.add_(0.3)
    changed = encoder.encode_history(history)
    assert not torch.allclose(actual, changed)
    assert_tree_close(history, payload)
    torch.testing.assert_close(changed, _oracle_history(encoder, history))


def test_prefix_has_no_gradient_and_ragged_tail_does_not_mix_memories() -> None:
    encoder = _dqn().encoder
    history = _batch()[RECURRENT]
    history[Columns.OBS].requires_grad_()
    actual = encoder.encode_history(history)
    actual[history["current_indices"]].sum().backward()
    gradients = history[Columns.OBS].grad
    assert torch.count_nonzero(gradients[:2]) == 0
    assert torch.count_nonzero(gradients[2:4]) > 0
    assert torch.count_nonzero(gradients[4]) == 0
    assert torch.count_nonzero(gradients[6:]) == 0
    # Adding another longer history cannot affect an earlier sequence or its h.
    first = {
        Columns.OBS: history[Columns.OBS][:5].detach(),
        "seq_lens": torch.tensor([5]),
        "prefix_lens": torch.tensor([2]),
    }
    torch.testing.assert_close(actual[:5], encoder.encode_history(first))


@pytest.mark.parametrize("num_atoms", [1, 5])
def test_dqn_bootstrap_uses_its_own_target_history(num_atoms: int) -> None:
    module = _dqn(graph=True, num_atoms=num_atoms)
    batch = _batch(graph=True)
    with torch.no_grad():
        module._target_encoder.gru.bias_ih_l0.add_(0.4)
    output = module.forward_train(batch)
    history = batch[RECURRENT]
    online = _oracle_history(module.encoder, history)
    target = _oracle_history(module._target_encoder, history)

    def values(latent: torch.Tensor, target_network: bool) -> torch.Tensor:
        heads = {"af": module._target_af, "vf": module._target_vf} if target_network else {"af": module.af, "vf": module.vf}
        return module._qf_forward_helper(
            {Columns.OBS: latent}, lambda row: {ENCODER_OUT: row[Columns.OBS]}, heads
        )[QF_PREDS]

    torch.testing.assert_close(output[QF_PREDS], values(online[history["current_indices"]], False))
    torch.testing.assert_close(output[QF_NEXT_PREDS], values(online[history["next_indices"]], False))
    torch.testing.assert_close(output[QF_TARGET_NEXT_PREDS], values(target[history["next_indices"]], True))
    assert not torch.allclose(online, target)
    assert not output[QF_TARGET_NEXT_PREDS].requires_grad


class _OracleEncoder(torch.nn.Module):
    def __init__(self, encoder: TorchHistoryEncoder, batch: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = encoder
        self.current_observations = batch[Columns.OBS]
        history = batch[RECURRENT]
        encoded = _oracle_history(encoder, history)
        self.current = encoded[history["current_indices"]]
        self.following = encoded[history["next_indices"]]

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {ENCODER_OUT: self.current if batch[Columns.OBS] is self.current_observations else self.following}


@pytest.mark.parametrize("continuous", [False, True])
def test_recurrent_sac_forward_matches_stock_forward_on_full_history_oracle(continuous: bool) -> None:
    module = _sac(graph=True, continuous=continuous)
    batch = _batch(graph=True, continuous=continuous)
    torch.manual_seed(51)
    actual = module.forward_train(batch)
    names = ("pi_encoder", "qf_encoder", "qf_twin_encoder", "target_qf_encoder", "target_qf_twin_encoder")
    encoders = {name: getattr(module, name) for name in names}
    try:
        for name, encoder in encoders.items():
            setattr(module, name, _OracleEncoder(encoder, batch))
        torch.manual_seed(51)
        expected = ActionMaskedSACTorchRLModule._forward_train(module, batch)
        assert set(actual) == set(expected)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    finally:
        for name, encoder in encoders.items():
            setattr(module, name, encoder)


def test_sac_critic_action_gradient_has_fixed_history_and_correct_parameter_owners() -> None:
    torch.manual_seed(72)
    module = _sac(continuous=True)
    batch = _batch(continuous=True)
    assert [encoder.gru.hidden_size for encoder in (module.pi_encoder, module.qf_encoder, module.qf_twin_encoder)] == [5, 7, 9]
    output = module.forward_train(batch)
    output["q_curr"].sum().backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in module.pi_encoder.parameters())
    for encoder in (module.qf_encoder, module.qf_twin_encoder, module.target_qf_encoder, module.target_qf_twin_encoder):
        assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in module.qf.parameters())
    assert all(parameter.grad is None for parameter in module.qf_twin.parameters())
    history = batch[RECURRENT]
    before = module.qf_encoder.encode_history(history).detach()
    actions = batch[Columns.ACTIONS].detach().clone().requires_grad_()
    latent = before[history["current_indices"]]
    q = module.qf(torch.cat((latent, actions), dim=-1))
    assert torch.count_nonzero(torch.autograd.grad(q.sum(), actions)[0]) > 0
    changed = module.qf(torch.cat((latent, actions + 0.5), dim=-1))
    assert not torch.allclose(q, changed)
    torch.testing.assert_close(module.qf_encoder.encode_history(history), before)
    for online, target in module.get_target_network_pairs():
        for current, delayed in zip(online.parameters(), target.parameters(), strict=True):
            assert current is not delayed
            assert not delayed.requires_grad


@pytest.mark.parametrize("algorithm", ["dqn", "sac"])
def test_inference_agent_memories_are_independent_and_sac_publication_is_actor_only(algorithm: str) -> None:
    module = _dqn() if algorithm == "dqn" else _sac()
    observations = torch.tensor([[[0.2, 0.1, -0.3]], [[-0.4, 0.2, 0.1]]])
    initial = module.get_initial_state()["h"]
    together = module.forward_inference({Columns.OBS: observations, Columns.STATE_IN: {"h": initial.repeat(2, 1)}})
    for index in range(2):
        separate = module.forward_inference({Columns.OBS: observations[index:index + 1], Columns.STATE_IN: {"h": initial.unsqueeze(0)}})
        torch.testing.assert_close(together[Columns.STATE_OUT]["h"][index:index + 1], separate[Columns.STATE_OUT]["h"])
    assert not torch.allclose(together[Columns.STATE_OUT]["h"][0], together[Columns.STATE_OUT]["h"][1])
    if algorithm == "sac":
        inference = _sac(inference_only=True)
        state = module.get_state(inference_only=True)
        assert not any("qf" in key or "target" in key for key in state)
        inference.set_state(state)
        copied = inference.forward_inference({Columns.OBS: observations, Columns.STATE_IN: {"h": initial.repeat(2, 1)}})
        assert_tree_close(together, copied)


@pytest.mark.parametrize("algorithm, graph", [("dqn", False), ("dqn", True), ("discrete_sac", False), ("continuous_sac", True)])
@pytest.mark.parametrize("gpus", [pytest.param(0, id="cpu"), pytest.param(1, marks=pytest.mark.gpu, id="gpu")])
def test_recurrent_learner_update_and_next_update_after_restore(algorithm: str, graph: bool, gpus: int, request: pytest.FixtureRequest) -> None:
    if gpus and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this container")
    continuous = algorithm == "continuous_sac"
    is_dqn = algorithm == "dqn"
    config = make_dqn_config() if is_dqn else make_sac_config().training(target_entropy=-0.5)
    config.learners(num_gpus_per_learner=gpus)
    config.rl_module(
        model_config={"fcnet_hiddens": [8], "head_fcnet_hiddens": [8], "recurrent_hidden_dim": 7},
        rl_module_spec=RLModuleSpec(module_class=RecurrentDQNTorchRLModule if is_dqn else RecurrentSACTorchRLModule),
    )
    spaces = {DEFAULT_MODULE_ID: (_space(graph), gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32) if continuous else gym.spaces.Discrete(3))}
    adapter_class = DQNLearnerAdapter if is_dqn else SACLearnerAdapter
    source = adapter_class(config, spaces=spaces, member_id="recurrent", publication_interval_updates=1, validate_startup=True)
    restored = None
    try:
        devices = assert_learner_parameter_devices(source._learner_group, "cuda" if gpus else "cpu")
        request.node.user_properties.append(("parameter_devices", json.dumps(devices)))
        batch = _arrays(_batch(graph=graph, continuous=continuous))
        initial = copy.deepcopy(source.get_state()["learner_group"]["learner"][COMPONENT_RL_MODULE][DEFAULT_MODULE_ID])
        update = source.update(batch, sampled_env_steps=8)
        assert update.performed
        assert_finite_losses(update.learner_results)
        checkpoint = copy.deepcopy(source.get_state())
        learned = checkpoint["learner_group"]["learner"][COMPONENT_RL_MODULE][DEFAULT_MODULE_ID]
        prefixes = ("encoder.gru.",) if is_dqn else ("pi_encoder.gru.", "qf_encoder.gru.", "qf_twin_encoder.gru.")
        for prefix in prefixes:
            assert any(not np.array_equal(value, learned[key]) for key, value in initial.items() if key.startswith(prefix)), prefix
        expected = source.update(batch, sampled_env_steps=16)
        restored = adapter_class(config, spaces=spaces, member_id="recurrent", publication_interval_updates=1, validate_startup=True)
        restored.set_state(copy.deepcopy(checkpoint))
        resumed = restored.update(batch, sampled_env_steps=16)
        assert_finite_losses(expected.learner_results)
        assert_finite_losses(resumed.learner_results)
        source_state = source.get_state()["learner_group"]["learner"]
        restored_state = restored.get_state()["learner_group"]["learner"]
        for key in (COMPONENT_RL_MODULE, COMPONENT_OPTIMIZER):
            assert_tree_close(source_state[key], restored_state[key])
        assert_tree_close(source.get_published_weights().state, restored.get_published_weights().state)
    finally:
        if restored is not None:
            restored.close()
        source.close()
