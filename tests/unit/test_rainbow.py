"""Rainbow's model semantics, weighted stock loss, and local PER composition."""

from __future__ import annotations

import copy
import json
import pickle
import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    ATOMS,
    QF_LOGITS,
    QF_NEXT_PREDS,
    QF_PREDS,
    QF_PROBS,
    QF_TARGET_NEXT_PROBS,
)
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import (
    DefaultDQNTorchRLModule,
)
from ray.rllib.core import COMPONENT_OPTIMIZER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib_async.gnn.graph_batch import pack_graph_observations
from rllib_async.gnn.rl_modules import GraphDQNTorchRLModule
from rllib_async.learner.action_masks import ActionMaskedDQNTorchRLModule
from rllib_async.learner.dqn_adapter import DQNLearnerAdapter, prepare_dqn_config
from rllib_async.learner.noisy import ControlledNoisyLayer
from rllib_async.learner.recurrent import RecurrentDQNTorchRLModule
from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    MultiModuleTransition,
    RawMultiModuleEpisodeCodec,
)
from rllib_async.replay import EpisodeStore, FastReplay
from rllib_async.replay.prioritized import PrioritizedSequenceSampler
from rllib_async.replay.sequences import SequenceConfig, collate_sequences
from tests.helpers import (
    assert_finite_losses,
    assert_learner_parameter_devices,
    assert_tree_close,
    make_dqn_config,
)


def _model_config(*, atoms=5, dueling=True, noisy=False):
    return {
        "fcnet_hiddens": [8],
        "head_fcnet_hiddens": [8],
        "num_atoms": atoms,
        "double_q": True,
        "dueling": dueling,
        "v_min": -4.0,
        "v_max": 4.0,
        "epsilon": 0.0,
        "noisy": noisy,
        "std_init": 0.5,
    }


def _vector_module(module_class=ActionMaskedDQNTorchRLModule, **options):
    module = module_class(
        observation_space=gym.spaces.Box(-5.0, 5.0, (2,), dtype=np.float32),
        action_space=gym.spaces.Discrete(3),
        model_config=_model_config(**options),
    )
    module.make_target_networks()
    return module


def _graph_space():
    return gym.spaces.Dict(
        {
            "node_features": gym.spaces.Box(-5.0, 5.0, (3, 2), dtype=np.float32),
            "edge_index": gym.spaces.Box(0, 2, (2, 2), dtype=np.int64),
            "node_count": gym.spaces.Box(1, 3, (), dtype=np.int64),
            "edge_count": gym.spaces.Box(0, 2, (), dtype=np.int64),
            "controlled_node": gym.spaces.Box(0, 2, (), dtype=np.int64),
            "action_mask": gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
        }
    )


def _graph(index, *, terminal=False):
    count = index % 3 + 1
    mask = np.ones(3, dtype=np.float32)
    mask[(index + 1) % 3] = 0
    if terminal:
        mask[:] = 0
    return {
        "node_features": np.arange(count * 2, dtype=np.float32).reshape(-1, 2)
        / 10 + index / 20,
        "edge_index": np.asarray([range(count - 1), range(1, count)], dtype=np.int64),
        "controlled_node": 0,
        "action_mask": mask,
    }


def _episode():
    rows = []
    world_t = 0
    for index, duration in enumerate((1, 2, 1, 3, 1, 2)):
        rewards = tuple((index - tick) / 5 for tick in range(duration))
        rows.append(
            MultiModuleTransition(
                env_t=index, agent_t=index, agent_id="agent",
                module_id=DEFAULT_MODULE_ID,
                data={
                    "obs": _graph(index),
                    "new_obs": _graph(index + 1, terminal=index == 5),
                    "actions": np.int64(index % 3),
                    "rewards": sum(rewards),
                    "terminateds": index == 5,
                    "truncateds": False,
                    "world_t": world_t,
                    "duration": duration,
                    "raw_rewards": rewards,
                    "state_reset": index == 0,
                    "subtask_terminated": False,
                },
            )
        )
        world_t += duration
    codec = RawMultiModuleEpisodeCodec()
    payload = codec.encode(rows)
    return EpisodeEnvelope(
        episode_id="rainbow/runner/0/0", schema_version=codec.schema_version,
        producer_member_id="rainbow", runner_id="runner", runner_generation=0,
        local_episode_seq=0, behavior_versions={DEFAULT_MODULE_ID: 0},
        env_steps=payload.env_steps, agent_steps=payload.transition_count,
        terminated=True, truncated=False, estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def test_c51_dueling_uses_per_atom_values_and_centers_over_actions():
    torch.manual_seed(18)
    module = _vector_module()
    batch = {Columns.OBS: torch.tensor([[0.1, 0.3], [0.2, -0.4]])}
    encoded = module.encoder(batch)[ENCODER_OUT]
    advantages = module.af(encoded).reshape(2, 3, 5)
    values = module.vf(encoded)
    assert values.shape == (2, 5)
    expected = values[:, None, :] + advantages - advantages.mean(1, keepdim=True)
    actual = module.compute_q_values(batch)
    torch.testing.assert_close(actual[QF_LOGITS], expected)
    torch.testing.assert_close(actual[QF_PROBS], expected.softmax(-1))
    torch.testing.assert_close(actual[QF_PROBS].sum(-1), torch.ones(2, 3))
    target = torch.tensor([0, 4])
    torch.nn.functional.cross_entropy(actual[QF_LOGITS][:, 0], target).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in module.vf.parameters()
    )


def test_nondueling_c51_heads_match_unmodified_pinned_module():
    stock = _vector_module(DefaultDQNTorchRLModule, dueling=False)
    actual = _vector_module(dueling=False)
    actual.set_state(stock.get_state())
    batch = {Columns.OBS: torch.tensor([[0.1, 0.3], [0.2, -0.4]])}
    expected = stock.compute_q_values(batch)
    result = actual.compute_q_values(batch)
    for key in (ATOMS, QF_LOGITS, QF_PROBS, QF_PREDS):
        torch.testing.assert_close(result[key], expected[key])


def test_noisy_primitive_reset_eval_gradients_and_autograd_lifetime():
    torch.manual_seed(19)
    layer = ControlledNoisyLayer(3, 2, 0.5)
    observations = torch.ones(4, 3)
    layer.noise_enabled = True
    layer.reset_noise()
    first = layer(observations)
    torch.testing.assert_close(first, layer(observations))
    layer.reset_noise()
    second = layer(observations)
    assert not torch.allclose(first, second)
    # Noise replacement must not invalidate tensors saved by the first graph.
    (first.square().sum() + second.square().sum()).backward()
    assert torch.count_nonzero(layer.sigma_w.grad)
    assert torch.count_nonzero(layer.sigma_b.grad)
    layer.noise_enabled = False
    rng = torch.get_rng_state().clone()
    torch.testing.assert_close(layer(observations), observations @ layer.w + layer.b)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert not any("epsilon" in name for name in layer.state_dict())


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_noisy_exploration_uses_masks_and_keeps_evaluation_deterministic(epsilon):
    torch.manual_seed(20)
    module = GraphDQNTorchRLModule(
        observation_space=_graph_space(), action_space=gym.spaces.Discrete(3),
        model_config={**_model_config(noisy=True), "epsilon": epsilon},
    )
    with torch.no_grad():
        for layer in module._noisy_online_layers:
            layer.w.zero_()
            layer.b.zero_()
            if epsilon == 1.0:
                layer.sigma_w.zero_()
                layer.sigma_b.zero_()
    packed = pack_graph_observations([_graph(0)])
    batch = {
        Columns.OBS: {key: torch.as_tensor(value) for key, value in packed.items()}
    }
    actions = {
        module.forward_exploration(batch, t=step)[Columns.ACTIONS].item()
        for step in range(48)
    }
    assert actions == {0, 2}
    rng = torch.get_rng_state().clone()
    evaluations = [
        module.forward_inference(batch)[Columns.ACTIONS].item() for _ in range(4)
    ]
    assert evaluations == [0] * 4
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert not any("epsilon" in key for key in module.get_state(inference_only=True))


def test_noisy_train_resets_once_and_shares_online_noise_for_double_q(monkeypatch):
    module = _vector_module(atoms=1, noisy=True)
    counts = dict.fromkeys(module._noisy_online_layers + module._noisy_target_layers, 0)
    for layer in counts:
        original = layer.reset_noise

        def reset(current=layer, reset_method=original):
            counts[current] += 1
            reset_method()

        monkeypatch.setattr(layer, "reset_noise", reset)
    observations = torch.tensor([[0.1, 0.3], [0.2, -0.4]])
    output = module.forward_train(
        {
            Columns.OBS: observations, Columns.NEXT_OBS: observations,
            Columns.TERMINATEDS: torch.zeros(2, dtype=torch.bool),
        }
    )
    assert set(counts.values()) == {1}
    torch.testing.assert_close(output[QF_PREDS], output[QF_NEXT_PREDS])
    assert not torch.equal(
        module._noisy_online_layers[0].epsilon_in,
        module._noisy_target_layers[0].epsilon_in,
    )


def _projection_oracle(batch, output, gamma, num_atoms):
    """Independent scalar C51 projection, including terminal exact-bin mass."""
    atoms = output[ATOMS].detach().cpu().numpy()
    values = output[QF_NEXT_PREDS].detach().cpu().numpy()
    probabilities = output[QF_TARGET_NEXT_PROBS].detach().cpu().numpy()
    selected = values.argmax(-1)
    result = np.zeros((len(selected), num_atoms), dtype=np.float64)
    spacing = atoms[1] - atoms[0]
    for row, action in enumerate(selected):
        discount = gamma ** int(batch["n_step"][row])
        if bool(batch[Columns.TERMINATEDS][row]):
            discount = 0.0
        reward = float(batch[Columns.REWARDS][row])
        for index, atom in enumerate(atoms):
            projected = min(
                float(atoms[-1]), max(float(atoms[0]), reward + discount * atom)
            )
            position = (projected - atoms[0]) / spacing
            low, high = int(np.floor(position)), int(np.ceil(position))
            mass = probabilities[row, action, index]
            if low == high:
                result[row, low] += mass
            else:
                result[row, low] += mass * (high - position)
                result[row, high] += mass * (position - low)
    return torch.as_tensor(result, device=output[QF_LOGITS].device, dtype=torch.float32)


def test_c51_priority_is_unweighted_cross_entropy_of_projected_target(monkeypatch):
    config = make_dqn_config().training(
        num_atoms=5, dueling=True, double_q=True, n_step=3, v_min=-4.0, v_max=4.0
    )
    spaces = {
        DEFAULT_MODULE_ID: (
            gym.spaces.Box(-5, 5, (2,), dtype=np.float32), gym.spaces.Discrete(3)
        )
    }
    with DQNLearnerAdapter(
        config, spaces=spaces, member_id="ce", publication_interval_updates=1
    ) as adapter:
        module = adapter._learner_group._learner.module[DEFAULT_MODULE_ID]
        forward = module._forward_train
        oracle = {}

        def capture(batch):
            output = forward(batch)
            projected = _projection_oracle(batch, output, config.gamma, 5)
            torch.testing.assert_close(projected.sum(-1), torch.ones(3))
            logits = output[QF_LOGITS][torch.arange(3), batch[Columns.ACTIONS].long()]
            signal = -(projected * logits.log_softmax(-1)).sum(-1)
            oracle["signal"] = signal.detach().numpy()
            oracle["loss"] = (batch["weights"] * signal).mean().item()
            return output

        monkeypatch.setattr(module, "_forward_train", capture)
        batch = {
            Columns.OBS: np.asarray(
                [[0.1, 0.3], [0.2, -0.4], [0.5, 0.6]], dtype=np.float32
            ),
            Columns.NEXT_OBS: np.asarray(
                [[0.2, 0.4], [0.3, -0.5], [0.6, 0.7]], dtype=np.float32
            ),
            Columns.ACTIONS: np.asarray([0, 2, 1], dtype=np.int64),
            Columns.REWARDS: np.asarray([0.0, 1.3, -3.0], dtype=np.float32),
            Columns.TERMINATEDS: np.asarray([True, False, False]),
            Columns.TRUNCATEDS: np.asarray([False, False, True]),
            "n_step": np.asarray([1, 5, 2], dtype=np.int64),
            "weights": np.asarray([0.2, 0.7, 1.0], dtype=np.float32),
        }
        result = adapter.update(batch, sampled_env_steps=8, return_priority_errors=True)
        np.testing.assert_allclose(
            result.priority_errors[DEFAULT_MODULE_ID], oracle["signal"],
            rtol=1e-6, atol=1e-7,
        )
        logged = result.learner_results[0][DEFAULT_MODULE_ID]["qf_loss"]
        actual_loss = logged.peek() if hasattr(logged, "peek") else logged
        assert float(actual_loss) == pytest.approx(oracle["loss"], rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("dueling, atoms", [(True, 1), (False, 5), (True, 5)])
def test_legacy_checkpoint_reject_is_limited_to_scalar_c51_value_head(dueling, atoms):
    config = make_dqn_config().training(dueling=dueling, num_atoms=atoms)
    spaces = {
        DEFAULT_MODULE_ID: (
            gym.spaces.Box(-5, 5, (2,), dtype=np.float32), gym.spaces.Discrete(3)
        )
    }
    with DQNLearnerAdapter(
        config, spaces=spaces, member_id="legacy", publication_interval_updates=1
    ) as adapter:
        before = copy.deepcopy(adapter.get_state())
        legacy = copy.deepcopy(before)
        legacy.pop("c51_dueling_head_version", None)
        legacy["config_contract"].pop("noisy")
        legacy["config_contract"].pop("sigma0")
        if dueling and atoms > 1:
            legacy_module = legacy["learner_group"]["learner"][COMPONENT_RL_MODULE][
                DEFAULT_MODULE_ID
            ]
            for name, value in tuple(legacy_module.items()):
                if name.startswith(("vf.", "_target_vf.")) and value.shape[0] == atoms:
                    legacy_module[name] = value[:1].copy()
            with pytest.raises(ValueError, match="scalar value head"):
                adapter.set_state(legacy)
        else:
            adapter.set_state(legacy)
        assert_tree_close(adapter.get_state(), before)


def _update_priorities(sampler, sample, errors):
    offset = 0
    priorities = []
    for window in sample.windows:
        priorities.append(np.max(np.abs(errors[offset:offset + window.valid_count])))
        offset += window.valid_count
    assert offset == len(errors)
    sampler.update_priorities(sample.handles, priorities)


@pytest.mark.parametrize(
    "recurrent", [False, True], ids=["rainbow-graph", "recurrent-graph"]
)
@pytest.mark.parametrize(
    "gpus",
    [pytest.param(0, id="cpu"), pytest.param(1, marks=pytest.mark.gpu, id="gpu")],
)
def test_rainbow_per_next_sample_and_update_after_restore(recurrent, gpus, request):
    if gpus and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this container")
    codec = RawMultiModuleEpisodeCodec()
    episode = _episode()
    immutable = pickle.dumps(episode)
    store = EpisodeStore(
        codec, capacity_transitions=100, capacity_bytes=1_000_000, journal_capacity=10
    )
    store.commit_episode(episode)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    replay.wait_for_idle(timeout=5)
    sequence = SequenceConfig(learning_length=2, burn_in=None if recurrent else 0)
    sampler = PrioritizedSequenceSampler(
        replay, DEFAULT_MODULE_ID, sequence_config=sequence, n_step=3
    )
    rng = random.Random(41)
    config = make_dqn_config().training(
        double_q=True, dueling=True, num_atoms=1 if recurrent else 5,
        n_step=3, noisy=True, sigma0=0.5, epsilon=0.0,
        v_min=-4.0, v_max=4.0,
    )
    config.learners(num_gpus_per_learner=gpus)
    config.rl_module(
        model_config={
            "fcnet_hiddens": [8], "head_fcnet_hiddens": [8], "recurrent_hidden_dim": 7
        },
        rl_module_spec=RLModuleSpec(
            module_class=RecurrentDQNTorchRLModule
            if recurrent else GraphDQNTorchRLModule
        ),
    )
    prepared = prepare_dqn_config(config)
    assert prepared.rl_module_spec.model_config["noisy"] is True
    spaces = {DEFAULT_MODULE_ID: (_graph_space(), gym.spaces.Discrete(3))}
    adapter = DQNLearnerAdapter(
        config, spaces=spaces, member_id="rainbow",
        publication_interval_updates=1, validate_startup=True,
    )
    restored = None

    def batch_for(sample):
        batch = collate_sequences(
            sample.windows, gamma=config.gamma, n_step=3, include_history=recurrent
        )
        batch["weights"] = sample.weights
        return batch

    try:
        devices = assert_learner_parameter_devices(
            adapter._learner_group, "cuda" if gpus else "cpu"
        )
        request.node.user_properties.append(("parameter_devices", json.dumps(devices)))
        initial = copy.deepcopy(adapter.get_state())
        sample = sampler.sample(8, rng=rng)
        update = adapter.update(
            batch_for(sample), sampled_env_steps=8, return_priority_errors=True
        )
        assert update.performed
        assert_finite_losses(update.learner_results)
        errors = update.priority_errors[DEFAULT_MODULE_ID]
        _update_priorities(sampler, sample, errors)
        learner_checkpoint = copy.deepcopy(adapter.get_state())
        per_checkpoint = copy.deepcopy(sampler.get_state())
        rng_checkpoint = rng.getstate()
        before = initial["learner_group"]["learner"][COMPONENT_RL_MODULE][
            DEFAULT_MODULE_ID
        ]
        after = learner_checkpoint["learner_group"]["learner"][COMPONENT_RL_MODULE][
            DEFAULT_MODULE_ID
        ]
        for head in ("af.", "vf."):
            assert any(
                not np.array_equal(value, after[key])
                for key, value in before.items()
                if key.startswith(head) and "sigma_" in key
            )
        assert not any(
            "epsilon" in key
            for key in adapter.get_inference_state()[DEFAULT_MODULE_ID]
        )
        expected_sample = sampler.sample(8, rng=rng)
        expected_batch = batch_for(expected_sample)
        expected = adapter.update(
            expected_batch, sampled_env_steps=16, return_priority_errors=True
        )
        restored = DQNLearnerAdapter(
            config, spaces=spaces, member_id="rainbow",
            publication_interval_updates=1, validate_startup=True,
        )
        restored.set_state(copy.deepcopy(learner_checkpoint))
        restored_devices = assert_learner_parameter_devices(
            restored._learner_group, "cuda" if gpus else "cpu"
        )
        request.node.user_properties.append(
            ("restored_parameter_devices", json.dumps(restored_devices))
        )
        restored_sampler = PrioritizedSequenceSampler(
            replay, DEFAULT_MODULE_ID, sequence_config=sequence, n_step=3
        )
        restored_sampler.set_state(copy.deepcopy(per_checkpoint))
        restored_rng = random.Random()
        restored_rng.setstate(rng_checkpoint)
        actual_sample = restored_sampler.sample(8, rng=restored_rng)
        assert actual_sample.handles == expected_sample.handles
        np.testing.assert_array_equal(
            actual_sample.probabilities, expected_sample.probabilities
        )
        np.testing.assert_array_equal(actual_sample.weights, expected_sample.weights)
        actual_batch = batch_for(actual_sample)
        assert_tree_close(actual_batch, expected_batch)
        actual = restored.update(
            actual_batch, sampled_env_steps=16, return_priority_errors=True
        )
        assert_finite_losses(actual.learner_results)
        np.testing.assert_allclose(
            actual.priority_errors[DEFAULT_MODULE_ID],
            expected.priority_errors[DEFAULT_MODULE_ID], rtol=1e-6, atol=1e-7,
        )
        expected_state = adapter.get_state()["learner_group"]["learner"]
        actual_state = restored.get_state()["learner_group"]["learner"]
        for key in (COMPONENT_RL_MODULE, COMPONENT_OPTIMIZER):
            assert_tree_close(actual_state[key], expected_state[key])
        assert pickle.dumps(episode) == immutable
    finally:
        if restored is not None:
            restored.close()
        adapter.close()
        replay.close()
