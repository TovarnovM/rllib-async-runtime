from __future__ import annotations

import pickle
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT

from rllib_async.examples import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    SHARED_GNN_MODULE_ID,
    EgoGraphCoordinationEnv,
    build_shared_gnn_sac_config,
    shared_gnn_policy_mapping_fn,
)
from rllib_async.gnn import (
    GraphBatchCollator,
    GraphEncoderConfig,
    GraphEpisodeCodec,
    TorchGraphEncoder,
    normalize_graph_observation,
    validate_packed_graph_batch,
)
from rllib_async.gnn.episodes import (
    ACTION_MASK,
    CONTROLLED_NODE,
    EDGE_FEATURES,
    EDGE_INDEX,
    NODE_FEATURES,
)
from rllib_async.protocols import (
    EpisodeEnvelope,
    EpisodeValidationError,
    FrozenVersions,
    MultiModuleTransition,
)
from rllib_async.replay import (
    BatchCollationError,
    EpisodeStore,
    FastReplay,
    ReferenceFastReplay,
)


def _transition(
    *,
    env_t: int,
    agent_t: int,
    agent_id: str,
    observation: object,
    next_observation: object | None = None,
    action_mask: bool = False,
) -> MultiModuleTransition:
    observation = normalize_graph_observation(observation)
    next_observation = normalize_graph_observation(
        observation if next_observation is None else next_observation
    )
    if action_mask:
        observation[ACTION_MASK] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
        next_observation[ACTION_MASK] = np.asarray(
            [1.0, 1.0, 0.0],
            dtype=np.float32,
        )
    return MultiModuleTransition(
        env_t=env_t,
        agent_t=agent_t,
        agent_id=agent_id,
        module_id=SHARED_GNN_MODULE_ID,
        data={
            Columns.OBS: observation,
            Columns.NEXT_OBS: next_observation,
            Columns.ACTIONS: np.int64(agent_t % 3),
            Columns.REWARDS: float(agent_t),
            Columns.TERMINATEDS: False,
            Columns.TRUNCATEDS: False,
        },
    )


def _episode(
    codec: GraphEpisodeCodec,
    transitions: list[MultiModuleTransition],
    *,
    sequence: int = 0,
    env_steps: int = 1,
) -> EpisodeEnvelope:
    payload = codec.encode(transitions)
    return EpisodeEnvelope(
        episode_id=f"member/runner/0/{sequence}",
        schema_version=codec.schema_version,
        producer_member_id="member",
        runner_id="runner",
        runner_generation=0,
        local_episode_seq=sequence,
        behavior_versions=FrozenVersions({SHARED_GNN_MODULE_ID: 2}),
        env_steps=env_steps,
        agent_steps=len(transitions),
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def test_graph_env_uses_variable_ego_sizes_and_one_policy_mapping() -> None:
    env = EgoGraphCoordinationEnv({"agent_count": 4, "episode_length": 4})

    observations, _ = env.reset(seed=20260726)

    assert {
        int(observation["node_count"]) for observation in observations.values()
    } == {1, 2, 3, 4}
    assert {shared_gnn_policy_mapping_fn(agent_id) for agent_id in observations} == {
        SHARED_GNN_MODULE_ID
    }
    for observation in observations.values():
        assert env.observation_spaces[next(iter(observations))].contains(observation)

    with pytest.raises(ValueError, match="every logical graph agent"):
        env.step({"agent_0": 1})


def test_shared_gnn_config_exposes_target_gpu_without_changing_default() -> None:
    assert build_shared_gnn_sac_config().num_gpus_per_learner == 0
    assert (
        build_shared_gnn_sac_config(
            num_gpus_per_learner=1,
        ).num_gpus_per_learner
        == 1
    )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        build_shared_gnn_sac_config(num_gpus_per_learner=2)


def test_graph_codec_strips_padding_and_reuses_module_replay_views() -> None:
    env = EgoGraphCoordinationEnv({"agent_count": 4, "episode_length": 4})
    observations, _ = env.reset(seed=20260726)
    codec = GraphEpisodeCodec(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
    )
    transitions = [
        _transition(
            env_t=0,
            agent_t=0,
            agent_id=agent_id,
            observation=observation,
        )
        for agent_id, observation in observations.items()
    ]
    episode = _episode(codec, transitions)

    codec.validate(episode)

    assert codec.module_ids(episode) == (SHARED_GNN_MODULE_ID,)
    restored = [
        codec.get_module_transition(episode, SHARED_GNN_MODULE_ID, index)
        for index in range(len(transitions))
    ]
    assert sorted(
        item.data[Columns.OBS][NODE_FEATURES].shape[0] for item in restored
    ) == [1, 2, 3, 4]
    assert all("node_count" not in item.data[Columns.OBS] for item in restored)
    non_graph_payload = super(GraphEpisodeCodec, codec).encode(transitions)
    with pytest.raises(EpisodeValidationError, match="GraphEpisodePayload"):
        codec.validate(replace(episode, payload=non_graph_payload))
    incompatible_codec = GraphEpisodeCodec(
        node_feature_dim=5,
        edge_feature_dim=1,
    )
    assert incompatible_codec.codec_id != codec.codec_id
    with pytest.raises(EpisodeValidationError, match="feature schema"):
        incompatible_codec.validate(episode)

    invalid_observation = dict(observations["agent_0"])
    invalid_observation[NODE_FEATURES] = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(EpisodeValidationError, match="node feature dimension"):
        codec.encode(
            [
                _transition(
                    env_t=0,
                    agent_t=0,
                    agent_id="agent_0",
                    observation=invalid_observation,
                )
            ]
        )

    store = EpisodeStore(
        codec,
        capacity_transitions=100,
        capacity_bytes=1_000_000,
        store_generation="graph-unit",
    )
    store.commit_episode(episode)
    fast = FastReplay(codec)
    reference = ReferenceFastReplay(codec)
    try:
        snapshot = store.get_snapshot()
        fast.load_snapshot(snapshot)
        reference.load_snapshot(snapshot)
        assert fast.module_transition_counts == (
            (SHARED_GNN_MODULE_ID, len(transitions)),
        )
        assert fast.sample_module_coordinates(
            SHARED_GNN_MODULE_ID,
            100,
            rng=random.Random(7),
        ) == reference.sample_module_coordinates(
            SHARED_GNN_MODULE_ID,
            100,
            rng=random.Random(7),
        )
    finally:
        fast.close()


@pytest.mark.parametrize("feature_name", [NODE_FEATURES, EDGE_FEATURES])
def test_graph_codec_rejects_values_that_overflow_float32(
    feature_name: str,
) -> None:
    valid_graph = {
        NODE_FEATURES: np.zeros((1, 1), dtype=np.float64),
        EDGE_INDEX: np.asarray([[0], [0]], dtype=np.int64),
        EDGE_FEATURES: np.zeros((1, 1), dtype=np.float64),
        CONTROLLED_NODE: np.int64(0),
    }
    invalid_graph = {
        **valid_graph,
        feature_name: valid_graph[feature_name].copy(),
    }
    invalid_graph[feature_name][0, 0] = 1e40
    transition = MultiModuleTransition(
        env_t=0,
        agent_t=0,
        agent_id="agent_0",
        module_id=SHARED_GNN_MODULE_ID,
        data={
            Columns.OBS: invalid_graph,
            Columns.NEXT_OBS: valid_graph,
            Columns.ACTIONS: np.int64(0),
            Columns.REWARDS: 0.0,
            Columns.TERMINATEDS: True,
            Columns.TRUNCATEDS: False,
        },
    )
    codec = GraphEpisodeCodec(
        node_feature_dim=1,
        edge_feature_dim=1,
    )

    with pytest.raises(
        EpisodeValidationError,
        match=rf"{feature_name}.*representable as float32",
    ):
        codec.encode([transition])


def test_graph_batch_collates_empty_edges_one_node_and_different_sizes() -> None:
    env = EgoGraphCoordinationEnv({"agent_count": 4, "episode_length": 4})
    observations, _ = env.reset(seed=20260726)
    transitions = [
        _transition(
            env_t=0,
            agent_t=0,
            agent_id=agent_id,
            observation=observations[agent_id],
            action_mask=True,
        )
        for agent_id in ("agent_0", "agent_1", "agent_2")
    ]

    batch = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID).collate(transitions)[
        SHARED_GNN_MODULE_ID
    ]
    graph_batch = batch[Columns.OBS]
    assert isinstance(graph_batch, dict)

    np.testing.assert_array_equal(
        graph_batch["graph_ptr"],
        np.asarray([0, 1, 3, 6], dtype=np.int64),
    )
    assert graph_batch[NODE_FEATURES].shape == (6, 4)
    assert graph_batch[EDGE_INDEX].shape == (2, 6)
    assert graph_batch[EDGE_FEATURES].shape == (6, 1)
    assert graph_batch[CONTROLLED_NODE].tolist() == [0, 1, 3]
    assert graph_batch[ACTION_MASK].shape == (3, 3)
    assert batch[Columns.ACTIONS].shape == (3,)

    mixed = [
        transitions[0],
        _transition(
            env_t=0,
            agent_t=0,
            agent_id="agent_1",
            observation=observations["agent_1"],
            action_mask=False,
        ),
    ]
    with pytest.raises(BatchCollationError, match="action_mask"):
        GraphBatchCollator(
            module_id=SHARED_GNN_MODULE_ID, validate_startup=True
        ).collate(mixed)


def test_graph_startup_checks_scalars_and_retries_until_batch_succeeds(
    monkeypatch,
) -> None:
    import rllib_async.gnn.graph_batch as batch_module

    transition = _transition(
        env_t=0,
        agent_t=0,
        agent_id="agent_0",
        observation={
            NODE_FEATURES: np.zeros((1, 1), dtype=np.float32),
            EDGE_INDEX: np.empty((2, 0), dtype=np.int64),
            CONTROLLED_NODE: 0,
        },
    )
    collator = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID, validate_startup=True)
    invalid_reward = replace(
        transition, data={**transition.data, Columns.REWARDS: "bad"}
    )
    with pytest.raises(BatchCollationError, match="numeric or boolean"):
        collator.collate([invalid_reward])

    invalid_graph = replace(
        transition,
        data={
            **transition.data,
            Columns.NEXT_OBS: {**transition.data[Columns.NEXT_OBS], CONTROLLED_NODE: 1},
        },
    )
    with pytest.raises(BatchCollationError, match="controlled_node"):
        collator.collate([invalid_graph])
    # A failure after scalar validation must not consume the startup check either.
    with pytest.raises(BatchCollationError, match="numeric or boolean"):
        collator.collate([invalid_reward])
    batch = collator.collate([transition])[SHARED_GNN_MODULE_ID]
    np.testing.assert_array_equal(batch[Columns.REWARDS], [0.0])

    scalar_collate = batch_module.FlatBatchCollator.collate

    def unchecked_scalars_only(self, rows):
        assert not self._validate, "scalar validation repeated after startup"
        return scalar_collate(self, rows)

    def unexpected_normalization(value):
        raise AssertionError("graph validation repeated after startup")

    monkeypatch.setattr(
        batch_module.FlatBatchCollator, "collate", unchecked_scalars_only
    )
    monkeypatch.setattr(
        batch_module, "normalize_graph_observation", unexpected_normalization
    )
    for _ in range(2):
        batch = collator.collate([transition])[SHARED_GNN_MODULE_ID]
        np.testing.assert_array_equal(batch[Columns.REWARDS], [0.0])


@pytest.mark.parametrize(
    "terminated,truncated", [(True, False), (False, False), (False, True)]
)
def test_empty_graph_action_mask_is_allowed_only_for_terminal_next_observation(
    terminated: bool, truncated: bool
) -> None:
    from rllib_async.learner.sac_adapter import build_rllib_multi_module_sac_batch

    transition = _transition(
        env_t=0,
        agent_t=0,
        agent_id="agent_0",
        observation={
            NODE_FEATURES: np.zeros((1, 1), dtype=np.float32),
            EDGE_INDEX: np.empty((2, 0), dtype=np.int64),
            CONTROLLED_NODE: 0,
        },
        action_mask=True,
    )
    empty_next = {
        **transition.data[Columns.NEXT_OBS],
        ACTION_MASK: np.zeros(3, dtype=np.float32),
    }
    transition = replace(
        transition,
        data={
            **transition.data,
            Columns.NEXT_OBS: empty_next,
            Columns.TERMINATEDS: terminated,
            Columns.TRUNCATEDS: truncated,
        },
    )
    codec = GraphEpisodeCodec(node_feature_dim=1, action_mask_dim=3)
    collator = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID, validate_startup=True)
    if not terminated:
        with pytest.raises(EpisodeValidationError, match="action_mask"):
            codec.encode([transition])
        with pytest.raises(BatchCollationError, match="action_mask"):
            collator.collate([transition])
        return

    episode = _episode(codec, [transition])
    restored = codec.get_module_transition(episode, SHARED_GNN_MODULE_ID, 0)
    batch = collator.collate([restored])
    learner_batch = build_rllib_multi_module_sac_batch(batch)
    assert learner_batch.policy_batches[SHARED_GNN_MODULE_ID].count == 1
    np.testing.assert_array_equal(
        batch[SHARED_GNN_MODULE_ID][Columns.NEXT_OBS][ACTION_MASK], [[0, 0, 0]]
    )

    # Termination permits an empty target mask, never an empty behavior mask.
    with pytest.raises(EpisodeValidationError, match="action_mask"):
        codec.encode(
            [replace(transition, data={**transition.data, Columns.OBS: empty_next})]
        )
    with pytest.raises(BatchCollationError, match="action_mask"):
        validate_packed_graph_batch(batch[SHARED_GNN_MODULE_ID][Columns.NEXT_OBS])


@pytest.mark.parametrize("decoder", ["get_transition", "get_module_transition"])
def test_public_graph_decode_rejects_malformed_contents(decoder: str) -> None:
    codec = GraphEpisodeCodec(node_feature_dim=1)
    transition = _transition(
        env_t=0,
        agent_t=0,
        agent_id="agent_0",
        observation={
            NODE_FEATURES: np.zeros((1, 1), dtype=np.float32),
            EDGE_INDEX: np.empty((2, 0), dtype=np.int64),
            CONTROLLED_NODE: 0,
        },
    )
    episode = _episode(codec, [transition])
    module_id, transitions = episode.payload.encoded_module_transitions[0]
    data = pickle.loads(transitions[0].encoded_data)
    data[Columns.OBS][CONTROLLED_NODE] = 1
    encoded = replace(transitions[0], encoded_data=pickle.dumps(data, protocol=5))
    payload = replace(
        episode.payload, encoded_module_transitions=((module_id, (encoded,)),)
    )
    malformed = replace(
        episode, payload=payload, estimated_bytes=payload.estimated_bytes
    )
    codec.validate(malformed)  # Valid metadata does not imply valid graph contents.
    args = (malformed, 0) if decoder == "get_transition" else (malformed, module_id, 0)
    with pytest.raises(EpisodeValidationError, match="controlled_node"):
        getattr(codec, decoder)(*args)


def test_packed_graph_batch_rejects_cross_graph_edges() -> None:
    batch = {
        NODE_FEATURES: np.zeros((2, 4), dtype=np.float32),
        EDGE_INDEX: np.asarray([[0], [1]], dtype=np.int64),
        "graph_ptr": np.asarray([0, 1, 2], dtype=np.int64),
        CONTROLLED_NODE: np.asarray([0, 1], dtype=np.int64),
    }

    with pytest.raises(BatchCollationError, match="connect different graphs"):
        validate_packed_graph_batch(batch)

    encoder = TorchGraphEncoder(
        GraphEncoderConfig(
            validate_startup=True,
            node_feature_dim=4,
            hidden_dim=8,
            message_layers=1,
        )
    )
    with pytest.raises(ValueError, match="connect different packed graphs"):
        encoder({Columns.OBS: batch})


def test_graph_encoder_batches_graphs_and_backpropagates() -> None:
    env = EgoGraphCoordinationEnv({"agent_count": 4, "episode_length": 4})
    observations, _ = env.reset(seed=20260726)
    transitions = [
        _transition(
            env_t=0,
            agent_t=0,
            agent_id=agent_id,
            observation=observation,
        )
        for agent_id, observation in observations.items()
    ]
    graph_batch = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID).collate(
        transitions
    )[SHARED_GNN_MODULE_ID][Columns.OBS]
    encoder = TorchGraphEncoder(
        GraphEncoderConfig(
            node_feature_dim=4,
            edge_feature_dim=1,
            hidden_dim=8,
            message_layers=2,
        )
    )

    output = encoder({Columns.OBS: graph_batch})[ENCODER_OUT]
    output.square().sum().backward()

    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()
    assert encoder.node_projection.weight.grad is not None
    assert torch.count_nonzero(encoder.node_projection.weight.grad) > 0


@pytest.mark.parametrize("validate_startup", [False, True])
def test_graph_encoder_scans_contents_only_at_opted_in_startup(
    monkeypatch, validate_startup
) -> None:
    encoder = TorchGraphEncoder(
        GraphEncoderConfig(node_feature_dim=4, validate_startup=validate_startup)
    )
    batch = {
        NODE_FEATURES: np.zeros((2, 4), dtype=np.float32),
        EDGE_INDEX: np.empty((2, 0), dtype=np.int64),
        "graph_ptr": np.asarray([0, 2], dtype=np.int64),
        CONTROLLED_NODE: np.asarray([0], dtype=np.int64),
    }
    calls = []
    original = torch.isfinite

    def finite(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(torch, "isfinite", finite)
    encoder({Columns.OBS: batch})
    assert len(calls) == int(validate_startup)
    encoder({Columns.OBS: batch})
    assert len(calls) == int(validate_startup)


@pytest.mark.parametrize("decoder", ["get_transition", "get_module_transition"])
def test_graph_replay_decode_and_collation_do_not_repeat_normalization(
    monkeypatch,
    decoder: str,
) -> None:
    import rllib_async.gnn.episodes as episodes_module
    import rllib_async.gnn.graph_batch as batch_module

    env = EgoGraphCoordinationEnv({"agent_count": 4, "episode_length": 4})
    observations, _ = env.reset(seed=20260726)
    transition = _transition(
        env_t=0, agent_t=0, agent_id="agent_0", observation=observations["agent_0"]
    )
    codec = GraphEpisodeCodec(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM, edge_feature_dim=GRAPH_EDGE_FEATURE_DIM
    )
    episode = _episode(codec, [transition])

    def unexpected_normalization(value):
        raise AssertionError("trusted replay graph was normalized again")

    monkeypatch.setattr(
        episodes_module, "normalize_graph_observation", unexpected_normalization
    )
    monkeypatch.setattr(
        batch_module, "normalize_graph_observation", unexpected_normalization
    )
    args = (
        (episode, 0)
        if decoder == "get_transition"
        else (episode, SHARED_GNN_MODULE_ID, 0)
    )
    sampled = getattr(codec, decoder)(*args, validate=False)
    batch = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID).collate([sampled])
    assert batch[SHARED_GNN_MODULE_ID][Columns.REWARDS].shape == (1,)
