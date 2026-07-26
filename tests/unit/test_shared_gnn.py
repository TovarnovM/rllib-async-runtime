from __future__ import annotations

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
        GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID).collate(mixed)


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
