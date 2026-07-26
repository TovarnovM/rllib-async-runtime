from __future__ import annotations

import copy
import random

import numpy as np
import pytest
from ray.rllib.core.columns import Columns

from rllib_async.examples import (
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    SHARED_GNN_MODULE_ID,
    build_shared_gnn_sac_config,
    graph_agent_ids,
    shared_gnn_module_spaces,
)
from rllib_async.gnn import GraphBatchCollator, GraphEpisodeCodec
from rllib_async.learner import SACLearnerAdapter
from rllib_async.replay import EpisodeStore, FastReplay
from rllib_async.replay.checkpoint import (
    read_replay_checkpoint,
    write_replay_checkpoint,
)
from rllib_async.rollout import MultiModuleEpisodeRunner


@pytest.mark.integration
def test_shared_gnn_runner_uses_one_module_for_all_variable_graph_agents() -> None:
    config = build_shared_gnn_sac_config(
        agent_count=4,
        episode_length=4,
        hidden_dim=16,
        message_layers=2,
        seed=20260726,
    )
    codec = GraphEpisodeCodec(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
    )
    adapter = SACLearnerAdapter(
        config,
        spaces=shared_gnn_module_spaces(4),
        member_id="graph-member",
        publication_interval_updates=1,
    )
    runner = None
    try:
        runner = MultiModuleEpisodeRunner(
            config,
            codec,
            member_id="graph-member",
            runner_id="runner-0",
            runner_generation=0,
            max_episode_steps=4,
            module_ids=(SHARED_GNN_MODULE_ID,),
            initial_weights=adapter.get_published_weights(),
            worker_index=0,
        )

        episode = runner.collect_episode(explore=True).episode
        transitions = [
            codec.get_module_transition(
                episode,
                SHARED_GNN_MODULE_ID,
                index,
            )
            for index in range(
                codec.module_transition_count(
                    episode,
                    SHARED_GNN_MODULE_ID,
                )
            )
        ]

        assert episode.env_steps == 4
        assert episode.agent_steps == 16
        assert tuple(episode.behavior_versions) == (SHARED_GNN_MODULE_ID,)
        assert codec.module_ids(episode) == (SHARED_GNN_MODULE_ID,)
        assert {item.agent_id for item in transitions} == set(graph_agent_ids(4))
        assert {item.module_id for item in transitions} == {SHARED_GNN_MODULE_ID}
        assert {
            item.data[Columns.OBS]["node_features"].shape[0] for item in transitions
        } == {1, 2, 3, 4}
    finally:
        if runner is not None:
            runner.close()
        adapter.close()


@pytest.mark.integration
def test_shared_gnn_long_smoke_updates_and_restores(
    tmp_path,
) -> None:
    config = build_shared_gnn_sac_config(
        agent_count=4,
        episode_length=4,
        hidden_dim=16,
        message_layers=2,
        seed=20260726,
    )
    codec = GraphEpisodeCodec(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
    )
    store = EpisodeStore(
        codec,
        capacity_transitions=4_096,
        capacity_bytes=30_000_000,
        journal_capacity=256,
        store_generation="graph-smoke",
    )
    fast = FastReplay(codec)
    adapter = SACLearnerAdapter(
        config,
        spaces=shared_gnn_module_spaces(4),
        member_id="graph-member",
        publication_interval_updates=1,
    )
    runner = MultiModuleEpisodeRunner(
        config,
        codec,
        member_id="graph-member",
        runner_id="runner-0",
        runner_generation=0,
        max_episode_steps=4,
        module_ids=(SHARED_GNN_MODULE_ID,),
        initial_weights=adapter.get_published_weights(),
        worker_index=0,
    )
    collator = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID)
    rng = random.Random(20260726)
    restored_fast = restored_adapter = restored_runner = None
    sampled_env_steps = 0
    sampled_agent_steps = 0
    initial_encoder_weight = copy.deepcopy(
        adapter.get_published_weights().state[SHARED_GNN_MODULE_ID][
            "pi_encoder.node_projection.weight"
        ]
    )
    try:
        fast.load_snapshot(store.get_snapshot())
        latest_weights = adapter.get_published_weights()
        for _ in range(8):
            result = runner.collect_episode(latest_weights, explore=True)
            assert store.commit_episode(result.episode).committed
            sampled_env_steps += result.episode.env_steps
            sampled_agent_steps += result.episode.agent_steps
            assert fast.cursor is not None
            fast.apply_delta(
                store.get_delta(
                    fast.cursor,
                    max_bytes=5_000_000,
                )
            )
            fast.wait_for_idle(timeout=5)
            sampled = fast.sample_module(
                SHARED_GNN_MODULE_ID,
                8,
                rng=rng,
            )
            update = adapter.update_modules(
                collator.collate(sampled),
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
            )
            assert update.performed
            if update.published_weights is not None:
                latest_weights = update.published_weights

        final_encoder_weight = adapter.get_published_weights().state[
            SHARED_GNN_MODULE_ID
        ]["pi_encoder.node_projection.weight"]
        assert not np.array_equal(initial_encoder_weight, final_encoder_weight)
        assert adapter.learner_updates == 8
        assert fast.module_transition_counts == ((SHARED_GNN_MODULE_ID, 128),)

        checkpoint_path = tmp_path / "graph-replay.bin"
        replay_checkpoint = write_replay_checkpoint(
            checkpoint_path,
            store.export_state(),
        )
        learner_state = copy.deepcopy(adapter.get_state())
        saved_updates = adapter.learner_updates
        saved_cursor = fast.cursor
        assert replay_checkpoint.cursor == saved_cursor

        runner.close()
        adapter.close()
        fast.close()
        restored_store = EpisodeStore.from_state(
            codec,
            read_replay_checkpoint(checkpoint_path),
        )
        restored_fast = FastReplay(codec)
        restored_fast.load_snapshot(restored_store.get_snapshot())
        restored_adapter = SACLearnerAdapter(
            config,
            spaces=shared_gnn_module_spaces(4),
            member_id="graph-member",
            publication_interval_updates=1,
        )
        restored_adapter.set_state(learner_state)
        restored_runner = MultiModuleEpisodeRunner(
            config,
            codec,
            member_id="graph-member",
            runner_id="runner-0",
            runner_generation=1,
            max_episode_steps=4,
            module_ids=(SHARED_GNN_MODULE_ID,),
            initial_weights=restored_adapter.get_published_weights(),
            worker_index=1,
        )

        assert restored_store.cursor == saved_cursor
        assert restored_fast.cursor == saved_cursor
        assert restored_adapter.learner_updates == saved_updates
        continued = restored_runner.collect_episode(explore=True)
        assert continued.episode.runner_generation == 1
        assert restored_store.commit_episode(continued.episode).committed
        assert restored_fast.cursor is not None
        restored_fast.apply_delta(
            restored_store.get_delta(
                restored_fast.cursor,
                max_bytes=5_000_000,
            )
        )
        restored_fast.wait_for_idle(timeout=5)
        restored_update = restored_adapter.update_modules(
            collator.collate(
                restored_fast.sample_module(
                    SHARED_GNN_MODULE_ID,
                    8,
                    rng=random.Random(20260727),
                )
            ),
            sampled_env_steps=sampled_env_steps + continued.episode.env_steps,
            sampled_agent_steps=(sampled_agent_steps + continued.episode.agent_steps),
        )
        assert restored_update.performed
        assert restored_adapter.learner_updates == saved_updates + 1
    finally:
        runner.close()
        adapter.close()
        fast.close()
        if restored_runner is not None:
            restored_runner.close()
        if restored_adapter is not None:
            restored_adapter.close()
        if restored_fast is not None:
            restored_fast.close()
