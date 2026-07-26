"""Run the Phase 10 shared-policy ego-GNN pipeline on CPU."""

from __future__ import annotations

import argparse
import json
import random

from rllib_async.examples import (
    DEFAULT_GRAPH_AGENT_COUNT,
    GRAPH_EDGE_FEATURE_DIM,
    GRAPH_NODE_FEATURE_DIM,
    SHARED_GNN_MODULE_ID,
    build_shared_gnn_sac_config,
    shared_gnn_module_spaces,
)
from rllib_async.gnn import GraphBatchCollator, GraphEpisodeCodec
from rllib_async.learner import SACLearnerAdapter
from rllib_async.replay import EpisodeStore, FastReplay
from rllib_async.rollout import MultiModuleEpisodeRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    config = build_shared_gnn_sac_config(seed=args.seed)
    codec = GraphEpisodeCodec(
        node_feature_dim=GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=GRAPH_EDGE_FEATURE_DIM,
    )
    store = EpisodeStore(
        codec,
        capacity_transitions=20_000,
        capacity_bytes=100_000_000,
        journal_capacity=1_024,
        store_generation="shared-gnn-example",
    )
    fast_replay = FastReplay(codec)
    adapter = SACLearnerAdapter(
        config,
        spaces=shared_gnn_module_spaces(DEFAULT_GRAPH_AGENT_COUNT),
        member_id="shared-gnn-member",
        publication_interval_updates=1,
    )
    runner = MultiModuleEpisodeRunner(
        config,
        codec,
        member_id="shared-gnn-member",
        runner_id="runner-0",
        runner_generation=0,
        max_episode_steps=8,
        module_ids=(SHARED_GNN_MODULE_ID,),
        initial_weights=adapter.get_published_weights(),
        worker_index=0,
    )
    collator = GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID)
    rng = random.Random(args.seed)
    env_steps = 0
    agent_steps = 0
    latest_weights = adapter.get_published_weights()
    try:
        fast_replay.load_snapshot(store.get_snapshot())
        for _ in range(args.episodes):
            result = runner.collect_episode(latest_weights, explore=True)
            store.commit_episode(result.episode)
            env_steps += result.episode.env_steps
            agent_steps += result.episode.agent_steps
            assert fast_replay.cursor is not None
            fast_replay.apply_delta(
                store.get_delta(
                    fast_replay.cursor,
                    max_bytes=10_000_000,
                )
            )
            fast_replay.wait_for_idle(timeout=5)
            sampled = fast_replay.sample_module(
                SHARED_GNN_MODULE_ID,
                8,
                rng=rng,
            )
            update = adapter.update_modules(
                collator.collate(sampled),
                sampled_env_steps=env_steps,
                sampled_agent_steps=agent_steps,
            )
            if update.published_weights is not None:
                latest_weights = update.published_weights

        print(
            json.dumps(
                {
                    "agents": DEFAULT_GRAPH_AGENT_COUNT,
                    "episodes": args.episodes,
                    "env_steps": env_steps,
                    "agent_steps": agent_steps,
                    "learner_updates": adapter.learner_updates,
                    "module_transition_counts": dict(
                        fast_replay.module_transition_counts
                    ),
                    "module_versions": dict(
                        adapter.get_published_weights().module_versions
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        runner.close()
        adapter.close()
        fast_replay.close()


if __name__ == "__main__":
    main()
