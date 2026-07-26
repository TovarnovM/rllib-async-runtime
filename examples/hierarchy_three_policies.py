"""Run the Phase 9 manager + two workers hierarchy pipeline on CPU."""

from __future__ import annotations

import argparse
import json
import random

from rllib_async.examples import (
    HIERARCHY_MODULE_IDS,
    build_hierarchy_sac_config,
    hierarchy_module_spaces,
)
from rllib_async.learner import SACLearnerAdapter
from rllib_async.protocols import MultiModuleEpisodeCodec
from rllib_async.replay import (
    EpisodeStore,
    FastReplay,
    MultiModuleBatchCollator,
)
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
    config = build_hierarchy_sac_config(
        episode_length=12,
        manager_period=3,
        seed=args.seed,
    )
    codec = MultiModuleEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=20_000,
        capacity_bytes=100_000_000,
        journal_capacity=1_024,
        store_generation="hierarchy-example",
    )
    fast_replay = FastReplay(codec)
    adapter = SACLearnerAdapter(
        config,
        spaces=hierarchy_module_spaces(),
        member_id="hierarchy-member",
        publication_interval_updates=1,
    )
    runner = MultiModuleEpisodeRunner(
        config,
        codec,
        member_id="hierarchy-member",
        runner_id="runner-0",
        runner_generation=0,
        max_episode_steps=12,
        module_ids=HIERARCHY_MODULE_IDS,
        initial_weights=adapter.get_published_weights(),
        worker_index=0,
    )
    collator = MultiModuleBatchCollator(
        expected_module_ids=HIERARCHY_MODULE_IDS,
    )
    rngs = {
        module_id: random.Random(args.seed + index)
        for index, module_id in enumerate(HIERARCHY_MODULE_IDS)
    }
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
            if set(fast_replay.module_ids) != set(HIERARCHY_MODULE_IDS):
                continue
            sampled = [
                transition
                for module_id in HIERARCHY_MODULE_IDS
                for transition in fast_replay.sample_module(
                    module_id,
                    8,
                    rng=rngs[module_id],
                )
            ]
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
