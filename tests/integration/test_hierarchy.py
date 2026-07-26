from __future__ import annotations

import copy
import random
from collections import Counter

import pytest

from rllib_async.examples import (
    HIERARCHY_MODULE_IDS,
    MANAGER_MODULE_ID,
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
from rllib_async.replay.checkpoint import (
    read_replay_checkpoint,
    write_replay_checkpoint,
)
from rllib_async.rollout import MultiModuleEpisodeRunner


@pytest.mark.integration
def test_hierarchy_runner_preserves_sparse_turns_and_module_versions() -> None:
    config = build_hierarchy_sac_config(
        episode_length=8,
        manager_period=2,
        seed=20260726,
    )
    codec = MultiModuleEpisodeCodec()
    adapter = SACLearnerAdapter(
        config,
        spaces=hierarchy_module_spaces(),
        member_id="hierarchy-member",
        publication_interval_updates=1,
    )
    runner = None
    try:
        runner = MultiModuleEpisodeRunner(
            config,
            codec,
            member_id="hierarchy-member",
            runner_id="runner-0",
            runner_generation=0,
            max_episode_steps=8,
            module_ids=HIERARCHY_MODULE_IDS,
            initial_weights=adapter.get_published_weights(),
            worker_index=0,
        )

        result = runner.collect_episode(explore=True)
        episode = result.episode
        transitions = [
            codec.get_transition(episode, index)
            for index in range(codec.transition_count(episode))
        ]
        by_env_t = Counter(item.env_t for item in transitions)
        manager_env_t = {
            item.env_t for item in transitions if item.module_id == MANAGER_MODULE_ID
        }
        worker_counts = Counter(
            item.env_t for item in transitions if item.module_id != MANAGER_MODULE_ID
        )

        assert episode.env_steps == 8
        assert episode.agent_steps == 12
        assert set(episode.behavior_versions) == set(HIERARCHY_MODULE_IDS)
        assert set(codec.module_ids(episode)) == set(HIERARCHY_MODULE_IDS)
        assert manager_env_t == {0, 2, 4, 6}
        assert worker_counts == Counter({env_t: 1 for env_t in range(8)})
        assert by_env_t == Counter(
            {env_t: 2 if env_t in manager_env_t else 1 for env_t in range(8)}
        )
        for module_id in HIERARCHY_MODULE_IDS:
            module_transitions = [
                item for item in transitions if item.module_id == module_id
            ]
            agent_ids = {item.agent_id for item in module_transitions}
            assert agent_ids == {module_id}
            assert [item.agent_t for item in module_transitions] == list(
                range(len(module_transitions))
            )
            assert all(
                episode.behavior_versions[module_id] == 0 for _ in module_transitions
            )
    finally:
        if runner is not None:
            runner.close()
        adapter.close()


@pytest.mark.integration
def test_hierarchy_long_smoke_survives_delta_and_checkpoint_restore(
    tmp_path,
) -> None:
    config = build_hierarchy_sac_config(
        episode_length=8,
        manager_period=2,
        seed=20260726,
    )
    codec = MultiModuleEpisodeCodec()
    store = EpisodeStore(
        codec,
        capacity_transitions=4_096,
        capacity_bytes=20_000_000,
        journal_capacity=256,
        store_generation="hierarchy-smoke",
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
        max_episode_steps=8,
        module_ids=HIERARCHY_MODULE_IDS,
        initial_weights=adapter.get_published_weights(),
        worker_index=0,
    )
    restored_fast = restored_adapter = restored_runner = None
    sampled_env_steps = 0
    sampled_agent_steps = 0
    rngs = {
        module_id: random.Random(20260726 + index)
        for index, module_id in enumerate(HIERARCHY_MODULE_IDS)
    }
    collator = MultiModuleBatchCollator(
        expected_module_ids=HIERARCHY_MODULE_IDS,
    )
    try:
        fast_replay.load_snapshot(store.get_snapshot())
        latest_weights = adapter.get_published_weights()
        for _ in range(12):
            result = runner.collect_episode(latest_weights, explore=True)
            acknowledgement = store.commit_episode(result.episode)
            assert acknowledgement.committed
            sampled_env_steps += result.episode.env_steps
            sampled_agent_steps += result.episode.agent_steps
            assert fast_replay.cursor is not None
            delta = store.get_delta(
                fast_replay.cursor,
                max_bytes=1_000_000,
            )
            fast_replay.apply_delta(delta)
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
            batches = collator.collate(sampled)
            update = adapter.update_modules(
                batches,
                sampled_env_steps=sampled_env_steps,
                sampled_agent_steps=sampled_agent_steps,
            )
            assert update.performed
            if update.published_weights is not None:
                latest_weights = update.published_weights

        assert adapter.learner_updates >= 8
        assert set(fast_replay.module_ids) == set(HIERARCHY_MODULE_IDS)
        assert all(count > 0 for _, count in fast_replay.module_transition_counts)
        checkpoint_path = tmp_path / "hierarchy-replay.bin"
        replay_checkpoint = write_replay_checkpoint(
            checkpoint_path,
            store.export_state(),
        )
        learner_state = copy.deepcopy(adapter.get_state())
        saved_cursor = fast_replay.cursor
        saved_updates = adapter.learner_updates
        assert replay_checkpoint.cursor == saved_cursor

        runner.close()
        adapter.close()
        fast_replay.close()
        restored_store = EpisodeStore.from_state(
            codec,
            read_replay_checkpoint(checkpoint_path),
        )
        restored_fast = FastReplay(codec)
        restored_fast.load_snapshot(restored_store.get_snapshot())
        restored_adapter = SACLearnerAdapter(
            config,
            spaces=hierarchy_module_spaces(),
            member_id="hierarchy-member",
            publication_interval_updates=1,
        )
        restored_adapter.set_state(learner_state)
        restored_runner = MultiModuleEpisodeRunner(
            config,
            codec,
            member_id="hierarchy-member",
            runner_id="runner-0",
            runner_generation=1,
            max_episode_steps=8,
            module_ids=HIERARCHY_MODULE_IDS,
            initial_weights=restored_adapter.get_published_weights(),
            worker_index=1,
        )

        assert restored_store.cursor == saved_cursor
        assert restored_fast.cursor == saved_cursor
        assert restored_adapter.learner_updates == saved_updates
        continued = restored_runner.collect_episode(explore=True)
        assert continued.episode.runner_generation == 1
        acknowledgement = restored_store.commit_episode(continued.episode)
        assert acknowledgement.committed
        assert restored_fast.cursor is not None
        restored_fast.apply_delta(
            restored_store.get_delta(
                restored_fast.cursor,
                max_bytes=1_000_000,
            )
        )
        restored_fast.wait_for_idle(timeout=5)
        assert restored_fast.cursor.mutation_seq == saved_cursor.mutation_seq + 1
    finally:
        runner.close()
        adapter.close()
        fast_replay.close()
        if restored_runner is not None:
            restored_runner.close()
        if restored_adapter is not None:
            restored_adapter.close()
        if restored_fast is not None:
            restored_fast.close()
