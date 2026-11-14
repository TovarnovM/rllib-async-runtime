"""Small, compatible teams shared by population contract/lifecycle tests."""

from rllib_async.examples.temporal_hierarchy import build_temporal_hierarchy_configs
from rllib_async.replay.prioritized import PrioritizedReplayConfig
from rllib_async.runtime import (
    AsyncRuntimeConfig,
    FloatMutation,
    PopulationMemberSpec,
    SimplePBTConfig,
)


def mixed_specs(*, num_gpus=0, seed=300, manager_lr=3e-4):
    members = []
    for index in range(2):
        rollout, modules, _ = build_temporal_hierarchy_configs(
            episode_length=6,
            manager_period=2,
            vector_worker=True,
            seed=seed + index,
        )
        for config in modules.values():
            config.learners(num_gpus_per_learner=num_gpus)
            config.training(
                n_step=2,
                train_batch_size_per_learner=2,
                num_steps_sampled_before_learning_starts=0,
                training_intensity=1.0,
                target_network_update_freq=1,
            )
        modules["manager"].training(
            lr=manager_lr,
            noisy=True,
            num_atoms=5,
            v_min=-10,
            v_max=10,
            dueling=True,
            double_q=True,
            epsilon=0.0,
        )
        settings = AsyncRuntimeConfig(
            member_id=f"team-{index}",
            runner_count=1,
            max_episode_steps=12,
            replay_capacity_transitions=512,
            replay_capacity_bytes=4_000_000,
            pending_commit_high_watermark=2,
            pending_commit_low_watermark=0,
            batch_size=2,
            batch_queue_capacity=0,
            learner_updates_per_tick=3,
            publication_interval_updates=1,
            evaluation_interval_env_steps=12,
            evaluation_num_episodes=1,
            report_interval_s=0.1,
            shutdown_timeout_s=15,
            num_cpus_per_replay=0,
            num_cpus_per_runner=0,
            num_cpus_per_evaluation_runner=0,
            num_cpus_per_learner=0,
            num_gpus_per_learner=num_gpus,
            seed=seed + index,
        )
        members.append(
            PopulationMemberSpec.for_team(
                rollout,
                modules,
                settings,
                per_configs={mid: PrioritizedReplayConfig() for mid in modules},
            )
        )
    return tuple(members)


def mixed_pbt():
    return SimplePBTConfig(
        perturbation_interval_reports=10_000,
        reward_window_episodes=4,
        min_episodes_after_restart=1,
        seed=39,
        mutations={"manager/lr": FloatMutation(1e-5, 1e-3, factors=(0.8,))},
    )
