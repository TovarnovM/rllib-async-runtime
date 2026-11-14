from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import pytest

import rllib_async.rollout.group as group_module
import rllib_async.runtime.evaluation as evaluation_module
from rllib_async.protocols import EpisodeEnvelope, FrozenVersions, WeightsDescriptor
from rllib_async.protocols.episodes import (
    MultiModuleTransition,
    RawMultiModuleEpisodeCodec,
)
from rllib_async.rollout import AsyncRolloutGroup, EpisodeRolloutMetrics
from rllib_async.rollout.episode_runner import (
    EpisodeMetadata,
    EpisodeRolloutResult,
    EpisodeRolloutSummary,
    WeightVersionError,
)
from rllib_async.runtime.evaluation import AsyncEvaluationGroup
from tests.helpers import make_sac_config
from tests.unit.test_rollout_scheduler import (
    FakeRef,
    FakeRemoteMethod,
    FakeReplayActor,
    FakeSplitRemoteMethod,
    install_fake_ray,
    make_group,
)


def team_weights(epoch: int, *, inactive: bool = False) -> WeightsDescriptor:
    module_ids = (
        ("manager", "worker", "inactive") if inactive else ("manager", "worker")
    )
    return WeightsDescriptor(
        member_id="member-0",
        module_versions={module_id: epoch for module_id in module_ids},
        learner_updates=epoch,
        published_at_monotonic=float(epoch),
        state={
            module_id: {"value": epoch if module_id == "manager" else 0}
            for module_id in module_ids
        },
    )


class TeamActor:
    def __init__(self, codec, *, initial_weights, **kwargs):
        self.codec = codec
        self.weights = initial_weights
        self.runner_id = kwargs["runner_id"]
        self.generation = kwargs["runner_generation"]
        self.sequence = 0
        self.clocks = []
        self.weight_arguments = []
        self.explore_arguments = []
        self.collect_episode_refs = FakeSplitRemoteMethod(self.collect)
        self.evaluate = FakeRemoteMethod(
            lambda weights: EpisodeRolloutSummary.from_result(
                self.collect(weights, explore=False)
            )
        )

    def collect(self, weights, *, explore, sampled_module_steps=None):
        self.clocks.append(sampled_module_steps)
        self.weight_arguments.append(weights)
        self.explore_arguments.append(explore)
        if weights is not None:
            self.weights = weights
        transitions = []
        for module_id, env_ts, duration in (
            ("manager", (0, 3), 2),
            ("worker", (1, 2, 4, 5), 1),
        ):
            for agent_t, env_t in enumerate(env_ts):
                transitions.append(
                    MultiModuleTransition(
                        env_t=env_t,
                        agent_t=agent_t,
                        agent_id=module_id,
                        module_id=module_id,
                        data={
                            "obs": float(agent_t),
                            "new_obs": float(agent_t + 1),
                            "actions": 0,
                            "rewards": float(duration),
                            "terminateds": agent_t == len(env_ts) - 1,
                            "truncateds": False,
                            "world_t": agent_t * duration,
                            "duration": duration,
                            "raw_rewards": (1.0,) * duration,
                            "state_reset": agent_t == 0,
                            "subtask_terminated": False,
                        },
                    )
                )
        payload = self.codec.encode(transitions)
        episode = EpisodeEnvelope(
            episode_id=(f"member-0/{self.runner_id}/{self.generation}/{self.sequence}"),
            schema_version=self.codec.schema_version,
            producer_member_id="member-0",
            runner_id=self.runner_id,
            runner_generation=self.generation,
            local_episode_seq=self.sequence,
            behavior_versions={
                module_id: self.weights.module_versions[module_id]
                for module_id in payload.module_ids
            },
            env_steps=6,
            agent_steps=6,
            terminated=True,
            truncated=False,
            estimated_bytes=payload.estimated_bytes,
            payload=payload,
        )
        self.sequence += 1
        return EpisodeRolloutResult(
            episode,
            EpisodeRolloutMetrics(
                episode_time_s=0.001,
                episode_return=4.0,
                env_steps=6,
                agent_steps=6,
                module_steps=(("manager", 2), ("worker", 4)),
                world_steps=4,
            ),
        )


class TeamActorClass:
    created: ClassVar[dict[str, TeamActor]] = {}

    @classmethod
    def options(cls, **kwargs):
        return cls

    @classmethod
    def remote(cls, config, codec, *, initial_weights, **kwargs):
        actor = TeamActor(codec, initial_weights=initial_weights.value, **kwargs)
        cls.created[actor.runner_id] = actor
        return actor


def install_team_ray(monkeypatch):
    install_fake_ray(monkeypatch)
    TeamActorClass.created.clear()
    monkeypatch.setattr(group_module, "EpisodeRolloutActor", TeamActorClass)
    monkeypatch.setattr(evaluation_module, "EpisodeRolloutActor", TeamActorClass)

    def get(ref):
        assert not isinstance(ref.value, EpisodeEnvelope), "payload fetched by driver"
        return ref.value

    monkeypatch.setattr(group_module.ray, "get", get)


def team_config(*, inactive: bool = False):
    policies = {"manager", "worker", "inactive"} if inactive else {"manager", "worker"}
    return make_sac_config().multi_agent(policies=policies)


def team_group(*, checkpoint_state=None, weights=None, inactive: bool = False):
    codec = RawMultiModuleEpisodeCodec()
    return AsyncRolloutGroup(
        team_config(inactive=inactive),
        codec,
        FakeReplayActor(codec),
        member_id="member-0",
        initial_weights=weights or team_weights(0, inactive=inactive),
        runner_count=1,
        max_episode_steps=10,
        pending_commit_high_watermark=1,
        pending_commit_low_watermark=0,
        num_cpus_per_runner=0,
        checkpoint_state=checkpoint_state,
    )


def test_module_clocks_resume_from_compact_summaries_without_fetching_payload(
    monkeypatch,
):
    install_team_ray(monkeypatch)
    group = team_group()
    try:
        group.start()
        group.pause()
        group.drain(timeout_s=1)
        assert TeamActorClass.created["runner-0"].clocks == [
            {"manager": 0, "worker": 0}
        ]
        assert group.module_steps == {"manager": 2, "worker": 4}
        assert group.world_steps == 4
        assert group.env_steps == group.agent_steps == 6
        state = group.get_checkpoint_state()
    finally:
        group.stop()

    restored = team_group(checkpoint_state=state)
    try:
        restored.start()
        restored.pause()
        restored.drain(timeout_s=1)
        assert TeamActorClass.created["runner-0"].clocks == [
            {"manager": 2, "worker": 4}
        ]
        stats = restored.get_stats()
        assert dict(stats.module_steps) == {"manager": 4, "worker": 8}
        assert stats.world_steps == 8
        assert stats.env_steps == 12
        assert restored.get_checkpoint_state()["module_steps"] == dict(
            stats.module_steps
        )
    finally:
        restored.stop()


def test_team_publication_installs_complete_bundle_on_next_episode(monkeypatch):
    install_team_ray(monkeypatch)
    group = team_group()
    try:
        group.start()
        newer = team_weights(1)
        assert group.update_weights(newer)
        group.pause()
        old = group.drain(timeout_s=1)
        assert dict(old[0].episode_metadata.behavior_versions) == {
            "manager": 0,
            "worker": 0,
        }
        group.resume()
        group.pause()
        new = group.drain(timeout_s=1)
        assert dict(new[0].episode_metadata.behavior_versions) == {
            "manager": 1,
            "worker": 1,
        }
        assert TeamActorClass.created["runner-0"].weight_arguments == [None, newer]
        incomplete = replace(
            newer, module_versions={"manager": 2}, state={"manager": {}}
        )
        with pytest.raises(WeightVersionError, match="module versions"):
            group.update_weights(incomplete)
    finally:
        group.stop()


def test_sparse_episode_versions_preserve_inactive_module_across_restore(monkeypatch):
    install_team_ray(monkeypatch)
    group = team_group(weights=team_weights(2, inactive=True), inactive=True)
    newer = team_weights(5, inactive=True)
    try:
        group.start()
        group.update_weights(newer)
        group.pause()
        completions = group.drain(timeout_s=1)
        assert dict(completions[0].episode_metadata.behavior_versions) == {
            "manager": 2,
            "worker": 2,
        }
        assert completions[0].policy_version_lag == 3
        assert group.module_steps == {"manager": 2, "worker": 4, "inactive": 0}
        state = group.get_checkpoint_state()
        assert state["latest_module_versions"] == dict(newer.module_versions)
    finally:
        group.stop()
    restored = team_group(checkpoint_state=state, weights=newer, inactive=True)
    try:
        restored.start()
        restored.pause()
        completions = restored.drain(timeout_s=1)
        assert completions[0].policy_version_lag == 0
        assert restored.module_steps == {"manager": 4, "worker": 8, "inactive": 0}
    finally:
        restored.stop()


@pytest.mark.parametrize(
    "behavior, message",
    [
        ({}, "module IDs"),
        ({"unknown": 2}, "module IDs"),
        ({"manager": 3}, "newer than publication"),
    ],
)
def test_sparse_rollout_rejects_empty_unknown_and_future_versions(
    monkeypatch, behavior, message
):
    install_team_ray(monkeypatch)
    group = team_group(weights=team_weights(2))
    try:
        metadata = EpisodeMetadata(
            "member-0/runner-0/0/0", "runner-0", 0, FrozenVersions(behavior), 1, 1
        )
        with pytest.raises(group_module.RolloutGroupError, match=message):
            group._policy_lag(metadata)
    finally:
        group.stop()


@pytest.mark.parametrize("missing", ["module_steps", "world_steps"])
def test_raw_checkpoint_rejects_missing_clocks_before_creating_actors(
    monkeypatch, missing
):
    install_team_ray(monkeypatch)
    group = team_group()
    try:
        group.start()
        group.pause()
        group.drain(timeout_s=1)
        state = group.get_checkpoint_state()
    finally:
        group.stop()
    del state[missing]
    TeamActorClass.created.clear()
    with pytest.raises(ValueError, match=missing):
        team_group(checkpoint_state=state)
    assert TeamActorClass.created == {}


def test_legacy_flat_checkpoint_infers_only_its_known_single_step_clock(monkeypatch):
    group = make_group(monkeypatch, high=4, low=1)
    try:
        group.start()
        group.pause()
        group.drain(timeout_s=1)
        state = group.get_checkpoint_state()
    finally:
        group.stop()
    del state["module_steps"]
    del state["world_steps"]
    restored = make_group(monkeypatch, high=4, low=1, checkpoint_state=state)
    try:
        assert sum(restored.module_steps.values()) == state["agent_steps"]
        assert restored.world_steps == state["env_steps"]
    finally:
        restored.stop()


def test_team_evaluation_uses_frozen_shared_bundle_and_separate_actors(monkeypatch):
    install_team_ray(monkeypatch)

    def unexpected_put(value):
        raise AssertionError("evaluation duplicated the shared publication")

    monkeypatch.setattr(group_module.ray, "put", unexpected_put)
    initial = team_weights(0, inactive=True)
    group = AsyncEvaluationGroup(
        team_config(inactive=True),
        RawMultiModuleEpisodeCodec(),
        member_id="member-0",
        initial_weights=initial,
        initial_weights_ref=FakeRef(initial),
        episode_count=2,
        max_episode_steps=10,
        num_cpus_per_runner=0,
    )
    try:
        group.start_round(initial)
        result = group.drain(timeout_s=1)
        assert result.module_versions == initial.module_versions
        newer = team_weights(1, inactive=True)
        group.start_round(newer, weights_ref=FakeRef(newer))
        result = group.drain(timeout_s=1)
        assert result.module_versions == newer.module_versions
        assert all(
            actor.explore_arguments == [False, False]
            and actor.weight_arguments == [None, newer]
            and actor.clocks == [None, None]
            for actor in TeamActorClass.created.values()
        )
        assert group.get_stats().episodes_completed == 4
    finally:
        group.stop()


@pytest.mark.parametrize(
    "behavior", [{}, {"unknown": 2}, {"manager": 1}, {"manager": 3}]
)
def test_sparse_evaluation_rejects_missing_unknown_and_wrong_publications(
    monkeypatch, behavior
):
    install_team_ray(monkeypatch)
    initial = team_weights(2, inactive=True)
    group = AsyncEvaluationGroup(
        team_config(inactive=True),
        RawMultiModuleEpisodeCodec(),
        member_id="member-0",
        initial_weights=initial,
        episode_count=1,
        max_episode_steps=10,
        num_cpus_per_runner=0,
    )
    try:
        group.start_round(initial)
        ref = next(iter(group._pending))
        ref.value = replace(
            ref.value,
            episode=replace(
                ref.value.episode, behavior_versions=FrozenVersions(behavior)
            ),
        )
        with pytest.raises(
            evaluation_module.EvaluationGroupError, match="mixed weight publications"
        ):
            group.drain(timeout_s=1)
    finally:
        group.stop()
