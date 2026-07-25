from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns

import rllib_async.rollout.group as group_module
from rllib_async.protocols import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
    WeightsDescriptor,
)
from rllib_async.replay.reference import EpisodeStore
from rllib_async.rollout import (
    AsyncRolloutGroup,
    EpisodeRolloutMetrics,
    EpisodeRolloutResult,
)
from tests.helpers import make_sac_config


@dataclass(eq=False)
class FakeRef:
    value: object


class FakeRemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, *args, **kwargs) -> FakeRef:
        return FakeRef(self._function(*args, **kwargs))


class FakeRolloutActor:
    def __init__(
        self,
        codec: FlatEpisodeCodec,
        *,
        member_id: str,
        runner_id: str,
        runner_generation: int,
        initial_weights: WeightsDescriptor,
    ) -> None:
        self._codec = codec
        self._member_id = member_id
        self._runner_id = runner_id
        self._runner_generation = runner_generation
        self._sequence = 0
        self._weights = initial_weights
        self.weight_arguments: list[WeightsDescriptor | None] = []
        self.collect_episode = FakeRemoteMethod(self._collect_episode)

    def _collect_episode(
        self,
        weights: WeightsDescriptor | None,
        *,
        explore: bool,
    ) -> EpisodeRolloutResult:
        self.weight_arguments.append(weights)
        if weights is not None:
            self._weights = weights
        transition = {
            Columns.OBS: 0.0,
            Columns.NEXT_OBS: 1.0,
            Columns.ACTIONS: 0.0,
            Columns.REWARDS: 1.0,
            Columns.TERMINATEDS: True,
            Columns.TRUNCATEDS: False,
        }
        payload = self._codec.encode([transition])
        sequence = self._sequence
        self._sequence += 1
        episode = EpisodeEnvelope(
            episode_id=(
                f"{self._member_id}/{self._runner_id}/"
                f"{self._runner_generation}/{sequence}"
            ),
            schema_version=self._codec.schema_version,
            producer_member_id=self._member_id,
            runner_id=self._runner_id,
            runner_generation=self._runner_generation,
            local_episode_seq=sequence,
            behavior_versions=FrozenVersions(self._weights.module_versions),
            env_steps=1,
            agent_steps=1,
            terminated=True,
            truncated=False,
            estimated_bytes=payload.estimated_bytes,
            payload=payload,
        )
        return EpisodeRolloutResult(
            episode=episode,
            metrics=EpisodeRolloutMetrics(
                episode_time_s=0.001,
                episode_return=1.0,
                env_steps=1,
                agent_steps=1,
            ),
        )


class FakeRolloutActorClass:
    created: ClassVar[dict[str, FakeRolloutActor]] = {}

    @classmethod
    def options(cls, **options):
        return cls

    @classmethod
    def remote(
        cls,
        config,
        codec,
        *,
        member_id,
        runner_id,
        runner_generation,
        max_episode_steps,
        initial_weights,
        worker_index,
    ) -> FakeRolloutActor:
        actor = FakeRolloutActor(
            codec,
            member_id=member_id,
            runner_id=runner_id,
            runner_generation=runner_generation,
            initial_weights=initial_weights,
        )
        cls.created[runner_id] = actor
        return actor


class FakeReplayActor:
    def __init__(self, codec: FlatEpisodeCodec) -> None:
        self._store = EpisodeStore(
            codec,
            capacity_transitions=1_000,
            capacity_bytes=1_000_000,
        )
        self.commit_episode = FakeRemoteMethod(self._store.commit_episode)


def install_fake_ray(monkeypatch) -> None:
    FakeRolloutActorClass.created.clear()
    monkeypatch.setattr(group_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        group_module.ray,
        "wait",
        lambda refs, *, num_returns, timeout: (
            refs[:num_returns],
            refs[num_returns:],
        ),
    )
    monkeypatch.setattr(group_module.ray, "get", lambda ref: ref.value)
    monkeypatch.setattr(group_module.ray, "cancel", lambda ref: None)
    monkeypatch.setattr(
        group_module.ray,
        "kill",
        lambda actor, *, no_restart: None,
    )
    monkeypatch.setattr(
        group_module,
        "EpisodeRolloutActor",
        FakeRolloutActorClass,
    )


def make_weights(version: int) -> WeightsDescriptor:
    return WeightsDescriptor(
        member_id="member-0",
        module_versions={DEFAULT_MODULE_ID: version},
        learner_updates=version,
        published_at_monotonic=float(version),
        state={DEFAULT_MODULE_ID: {}},
    )


def make_group(
    monkeypatch,
    *,
    high: int,
    low: int,
) -> AsyncRolloutGroup:
    install_fake_ray(monkeypatch)
    codec = FlatEpisodeCodec()
    return AsyncRolloutGroup(
        make_sac_config(),
        codec,
        FakeReplayActor(codec),
        member_id="member-0",
        initial_weights=make_weights(0),
        runner_count=4,
        max_episode_steps=10,
        pending_commit_high_watermark=high,
        pending_commit_low_watermark=low,
        num_cpus_per_runner=0,
    )


def test_watermark_reserves_commit_capacity_before_sampling(monkeypatch) -> None:
    group = make_group(monkeypatch, high=4, low=1)
    try:
        group.start()
        initial = group.get_stats()
        assert initial.pending_sample_calls == 4
        assert initial.pending_episode_commits == 0

        assert group.poll(max_events=4) == []
        saturated = group.get_stats()
        assert saturated.pending_sample_calls == 0
        assert saturated.pending_episode_commits == 4
        assert saturated.backpressured

        completions = group.poll(max_events=3)
        assert len(completions) == 3
        resumed = group.get_stats()
        assert resumed.sample_calls_started == 7
        assert resumed.outstanding_high_watermark == 4
        assert resumed.pending_sample_calls + resumed.pending_episode_commits == 4
    finally:
        group.stop()


def test_new_weights_apply_to_next_episode_and_lag_is_measured(monkeypatch) -> None:
    group = make_group(monkeypatch, high=8, low=3)
    try:
        group.start()
        assert group.update_weights(make_weights(1))
        assert not group.update_weights(make_weights(0))

        first_generation = []
        for _ in range(8):
            first_generation.extend(group.poll(max_events=4))
            if first_generation:
                break
        assert first_generation
        assert all(
            completion.policy_version_lag == 1 for completion in first_generation
        )
        assert all(
            completion.episode.behavior_versions[DEFAULT_MODULE_ID] == 0
            for completion in first_generation
        )

        second_generation = []
        for _ in range(8):
            second_generation.extend(group.poll(max_events=4))
            if second_generation:
                break
        assert second_generation
        assert all(
            completion.episode.behavior_versions[DEFAULT_MODULE_ID] == 1
            for completion in second_generation
        )
    finally:
        group.stop()


def test_weights_are_sent_only_when_the_runner_version_changes(monkeypatch) -> None:
    group = make_group(monkeypatch, high=100, low=50)
    try:
        group.start()
        actors = tuple(FakeRolloutActorClass.created.values())
        assert all(actor.weight_arguments == [None] for actor in actors)

        assert group.poll(max_events=4) == []
        assert all(actor.weight_arguments == [None, None] for actor in actors)

        assert group.update_weights(make_weights(1))
        assert group.poll(max_events=4) == []
        for actor in actors:
            assert actor.weight_arguments[:2] == [None, None]
            publication = actor.weight_arguments[2]
            assert publication is not None
            assert publication.module_versions[DEFAULT_MODULE_ID] == 1

        assert group.poll(max_events=4) == []
        assert all(actor.weight_arguments[3] is None for actor in actors)
    finally:
        group.stop()


def test_runner_restart_advances_generation_and_resets_sequence(monkeypatch) -> None:
    group = make_group(monkeypatch, high=8, low=3)
    try:
        group.start()
        assert group.restart_runner("runner-0") == 1
        assert FakeRolloutActorClass.created["runner-0"].weight_arguments == [None]

        restarted = []
        for _ in range(8):
            restarted.extend(
                completion
                for completion in group.poll(max_events=4)
                if completion.episode.runner_id == "runner-0"
            )
            if restarted:
                break
        assert len(restarted) == 1
        assert restarted[0].episode.runner_generation == 1
        assert restarted[0].episode.local_episode_seq == 0
        assert restarted[0].episode.episode_id == "member-0/runner-0/1/0"
    finally:
        group.stop()
