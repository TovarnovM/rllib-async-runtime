from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns

import rllib_async.runtime.evaluation as evaluation_module
from rllib_async.protocols import (
    EpisodeEnvelope,
    FlatEpisodeCodec,
    FrozenVersions,
    WeightsDescriptor,
)
from rllib_async.rollout import EpisodeRolloutMetrics, EpisodeRolloutResult
from rllib_async.runtime import AsyncEvaluationGroup
from tests.helpers import make_sac_config


@dataclass(eq=False)
class FakeRef:
    value: object


class FakeRemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, *args, **kwargs) -> FakeRef:
        return FakeRef(self._function(*args, **kwargs))


class FakeEvaluationActor:
    def __init__(
        self,
        codec: FlatEpisodeCodec,
        runner_id: str,
        initial_weights: WeightsDescriptor,
    ) -> None:
        self._codec = codec
        self._runner_id = runner_id
        self._weights = initial_weights
        self._sequence = 0
        self.explore_arguments: list[bool] = []
        self.weight_arguments: list[WeightsDescriptor | None] = []
        self.collect_episode = FakeRemoteMethod(self._collect_episode)

    def _collect_episode(
        self,
        weights: WeightsDescriptor | None,
        *,
        explore: bool,
    ) -> EpisodeRolloutResult:
        self.weight_arguments.append(weights)
        self.explore_arguments.append(explore)
        if weights is not None:
            self._weights = weights
        payload = self._codec.encode(
            [
                {
                    Columns.OBS: 0.0,
                    Columns.NEXT_OBS: 1.0,
                    Columns.ACTIONS: 0.0,
                    Columns.REWARDS: 1.0,
                    Columns.TERMINATEDS: True,
                    Columns.TRUNCATEDS: False,
                }
            ]
        )
        sequence = self._sequence
        self._sequence += 1
        episode = EpisodeEnvelope(
            episode_id=f"member-0/{self._runner_id}/0/{sequence}",
            schema_version=self._codec.schema_version,
            producer_member_id="member-0",
            runner_id=self._runner_id,
            runner_generation=0,
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
                episode_return=float(sequence + 1),
                env_steps=1,
                agent_steps=1,
            ),
        )


class FakeEvaluationActorClass:
    created: ClassVar[list[FakeEvaluationActor]] = []

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
    ) -> FakeEvaluationActor:
        actor = FakeEvaluationActor(codec, runner_id, initial_weights)
        cls.created.append(actor)
        return actor


def make_weights(version: int) -> WeightsDescriptor:
    return WeightsDescriptor(
        member_id="member-0",
        module_versions={DEFAULT_MODULE_ID: version},
        learner_updates=version,
        published_at_monotonic=float(version),
        state={DEFAULT_MODULE_ID: {}},
    )


def install_fake_ray(monkeypatch) -> None:
    FakeEvaluationActorClass.created.clear()
    monkeypatch.setattr(evaluation_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        evaluation_module.ray,
        "wait",
        lambda refs, *, num_returns, timeout: (
            refs[:num_returns],
            refs[num_returns:],
        ),
    )
    monkeypatch.setattr(evaluation_module.ray, "get", lambda ref: ref.value)
    monkeypatch.setattr(evaluation_module.ray, "cancel", lambda ref: None)
    monkeypatch.setattr(
        evaluation_module.ray,
        "kill",
        lambda actor, *, no_restart: None,
    )
    monkeypatch.setattr(
        evaluation_module,
        "EpisodeRolloutActor",
        FakeEvaluationActorClass,
    )


def test_evaluation_round_is_frozen_and_has_no_replay_dependency(monkeypatch) -> None:
    install_fake_ray(monkeypatch)
    group = AsyncEvaluationGroup(
        make_sac_config(),
        FlatEpisodeCodec(),
        member_id="member-0",
        initial_weights=make_weights(0),
        episode_count=3,
        max_episode_steps=10,
        num_cpus_per_runner=0,
    )
    try:
        assert group.start_round(make_weights(0)) == 0
        first = group.drain(timeout_s=1)
        assert first is not None
        assert dict(first.module_versions) == {DEFAULT_MODULE_ID: 0}
        assert first.episode_returns == (1.0, 1.0, 1.0)

        assert group.start_round(make_weights(1)) == 1
        second = group.drain(timeout_s=1)
        assert second is not None
        assert dict(second.module_versions) == {DEFAULT_MODULE_ID: 1}
        assert second.episode_returns == (2.0, 2.0, 2.0)

        actors = FakeEvaluationActorClass.created
        assert all(actor.explore_arguments == [False, False] for actor in actors)
        assert all(actor.weight_arguments[0] is None for actor in actors)
        assert all(
            actor.weight_arguments[1] is not None
            and actor.weight_arguments[1].module_versions[DEFAULT_MODULE_ID] == 1
            for actor in actors
        )
        stats = group.get_stats()
        assert stats.pending_high_watermark == 3
        assert stats.rounds_completed == 2
        assert stats.latest_return_mean == 2.0
    finally:
        group.stop()


def test_evaluation_checkpoint_preserves_completed_round_metrics(monkeypatch) -> None:
    install_fake_ray(monkeypatch)
    group = AsyncEvaluationGroup(
        make_sac_config(),
        FlatEpisodeCodec(),
        member_id="member-0",
        initial_weights=make_weights(1),
        episode_count=2,
        max_episode_steps=10,
        num_cpus_per_runner=0,
    )
    restored = None
    try:
        group.start_round(make_weights(1))
        result = group.drain(timeout_s=1)
        assert result is not None
        checkpoint = group.get_checkpoint_state()

        install_fake_ray(monkeypatch)
        restored = AsyncEvaluationGroup(
            make_sac_config(),
            FlatEpisodeCodec(),
            member_id="member-0",
            initial_weights=make_weights(2),
            episode_count=2,
            max_episode_steps=10,
            num_cpus_per_runner=0,
            checkpoint_state=checkpoint,
        )
        recovered = restored.get_stats()
        assert recovered.rounds_completed == 1
        assert recovered.episodes_completed == 2
        assert recovered.latest_module_version == 1
        assert recovered.latest_return_mean == 1.0

        assert restored.start_round(make_weights(2)) == 1
        restored.drain(timeout_s=1)
        assert restored.get_stats().rounds_completed == 2
    finally:
        group.stop()
        if restored is not None:
            restored.stop()
