from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rllib_async.runtime.controller as controller_module
from rllib_async.runtime import LearnerHostTick, RuntimeState, SingleMemberAsyncSAC


class RemoteMethod:
    def remote(self, **kwargs):
        return kwargs


class LearnerActor:
    pause = RemoteMethod()


class RolloutGroup:
    def __init__(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True


class RecordingRemoteMethod:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class SchedulerLearnerActor:
    def __init__(self) -> None:
        self.tick = RecordingRemoteMethod()


class SchedulerRolloutGroup:
    def __init__(self, *, env_steps: int, agent_steps: int | None = None) -> None:
        self.env_steps = env_steps
        self.agent_steps = env_steps if agent_steps is None else agent_steps

    def get_stats(self):
        return SimpleNamespace(
            env_steps=self.env_steps,
            agent_steps=self.agent_steps,
        )

    def update_weights(self, weights) -> None:
        raise AssertionError(f"unexpected weights publication {weights!r}")


def make_scheduler_runtime(
    *,
    sampled_steps: int,
    batch_size: int,
    training_intensity: float,
    learning_starts: int = 0,
    learner_updates_per_tick: int = 4,
) -> tuple[SingleMemberAsyncSAC, SchedulerLearnerActor, SchedulerRolloutGroup]:
    runtime = object.__new__(SingleMemberAsyncSAC)
    runtime._state = RuntimeState.RUNNING
    runtime._sac_config = SimpleNamespace(
        count_steps_by="env_steps",
        num_steps_sampled_before_learning_starts=learning_starts,
    )
    runtime._config = SimpleNamespace(
        batch_size=batch_size,
        learner_updates_per_tick=learner_updates_per_tick,
    )
    runtime._target_training_intensity = training_intensity
    runtime._learner_updates_completed = 0
    runtime._pending_learner_tick = None
    learner = SchedulerLearnerActor()
    rollout = SchedulerRolloutGroup(env_steps=sampled_steps)
    runtime._learner_actor = learner
    runtime._rollout_group = rollout
    return runtime, learner, rollout


def completed_tick(updates: int) -> LearnerHostTick:
    return LearnerHostTick(
        synced_transactions=0,
        sync_has_more=False,
        updates_performed=updates,
        updates_skipped_learning_start=0,
        published_weights=None,
        learner_metrics=(),
    )


def test_training_intensity_resolution_matches_async_sampling_semantics() -> None:
    assert SingleMemberAsyncSAC._resolve_training_intensity(None, batch_size=8) == 8
    assert SingleMemberAsyncSAC._resolve_training_intensity(0, batch_size=8) == 8
    assert SingleMemberAsyncSAC._resolve_training_intensity(2.5, batch_size=8) == 2.5

    for invalid in (False, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="training_intensity"):
            SingleMemberAsyncSAC._resolve_training_intensity(
                invalid,
                batch_size=8,
            )


def test_learner_tick_requires_new_cumulative_update_budget() -> None:
    runtime, learner, rollout = make_scheduler_runtime(
        sampled_steps=1,
        batch_size=8,
        training_intensity=8,
    )

    runtime._schedule_learner_tick()
    assert learner.tick.calls == [
        {
            "sampled_env_steps": 1,
            "sampled_agent_steps": 1,
            "max_updates": 1,
        }
    ]

    runtime._pending_learner_tick = None
    runtime._accept_learner_tick(completed_tick(1))
    runtime._schedule_learner_tick()
    assert len(learner.tick.calls) == 1

    rollout.env_steps = 2
    rollout.agent_steps = 2
    runtime._schedule_learner_tick()
    assert learner.tick.calls[-1]["max_updates"] == 1


def test_explicit_training_intensity_preserves_fractional_credit_and_tick_cap() -> None:
    runtime, learner, rollout = make_scheduler_runtime(
        sampled_steps=3,
        batch_size=8,
        training_intensity=2,
    )

    runtime._schedule_learner_tick()
    assert learner.tick.calls == []

    rollout.env_steps = 4
    rollout.agent_steps = 4
    runtime._schedule_learner_tick()
    assert learner.tick.calls[-1]["max_updates"] == 1
    runtime._pending_learner_tick = None
    runtime._accept_learner_tick(completed_tick(1))

    rollout.env_steps = 7
    rollout.agent_steps = 7
    runtime._schedule_learner_tick()
    assert len(learner.tick.calls) == 1

    rollout.env_steps = 100
    rollout.agent_steps = 100
    runtime._schedule_learner_tick()
    assert learner.tick.calls[-1]["max_updates"] == 4


def test_learning_start_does_not_create_a_warmup_update_backlog() -> None:
    runtime, learner, rollout = make_scheduler_runtime(
        sampled_steps=7,
        batch_size=8,
        training_intensity=8,
        learning_starts=8,
    )

    runtime._schedule_learner_tick()
    assert learner.tick.calls == []

    rollout.env_steps = 8
    rollout.agent_steps = 8
    runtime._schedule_learner_tick()
    assert learner.tick.calls[-1]["max_updates"] == 1

    low_intensity, low_learner, _ = make_scheduler_runtime(
        sampled_steps=8,
        batch_size=8,
        training_intensity=1,
        learning_starts=8,
    )
    low_intensity._schedule_learner_tick()
    assert low_learner.tick.calls[-1]["max_updates"] == 1


@pytest.mark.parametrize("timeout_stage", ["learner_tick", "learner_pause"])
def test_pause_timeout_marks_runtime_failed(
    monkeypatch,
    timeout_stage: str,
) -> None:
    runtime = object.__new__(SingleMemberAsyncSAC)
    runtime._state = RuntimeState.RUNNING
    runtime._config = SimpleNamespace(shutdown_timeout_s=1.0)
    runtime._rollout_group = RolloutGroup()
    runtime._learner_actor = LearnerActor()

    if timeout_stage == "learner_tick":
        monkeypatch.setattr(
            runtime,
            "_finish_learner_tick",
            Mock(side_effect=TimeoutError("tick timeout")),
        )
    else:
        monkeypatch.setattr(runtime, "_finish_learner_tick", Mock())
        monkeypatch.setattr(
            controller_module.ray,
            "get",
            Mock(side_effect=TimeoutError("pause timeout")),
        )

    with pytest.raises(TimeoutError):
        runtime.pause(timeout_s=1.0)

    assert runtime._rollout_group.paused
    assert runtime.state is RuntimeState.FAILED
    with pytest.raises(RuntimeError, match="must be 'paused'"):
        runtime.resume()
    with pytest.raises(RuntimeError, match="must be 'running'"):
        runtime.pump_once()


def test_stopping_member_does_not_kill_externally_owned_replay(monkeypatch) -> None:
    runtime = object.__new__(SingleMemberAsyncSAC)
    runtime._state = RuntimeState.CREATED
    runtime._config = SimpleNamespace(shutdown_timeout_s=1.0)
    runtime._pending_learner_tick = None
    runtime._evaluation_group = None
    runtime._rollout_group = None
    runtime._learner_actor = None
    runtime._replay_actor = object()
    runtime._owns_replay_actor = False
    killed: list[object] = []
    monkeypatch.setattr(
        controller_module.ray,
        "kill",
        lambda actor, **_: killed.append(actor),
    )

    runtime.stop(graceful=False)

    assert runtime.state is RuntimeState.STOPPED
    assert killed == []
