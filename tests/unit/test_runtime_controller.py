from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rllib_async.runtime.controller as controller_module
from rllib_async.runtime import RuntimeState, SingleMemberAsyncSAC


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
