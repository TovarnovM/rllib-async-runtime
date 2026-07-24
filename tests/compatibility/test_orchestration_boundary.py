from __future__ import annotations

import inspect

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SAC


def test_standard_algorithm_setup_owns_the_control_plane() -> None:
    """Pin the source-level boundary without booting an unused control plane."""
    assert issubclass(SAC, Algorithm)

    setup_source = inspect.getsource(Algorithm.setup)
    assert "env_runner_group" in setup_source
    assert "learner_group" in setup_source
    assert "local_replay_buffer" in setup_source
    assert "super().setup(config)" in inspect.getsource(SAC.setup)
