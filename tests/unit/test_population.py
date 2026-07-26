from __future__ import annotations

import pytest
import ray

from rllib_async.runtime import (
    AsyncSACRuntimeConfig,
    PopulationLauncher,
    PopulationMemberSpec,
)
from tests.helpers import make_sac_config


def test_population_rejects_incompatible_observation_action_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    members = (
        PopulationMemberSpec(
            make_sac_config().environment("Pendulum-v1"),
            AsyncSACRuntimeConfig(member_id="member-0"),
        ),
        PopulationMemberSpec(
            make_sac_config().environment("CartPole-v1"),
            AsyncSACRuntimeConfig(member_id="member-1"),
        ),
    )

    with pytest.raises(ValueError, match="observation/action spaces"):
        PopulationLauncher(
            members,
            replay_actor_name="incompatible-spaces",
            namespace="test",
        )
