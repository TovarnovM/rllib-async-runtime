from __future__ import annotations

import pickle

import numpy as np
import pytest

from rllib_async.protocols import FrozenVersions, WeightsDescriptor


def test_weights_descriptor_is_validated_and_pickle_safe() -> None:
    descriptor = WeightsDescriptor(
        member_id="member-0",
        module_versions={"default_policy": 3},
        learner_updates=17,
        published_at_monotonic=12.5,
        state={"default_policy": {"weight": np.array([1.0], dtype=np.float32)}},
    )

    restored = pickle.loads(pickle.dumps(descriptor))

    assert isinstance(restored.module_versions, FrozenVersions)
    assert dict(restored.module_versions) == {"default_policy": 3}
    assert restored.member_id == "member-0"
    assert restored.learner_updates == 17
    np.testing.assert_array_equal(
        restored.state["default_policy"]["weight"],
        np.array([1.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"member_id": ""},
        {"module_versions": {}},
        {"module_versions": {"default_policy": -1}},
        {"learner_updates": -1},
        {"published_at_monotonic": float("nan")},
    ],
)
def test_weights_descriptor_rejects_invalid_metadata(
    kwargs: dict[str, object],
) -> None:
    values = {
        "member_id": "member-0",
        "module_versions": {"default_policy": 0},
        "learner_updates": 0,
        "published_at_monotonic": 0.0,
        "state": {},
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        WeightsDescriptor(**values)
