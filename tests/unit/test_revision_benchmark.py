from __future__ import annotations

import pytest

from benchmarks.compare_flat_revisions import summarize


def _pair(ratio=1.0, updates=100):
    def report(duration, count):
        return {
            "results": [
                {
                    "duration_s": duration,
                    "measured_env_steps": 1_000,
                    "measured_learner_updates": count,
                    "gates": {"all_passed": True},
                }
            ]
        }

    return {"baseline": report(10, 100), "candidate": report(10 * ratio, updates)}


def test_time_gate_does_not_reward_less_learning_or_cherry_pick_noisy_repeats():
    assert summarize([_pair(0.9)] * 5)["time_regression_gate"] == "pass"
    assert summarize([_pair(1.2)] * 5)["time_regression_gate"] == "regression"
    assert summarize([_pair(0.5, 40)] * 5)["time_regression_gate"] == "inconclusive"
    noisy = [_pair(value) for value in (0.8, 0.9, 1.0, 1.1, 1.2)]
    assert summarize(noisy)["time_regression_gate"] == "inconclusive"
    assert summarize([_pair()] * 4)["time_regression_gate"] == "inconclusive"


def test_invalid_runtime_measurements_cannot_pass_the_regression_gate():
    with pytest.raises(ValueError, match="paired measurement"):
        summarize([])
    bad = _pair()
    bad["candidate"]["results"][0]["gates"]["all_passed"] = False
    with pytest.raises(ValueError, match="runtime invariant"):
        summarize([bad] * 5)
