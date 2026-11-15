"""Counter-based telemetry must not confuse capacity, resets or PID reuse."""

from copy import deepcopy

import pytest

from benchmarks.cluster_monitor import add_rates, gpu_snapshot


def test_monitor_rates_use_one_core_and_full_duplex_link_directions():
    previous = {
        "monotonic_s": 10.0,
        "cpu_ticks": {"cpu": [100, 0, 0, 100, 0, 0, 0, 0]},
        "network": {"eth0": {"rx_bytes": 0, "tx_bytes": 0, "speed_mbps": 1000}},
        "processes": {"42": {"start_ticks": 5, "cpu_ticks": 100}},
    }
    current = deepcopy(previous)
    current["monotonic_s"] = 12.0
    current["cpu_ticks"]["cpu"] = [200, 0, 0, 150, 50, 0, 0, 0]
    current["network"]["eth0"].update(rx_bytes=250_000_000, tx_bytes=125_000_000)
    current["processes"]["42"]["cpu_ticks"] = 300
    result = add_rates(current, previous, clock_ticks=100)
    assert result["cpu_percent"]["cpu"] == {"busy": 50, "iowait": 25}
    assert result["processes"]["42"]["cpu_percent_one_core"] == 100
    assert result["network"]["eth0"]["rx_link_fraction"] == 1
    assert result["network"]["eth0"]["tx_link_fraction"] == 0.5
    current["processes"]["42"].pop("cpu_percent_one_core")
    current["processes"]["42"]["start_ticks"] = 6
    assert (
        "cpu_percent_one_core"
        not in add_rates(current, previous, clock_ticks=100)["processes"]["42"]
    )


def test_gpu_monitor_records_missing_tool_instead_of_faking_zero_load():
    result = gpu_snapshot("/nonexistent/nvidia-smi")
    assert result == {
        "devices_error": "FileNotFoundError",
        "processes_error": "FileNotFoundError",
    }


def test_counter_reset_is_not_negative_network_throughput():
    previous = {
        "monotonic_s": 1,
        "cpu_ticks": {},
        "processes": {},
        "network": {"eth0": {"rx_bytes": 10, "tx_bytes": 20}},
    }
    current = {
        "monotonic_s": 2,
        "cpu_ticks": {},
        "processes": {},
        "network": {"eth0": {"rx_bytes": 1, "tx_bytes": 2}},
    }
    assert add_rates(current, previous, clock_ticks=100)["network"]["eth0"][
        "rx_bytes_per_s"
    ] == pytest.approx(0)
