"""External Linux CPU/network/GPU sampler; never called from the runtime pump.

Run one process per node, inside the workload container with host networking
and visibility of its Ray processes. Raw counters are kept alongside rates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path


def read_snapshot(proc: Path = Path("/proc"), sys: Path = Path("/sys")) -> dict:
    cpu = {}
    for line in (proc / "stat").read_text().splitlines():
        fields = line.split()
        if fields[0].startswith("cpu"):
            cpu[fields[0]] = [int(value) for value in fields[1:9]]
    network = {}
    for line in (proc / "net/dev").read_text().splitlines()[2:]:
        name, values = line.split(":", 1)
        name = name.strip()
        values = [int(value) for value in values.split()]
        item = {
            "rx_bytes": values[0],
            "rx_errors": values[2],
            "rx_drops": values[3],
            "tx_bytes": values[8],
            "tx_errors": values[10],
            "tx_drops": values[11],
            "physical": (sys / "class/net" / name / "device").exists(),
        }
        try:
            item["speed_mbps"] = int((sys / "class/net" / name / "speed").read_text())
        except (OSError, ValueError):
            item["speed_mbps"] = None
        network[name] = item
    processes = {}
    page_size = os.sysconf("SC_PAGE_SIZE")
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text()
            name = stat[stat.index("(") + 1 : stat.rindex(")")]
            if not name.startswith("ray::"):
                continue
            fields = stat[stat.rindex(")") + 2 :].split()
            processes[path.name] = {
                "name": name,
                "cpu_ticks": int(fields[11]) + int(fields[12]),
                "start_ticks": int(fields[19]),
                "rss_bytes": int(fields[21]) * page_size,
            }
        except (OSError, ValueError, IndexError):
            continue  # A Ray worker can exit while /proc is being sampled.
    memory = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    tcp = {}
    lines = (proc / "net/snmp").read_text().splitlines()
    for header, values in zip(lines[::2], lines[1::2], strict=True):
        if header.startswith("Tcp:"):
            tcp = dict(
                zip(header.split()[1:], map(int, values.split()[1:]), strict=True)
            )
    return {
        "time_unix_s": time.time(),
        "monotonic_s": time.monotonic(),
        "cpu_ticks": cpu,
        "network": network,
        "processes": processes,
        "memory": memory,
        "tcp": tcp,
        "load_average": list(os.getloadavg()),
    }


def add_rates(current: dict, previous: dict, *, clock_ticks: int) -> dict:
    elapsed = current["monotonic_s"] - previous["monotonic_s"]
    if elapsed <= 0:
        return current
    current["interval_s"] = elapsed
    cpu_percent = {}
    for name, values in current["cpu_ticks"].items():
        old = previous["cpu_ticks"].get(name)
        if old is None:
            continue
        delta = [new - before for new, before in zip(values, old, strict=True)]
        total = sum(delta)
        if total > 0 and min(delta) >= 0:
            cpu_percent[name] = {
                "busy": 100 * (total - delta[3] - delta[4]) / total,
                "iowait": 100 * delta[4] / total,
            }
    current["cpu_percent"] = cpu_percent
    for name, values in current["network"].items():
        old = previous["network"].get(name)
        if old is None:
            continue
        for direction in ("rx", "tx"):
            delta = values[direction + "_bytes"] - old[direction + "_bytes"]
            values[direction + "_bytes_per_s"] = max(0, delta) / elapsed
            speed = values.get("speed_mbps")
            if speed is not None and speed > 0:
                values[direction + "_link_fraction"] = (
                    max(0, delta) * 8 / (elapsed * speed * 1e6)
                )
    for pid, values in current["processes"].items():
        old = previous["processes"].get(pid)
        if old and old["start_ticks"] == values["start_ticks"]:
            values["cpu_percent_one_core"] = (
                max(0, values["cpu_ticks"] - old["cpu_ticks"])
                * 100
                / (clock_ticks * elapsed)
            )
    return current


def gpu_snapshot(executable: str) -> dict:
    result = {}
    queries = {
        "devices": (
            "gpu",
            "index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        ),
        "processes": ("compute-apps", "gpu_uuid,pid,used_gpu_memory"),
    }
    for label, (kind, fields) in queries.items():
        try:
            output = subprocess.run(
                [
                    executable,
                    f"--query-{kind}={fields}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            ).stdout
            result[label] = [
                dict(
                    zip(fields.split(","), (item.strip() for item in row), strict=True)
                )
                for row in csv.reader(output.splitlines())
                if row
            ]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            result[label + "_error"] = type(error).__name__
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if not 0 < args.interval_s < float("inf"):
        parser.error("--interval-s must be finite and positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    nvidia_smi = shutil.which("nvidia-smi")
    with gzip.open(args.output, "xt", encoding="utf-8") as stream:
        while not stop.is_set():
            if args.parent_pid is not None and os.getppid() != args.parent_pid:
                break
            started = time.monotonic()
            current = read_snapshot()
            if previous is not None:
                add_rates(current, previous, clock_ticks=os.sysconf("SC_CLK_TCK"))
            current["schema_version"] = 1
            current["hostname"] = socket.gethostname()
            current["gpu"] = (
                gpu_snapshot(nvidia_smi) if nvidia_smi else {"available": False}
            )
            stream.write(json.dumps(current, allow_nan=False) + "\n")
            stream.flush()
            previous = current
            stop.wait(max(0.0, args.interval_s - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
