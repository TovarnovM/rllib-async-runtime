"""Summarize saved MARL telemetry offline; no profiling work enters training."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def distribution(values: list[float]) -> dict:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
        "max": values[-1],
    }


def summarize(directory: Path) -> dict:
    candidates = list((directory / "marl").glob("*/result.json"))
    reports = []
    for path in candidates:
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if "population" in item:
                reports.append(item)
    reports.sort(key=lambda item: item["timestamp"])
    if not reports:
        raise ValueError("No Tune population reports; inspect the failed run logs")
    start, end = reports[0]["timestamp"], reports[-1]["timestamp"]
    nodes = {}
    for path in sorted(directory.glob("node-*.jsonl.gz")):
        metrics = defaultdict(list)
        records = 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                sample = json.loads(line)
                if not start <= sample["time_unix_s"] <= end:
                    continue
                records += 1
                for key, value in sample.get("cpu_percent", {}).get("cpu", {}).items():
                    metrics[f"cpu/{key}_percent"].append(value)
                for interface, values in sample["network"].items():
                    for direction in ("rx", "tx"):
                        for suffix in ("bytes_per_s", "link_fraction"):
                            key = f"{direction}_{suffix}"
                            if key in values:
                                metrics[f"network/{interface}/{key}"].append(values[key])
                for values in sample["processes"].values():
                    if "cpu_percent_one_core" in values:
                        metrics[f"process/{values['name']}/cpu_percent_one_core"].append(
                            values["cpu_percent_one_core"]
                        )
                for gpu in sample["gpu"].get("devices", []):
                    for key in ("utilization.gpu", "memory.used", "power.draw"):
                        try:
                            metrics[f"gpu/{gpu['index']}/{key}"].append(float(gpu[key]))
                        except ValueError:
                            continue
        nodes[path.name] = {
            "records": records,
            "metrics": {key: distribution(values) for key, values in metrics.items()},
        }
    last = reports[-1]
    return {
        "schema_version": 1,
        "measured_window_unix_s": [start, end],
        "window_note": "Excludes initial actor startup, before the first Tune report.",
        "reports": len(reports),
        "population": last["population"],
        "members": last["members"],
        "replay": last["replay"],
        "monitor_errors": json.loads((directory / "monitor-errors.json").read_text()),
        "nodes": nodes,
        "quality_gate": "Not inferred from utilization or optimizer progress; compare held-out returns.",
        "network_note": "RX and TX are independent; do not sum virtual and physical interfaces.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = summarize(args.directory)
    target = args.directory / "summary.json"
    target.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(target)


if __name__ == "__main__":
    main()
