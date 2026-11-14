"""Compare two flat SAC revisions on one host, outside the ordinary test suite.

Run in the project devcontainer with both Git commits available locally.
The existing end-to-end benchmark owns warm-up, work counters and Ray cleanup.
Each revision runs in its own worktree and locked CPU environment. Five paired
repeats alternate execution order; profiling is deliberately disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

BASELINE = "6d22868bc1a9d04754409aabc4b0c9192648ca58"
_WORKER = """
import json, os, resource, runpy
from pathlib import Path
try:
    runpy.run_module('benchmarks.end_to_end_throughput', run_name='__main__')
finally:
    Path(os.environ['RLLIB_BENCH_MEMORY_PATH']).write_text(json.dumps({
        'driver_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'largest_exited_child_peak_rss_kib':
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }))
"""


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve paired work comparisons and report noisy results as inconclusive."""

    if not pairs:
        raise ValueError("at least one paired measurement is required")
    ratios = []
    comparable = True
    for pair in pairs:
        measurements = {}
        for label in ("baseline", "candidate"):
            result = pair[label]["results"][0]
            duration = float(result["duration_s"])
            steps = int(result["measured_env_steps"])
            updates = int(result["measured_learner_updates"])
            if (
                not math.isfinite(duration)
                or duration <= 0
                or steps <= 0
                or updates <= 0
            ):
                raise ValueError("each measurement must contain real work and time")
            if not result["gates"]["all_passed"]:
                raise ValueError("a runtime invariant failed during measurement")
            measurements[label] = (duration / steps, updates / steps)
        before, after = measurements["baseline"], measurements["candidate"]
        ratios.append(after[0] / before[0])
        comparable &= 0.9 <= after[1] / before[1] <= 1.1
    median = statistics.median(ratios)
    relative_span = (max(ratios) - min(ratios)) / median
    verdict = (
        "inconclusive"
        if len(pairs) < 5 or relative_span >= 0.1 or not comparable
        else "pass" if median <= 1.1 else "regression"
    )
    return {
        "paired_time_per_world_step_ratios": ratios,
        "median_ratio": median,
        "relative_ratio_span": relative_span,
        "learner_updates_per_world_step_comparable": comparable,
        "time_regression_gate": verdict,
        "limit_ratio": 1.1,
        "aggregate_peak_memory_gate": "not_measured",
        "memory_scope": "driver and largest exited child separately, not their sum",
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _run(command: list[str], cwd: Path, log: Path, env: dict[str, str], timeout: int):
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"benchmark timed out; see {log}") from None
    if code:
        raise RuntimeError(f"benchmark exited {code}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-timesteps", type=int, default=2_000)
    parser.add_argument("--measure-timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--max-duration-s", type=int, default=180)
    args = parser.parse_args()
    if args.repeats < 5 or min(
        args.warmup_timesteps, args.measure_timesteps, args.max_duration_s
    ) < 1:
        parser.error("at least five repeats and positive work/time budgets required")
    if platform.system() != "Linux":
        parser.error("run in the Linux project devcontainer")
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    revisions = {
        label: _git(
            root, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"
        )
        for label, ref in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "revisions": revisions,
        "arguments": {**vars(args), "output": str(output)},
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "runs": [],
        "complete": False,
    }
    env = dict(os.environ)
    for name in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "RAY_ADDRESS"):
        env.pop(name, None)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    manifest["thread_limits"] = {
        name: env[name]
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    pairs = []
    with tempfile.TemporaryDirectory(prefix="rllib-flat-revisions-") as temporary:
        worktrees = {}
        try:
            for label, sha in revisions.items():
                path = Path(temporary) / label
                _git(root, "worktree", "add", "--detach", str(path), sha)
                worktrees[label] = path
            locks = {
                label: hashlib.sha256((path / "uv.lock").read_bytes()).hexdigest()
                for label, path in worktrees.items()
            }
            if len(set(locks.values())) != 1:
                raise ValueError("paired measurements require the same dependency lock")
            manifest["uv_lock_sha256"] = locks["baseline"]
            for repeat in range(args.repeats):
                pair = {}
                order = ("baseline", "candidate")
                if repeat % 2:
                    order = tuple(reversed(order))
                for label in order:
                    prefix = output / f"{repeat:02d}-{label}"
                    report_path = prefix.with_suffix(".json")
                    memory_path = prefix.with_suffix(".memory.json")
                    log_path = prefix.with_suffix(".log")
                    command = [
                        "uv",
                        "run",
                        "--locked",
                        "--extra",
                        "cpu",
                        "--group",
                        "dev",
                        "python",
                        "-c",
                        _WORKER,
                    ]
                    options = {
                        "mode": "direct",
                        "members": 1,
                        "runner-count": 1,
                        "batch-size": 128,
                        "training-intensity": 1,
                        "warmup-timesteps": args.warmup_timesteps,
                        "measure-timesteps": args.measure_timesteps,
                        "max-duration-s": args.max_duration_s,
                        "seed": args.seed + repeat,
                        "output": report_path,
                    }
                    for name, value in options.items():
                        command.extend((f"--{name}", str(value)))
                    run = {
                        "label": label,
                        "repeat": repeat,
                        "command": command,
                        "sha": revisions[label],
                        "status": "running",
                    }
                    manifest["runs"].append(run)
                    manifest_path = output / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    run_env = {
                        **env,
                        "RLLIB_BENCH_MEMORY_PATH": str(memory_path),
                        "PYTHONHASHSEED": str(args.seed + repeat),
                    }
                    try:
                        _run(
                            command,
                            worktrees[label],
                            log_path,
                            run_env,
                            args.max_duration_s + 120,
                        )
                        pair[label] = json.loads(report_path.read_text())
                        pair[label]["process_memory"] = json.loads(
                            memory_path.read_text()
                        )
                        run["status"] = "passed"
                    except Exception:
                        run["status"] = "failed"
                        raise
                    finally:
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                pairs.append(pair)
            result = {
                "manifest": manifest,
                "pairs": pairs,
                "summary": summarize(pairs),
            }
            manifest["complete"] = True
            (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(json.dumps(result["summary"], indent=2))
        finally:
            for path in worktrees.values():
                subprocess.run(
                    [
                        "git", "-C", str(root), "worktree", "remove", "--force",
                        str(path),
                    ],
                    check=False,
                )
    if result["summary"]["time_regression_gate"] == "regression":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
