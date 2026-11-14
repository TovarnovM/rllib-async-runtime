"""Run the ordered CPU and combined stage-2/3 GPU acceptance blocks.

Invoke with the locked cu118 environment inside the repository devcontainer.
Every block gets a new process; skips and incomplete JUnit evidence fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

STAGE2_PATHS = (
    "tests/unit/test_graph_algorithms.py"
    "::test_graph_learner_update_has_finite_loss_and_updates_encoders",
    "tests/unit/test_dqn_adapter.py"
    "::test_full_restore_preserves_optimizer_rng_and_target_update_schedule",
    "tests/integration/test_single_member_async_dqn.py"
    "::test_dqn_runtime_publishes_evaluates_and_resumes",
    "tests/unit/test_recurrent_algorithms.py"
    "::test_recurrent_learner_update_and_next_update_after_restore",
    "tests/integration/test_single_member_async_team.py"
    "::test_mixed_recurrent_runtime_publishes_evaluates_and_resumes",
)


def validate_junit(path: Path, expected: set[str], *, require_devices: bool) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    observed = {
        f"{case.attrib['classname'].replace('.', '/')}.py::{case.attrib['name']}"
        for case in cases
    }
    bad = [
        case.attrib.get("name", "")
        for case in cases
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
    ]
    if bad or observed != expected or len(cases) != len(expected):
        raise ValueError(
            f"JUnit coverage failed: failed/skipped={bad}, "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    properties = {
        case.attrib["name"]: {
            item.attrib["name"]: item.attrib.get("value", "")
            for item in case.findall("properties/property")
        }
        for case in cases
    }
    if require_devices and any(
        not any(
            "device" in key and "cuda" in value.lower() for key, value in values.items()
        )
        for values in properties.values()
    ):
        raise ValueError(
            "every GPU case must record actual CUDA parameter-device evidence"
        )
    return {"passed": len(cases), "skipped": 0, "properties": properties}


def run_command(command, *, environment, log_path: Path, timeout_s: int):
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = 124
    return {
        "command": list(command),
        "exit_code": exit_code,
        "elapsed_s": time.monotonic() - started,
        "log": str(log_path),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
    }


def allocated_devices(environment: dict[str, str], output: Path) -> list[str]:
    probe = (
        "import json,ray,torch; "
        "print(json.dumps({'count':torch.cuda.device_count(),"
        "'available':torch.cuda.is_available(),'ray':ray.__version__,"
        "'torch':str(torch.__version__),"
        "'cuda':torch.version.cuda,'names':[torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]}))"
    )
    result = run_command(
        [sys.executable, "-c", probe],
        environment=environment,
        log_path=output / "allocated-devices.json",
        timeout_s=60,
    )
    if result["exit_code"]:
        raise RuntimeError("CUDA allocation probe failed; see allocated-devices.json")
    lines = (output / "allocated-devices.json").read_text().splitlines()
    info = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
    if not info["available"] or info["count"] < 2:
        raise RuntimeError("combined acceptance requires two allocated CUDA devices")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    devices = (
        visible.split(",")
        if visible is not None
        else [str(i) for i in range(info["count"])]
    )
    devices = [device.strip() for device in devices if device.strip()]
    if len(devices) < 2:
        raise RuntimeError(
            "the parent CUDA_VISIBLE_DEVICES allocation contains fewer than two GPUs"
        )
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-timeout-s", type=int, default=1800)
    args = parser.parse_args()
    if args.block_timeout_s < 1:
        parser.error("--block-timeout-s must be positive")
    if args.output.exists():
        parser.error("--output must be a new evidence directory")
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    if dirty:
        parser.error("acceptance requires a clean worktree")
    args.output.mkdir(parents=True)
    environment = dict(os.environ)
    manifest = {
        "schema_version": 1,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "parent_cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "blocks": [],
        "status": "running",
    }

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    save()
    try:
        devices = allocated_devices(environment, args.output)
        inventory = run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv",
            ],
            environment=environment,
            log_path=args.output / "nvidia-inventory.csv",
            timeout_s=30,
        )
        manifest["inventory"] = inventory
        if inventory["exit_code"]:
            raise RuntimeError("nvidia-smi inventory failed")
        blocks = (
            ("cpu", "not gpu and not cluster and not stress", ("tests",), "", None),
            ("stage2-gpu", "gpu", STAGE2_PATHS, devices[0], 11),
            (
                "stage3-rainbow-gpu",
                "gpu",
                ("tests/unit/test_rainbow.py",),
                devices[0],
                2,
            ),
            (
                "stage3-population-two-gpu",
                "gpu",
                (
                    "tests/integration/test_mixed_population.py"
                    "::test_mixed_population_replacement_checkpoint_and_resume",
                ),
                ",".join(devices[:2]),
                1,
            ),
        )
        for name, marker, paths, visible, expected_count in blocks:
            block = {"name": name, "status": "running"}
            manifest["blocks"].append(block)
            save()
            # Ray appends session/socket names. Keep their live path short even
            # when the checkout, evidence directory or inherited TMPDIR is long.
            with tempfile.TemporaryDirectory(prefix="r23-", dir="/tmp") as ray_tmp:
                logs_output = args.output / f"{name}-ray"
                block["ray_tmpdir"] = ray_tmp
                block["ray_logs"] = str(logs_output)
                env = {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": visible,
                    "RAY_TMPDIR": ray_tmp,
                }
                save()
                try:
                    selection = [sys.executable, "-m", "pytest", "-m", marker, *paths]
                    collected = run_command(
                        [*selection, "--collect-only", "-q"],
                        environment=env,
                        log_path=args.output / f"{name}-collection.log",
                        timeout_s=300,
                    )
                    block["collection"] = collected
                    save()
                    if collected["exit_code"]:
                        raise RuntimeError(f"{name}: collection failed")
                    expected = {
                        line.strip()
                        for line in Path(collected["log"]).read_text().splitlines()
                        if line.startswith("tests/") and "::" in line
                    }
                    if not expected or (
                        expected_count is not None and len(expected) != expected_count
                    ):
                        raise RuntimeError(
                            f"{name}: unexpected collected case count {len(expected)}"
                        )
                    block["expected_node_ids"] = sorted(expected)
                    junit = args.output / f"{name}.xml"
                    execution = run_command(
                        [
                            *selection,
                            "-vv",
                            "--durations=20",
                            "-o",
                            "junit_family=xunit1",
                            f"--junitxml={junit}",
                        ],
                        environment=env,
                        log_path=args.output / f"{name}.log",
                        timeout_s=args.block_timeout_s,
                    )
                    block["execution"] = execution
                    save()
                    if execution["exit_code"]:
                        raise RuntimeError(
                            f"{name}: pytest exit code {execution['exit_code']}"
                        )
                    block["junit"] = validate_junit(
                        junit, expected, require_devices=name != "cpu"
                    )
                finally:
                    failure_pending = sys.exc_info()[0] is not None
                    try:
                        for session in Path(ray_tmp).glob("ray/session_*"):
                            logs = session / "logs"
                            if not session.is_symlink() and logs.is_dir():
                                shutil.copytree(
                                    logs,
                                    logs_output / "ray" / session.name / "logs",
                                    symlinks=True,
                                )
                    except OSError as error:
                        block["ray_log_error"] = str(error)
                        if not failure_pending:
                            raise
            block["status"] = "passed"
            save()
            print(f"{name}: {block['junit']['passed']} passed, 0 skipped", flush=True)
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = str(error)
        for block in manifest["blocks"]:
            if block["status"] == "running":
                block["status"] = "failed"
                block["failure"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
