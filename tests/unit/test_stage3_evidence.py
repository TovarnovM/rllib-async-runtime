from __future__ import annotations

import json
import os
import socket
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from benchmarks import validate_dqn_gpu as gpu_acceptance
from benchmarks.validate_dqn_gpu import validate_junit


def _junit(path, *, marker=None, device="cuda:0"):
    suite = ET.Element("testsuite")
    case = ET.SubElement(
        suite, "testcase", classname="tests.unit.test_gate", name="test_gate[gpu]"
    )
    props = ET.SubElement(case, "properties")
    ET.SubElement(props, "property", name="parameter_devices", value=device)
    if marker is not None:
        ET.SubElement(case, marker)
    ET.ElementTree(suite).write(path)


def test_gpu_evidence_requires_matching_case_and_actual_device(tmp_path):
    path = tmp_path / "results.xml"
    _junit(path)
    result = validate_junit(
        path, {"tests/unit/test_gate.py::test_gate[gpu]"}, require_devices=True
    )
    assert result["passed"] == 1
    assert result["skipped"] == 0


@pytest.mark.parametrize("marker", ["skipped", "failure", "error"])
def test_gpu_evidence_does_not_accept_skips_or_failed_cases(tmp_path, marker):
    path = tmp_path / "results.xml"
    _junit(path, marker=marker)
    with pytest.raises(ValueError, match="JUnit coverage failed"):
        validate_junit(
            path, {"tests/unit/test_gate.py::test_gate[gpu]"}, require_devices=True
        )


def test_gpu_evidence_rejects_cpu_fallback_and_missing_matrix_point(tmp_path):
    path = tmp_path / "results.xml"
    _junit(path, device="cpu")
    with pytest.raises(ValueError, match="CUDA parameter-device"):
        validate_junit(
            path, {"tests/unit/test_gate.py::test_gate[gpu]"}, require_devices=True
        )
    with pytest.raises(ValueError, match="JUnit coverage failed"):
        validate_junit(
            path, {"tests/unit/test_other.py::test_gate[gpu]"}, require_devices=False
        )


@pytest.mark.parametrize(
    "sampling, length, accepted",
    [("per", 5, False), ("per", 1, True), ("uniform", 5, True)],
)
def test_exact_position_benchmark_handles_per_window_credit(
    monkeypatch, capsys, sampling, length, accepted
):
    from benchmarks.team_acceptance import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "team_acceptance",
            "components",
            "--sampling",
            sampling,
            "--learning-length",
            str(length),
            "--output",
            "unused.json",
        ],
    )
    if accepted:
        args = parse_args()
        assert args.sampling == sampling
        assert args.learning_length == length
    else:
        with pytest.raises(SystemExit) as error:
            parse_args()
        assert error.value.code == 2
        assert "retain residual credit" in capsys.readouterr().err


@pytest.mark.parametrize(
    "outcome", ["passed", "collection", "pytest", "timeout", "pytest-and-log-copy"]
)
def test_acceptance_keeps_ray_sockets_short_and_retains_failure_evidence(
    tmp_path, monkeypatch, outcome
):
    checkout = tmp_path / ("nested-checkout-" * 12)
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    (checkout / "uv.lock").write_text("test lock\n")
    output = checkout / "artifacts" / "validation" / "evidence"
    inherited = checkout / "parent-temporary-directory"
    inherited.mkdir()
    marker = inherited / "foreign.log"
    marker.write_text("not owned by the runner")
    monkeypatch.setenv("TMPDIR", str(inherited))
    monkeypatch.setenv("RAY_TMPDIR", str(inherited))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    monkeypatch.setattr("sys.argv", ["validate_dqn_gpu", "--output", str(output)])
    metadata = {
        ("git", "status", "--porcelain=v1"): "",
        ("git", "rev-parse", "HEAD"): "a" * 40,
        ("uv", "--version"): "uv test",
    }
    monkeypatch.setattr(
        gpu_acceptance.subprocess,
        "check_output",
        lambda command, **kwargs: metadata[tuple(command)],
    )
    monkeypatch.setattr(
        gpu_acceptance, "allocated_devices", lambda *args: ["GPU-first", "GPU-second"]
    )
    counts = {
        "cpu": 1,
        "stage2-gpu": 11,
        "stage3-rainbow-gpu": 2,
        "stage3-population-two-gpu": 1,
    }
    roots = {}
    visible = {}

    def run(command, *, environment, log_path, timeout_s):
        result = {
            "command": list(command),
            "exit_code": 0,
            "log": str(log_path),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        }
        if command[0] == "nvidia-smi":
            log_path.write_text("inventory\n")
            return result
        collecting = "--collect-only" in command
        name = (
            log_path.name.removesuffix("-collection.log")
            if collecting
            else log_path.stem
        )
        root = Path(environment["RAY_TMPDIR"])
        roots[name] = root
        visible[name] = environment["CUDA_VISIBLE_DEVICES"]
        assert root.is_dir() and root != inherited
        if collecting:
            log_path.write_text(
                "\n".join(
                    f"tests/unit/test_gate.py::test_gate[{i}]"
                    for i in range(counts[name])
                )
            )
            if outcome == "collection":
                result["exit_code"] = 1
            return result

        session = root / "ray" / "session_2026-11-14_18-51-25_578300_166"
        sockets = session / "sockets"
        sockets.mkdir(parents=True)
        # This actual bind fails if live Ray state inherits the long output path.
        with socket.socket(socket.AF_UNIX) as unix_socket:
            unix_socket.bind(str(sockets / "plasma_store"))
        logs = session / "logs"
        logs.mkdir()
        (logs / "raylet.err").write_text(f"diagnostics for {name}")
        (session.parent / "session_latest").symlink_to(
            session, target_is_directory=True
        )
        log_path.write_text("pytest output\n")
        if outcome != "passed":
            result["exit_code"] = 124 if outcome == "timeout" else 1
            return result
        suite = ET.Element("testsuite")
        for i in range(counts[name]):
            case = ET.SubElement(
                suite,
                "testcase",
                classname="tests.unit.test_gate",
                name=f"test_gate[{i}]",
            )
            props = ET.SubElement(case, "properties")
            ET.SubElement(
                props,
                "property",
                name="parameter_devices",
                value="cpu" if name == "cpu" else "cuda:0",
            )
        ET.ElementTree(suite).write(output / f"{name}.xml")
        return result

    monkeypatch.setattr(gpu_acceptance, "run_command", run)
    if outcome == "pytest-and-log-copy":

        def cannot_copy(*args, **kwargs):
            raise OSError("diagnostic copy failed")

        monkeypatch.setattr(gpu_acceptance.shutil, "copytree", cannot_copy)
    if outcome == "passed":
        gpu_acceptance.main()
    else:
        expected = (
            "collection failed" if outcome == "collection" else "pytest exit code"
        )
        with pytest.raises(RuntimeError, match=expected):
            gpu_acceptance.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("passed" if outcome == "passed" else "failed")
    assert len(manifest["blocks"]) == (4 if outcome == "passed" else 1)
    assert all(block["status"] == manifest["status"] for block in manifest["blocks"])
    assert all(not root.exists() for root in roots.values())
    assert len(set(roots.values())) == len(roots)
    assert marker.read_text() == "not owned by the runner"
    assert os.environ["RAY_TMPDIR"] == str(inherited)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-first,GPU-second"
    for block in manifest["blocks"]:
        assert block["ray_tmpdir"] == str(roots[block["name"]])
        if outcome == "pytest-and-log-copy":
            assert block["ray_log_error"] == "diagnostic copy failed"
        elif outcome != "collection":
            saved_logs = list(Path(block["ray_logs"]).rglob("raylet.err"))
            assert len(saved_logs) == 1
            assert saved_logs[0].read_text() == f"diagnostics for {block['name']}"
            assert not list(Path(block["ray_logs"]).rglob("plasma_store"))
        if outcome != "passed":
            assert block["failure"] == manifest["failure"]
            if outcome != "collection":
                assert block["execution"]["exit_code"] == (
                    124 if outcome == "timeout" else 1
                )
    assert visible == (
        {
            "cpu": "",
            "stage2-gpu": "GPU-first",
            "stage3-rainbow-gpu": "GPU-first",
            "stage3-population-two-gpu": "GPU-first,GPU-second",
        }
        if outcome == "passed"
        else {"cpu": ""}
    )
