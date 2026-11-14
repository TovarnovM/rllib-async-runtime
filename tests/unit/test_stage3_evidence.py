from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from benchmarks.validate_dqn_gpu import validate_junit


def _junit(path, *, marker=None, device="cuda:0"):
    suite = ET.Element("testsuite")
    case = ET.SubElement(suite, "testcase", classname="tests.unit.test_gate", name="test_gate[gpu]")
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
        validate_junit(path, {"tests/unit/test_gate.py::test_gate[gpu]"}, require_devices=True)


def test_gpu_evidence_rejects_cpu_fallback_and_missing_matrix_point(tmp_path):
    path = tmp_path / "results.xml"
    _junit(path, device="cpu")
    with pytest.raises(ValueError, match="CUDA parameter-device"):
        validate_junit(path, {"tests/unit/test_gate.py::test_gate[gpu]"}, require_devices=True)
    with pytest.raises(ValueError, match="JUnit coverage failed"):
        validate_junit(path, {"tests/unit/test_other.py::test_gate[gpu]"}, require_devices=False)
