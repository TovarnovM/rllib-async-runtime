from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.validate_evidence import (
    EXPECTED_PROFILE_COUNT,
    expected_document_names,
    validate_evidence,
)

OLD_COMMIT = "a" * 40
NEW_COMMIT = "b" * 40


def _write_evidence(root: Path) -> dict[str, Path]:
    profiles = root / "profiles"
    profiles.mkdir()
    paths = {}
    profile_index = 0
    names = expected_document_names()
    assert len(names) == 70

    for name in sorted(names):
        if name.startswith("gpu-single-"):
            result_count = 3
        elif name.startswith("gpu-population-"):
            result_count = 2
        else:
            result_count = 2 if name.startswith("replay-ingest-") else 4

        results = []
        for _ in range(result_count):
            profile = profiles / f"profile-{profile_index}.prof"
            profile.write_bytes(b"profile")
            result = {
                "gates": {"all_passed": True},
                "profile": str(profile),
            }
            if name.startswith("gpu-population-"):
                result["member_results"] = [
                    {"gates": {"all_passed": True}},
                    {"gates": {"all_passed": True}},
                ]
            results.append(result)
            profile_index += 1

        commit = NEW_COMMIT if name.startswith("gpu-population-") else OLD_COMMIT
        document = {
            "schema_version": 1,
            "benchmark": "test",
            "environment": {
                "git_commit": commit,
                "git_dirty": False,
                "cuda_available": True,
                "cuda_device_count": 2,
            },
            "results": results,
        }
        path = root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        paths[name] = path

    assert profile_index == EXPECTED_PROFILE_COUNT
    return paths


def test_evidence_validator_accepts_multiple_clean_commits(tmp_path: Path) -> None:
    _write_evidence(tmp_path)

    document_count, profile_count, commits = validate_evidence(tmp_path)

    assert document_count == 70
    assert profile_count == EXPECTED_PROFILE_COUNT
    assert commits == (OLD_COMMIT, NEW_COMMIT)


def test_evidence_validator_still_rejects_dirty_documents(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    path = paths["gpu-population-r1-e32-b128-u1.json"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["environment"]["git_dirty"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{path.name}:environment"):
        validate_evidence(tmp_path)
