"""Validate a complete Phase 11 target-hardware evidence bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

EXPECTED_PROFILE_COUNT = 180
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def expected_document_names() -> set[str]:
    names = {
        "replay-ingest-short.json",
        "replay-ingest-long.json",
    }
    names.update(
        f"fast-replay-e{episode_length}-b{batch_size}.json"
        for episode_length in (32, 512)
        for batch_size in (128, 512)
    )
    names.update(
        f"gpu-single-r{runners}-e{episode_length}-b{batch_size}-u{intensity}.json"
        for runners in (1, 4, 8, 16)
        for episode_length in (32, 512)
        for batch_size in (128, 512)
        for intensity in (1, 4)
    )
    names.update(
        f"gpu-population-r{runners}-e{episode_length}-b{batch_size}-u{intensity}.json"
        for runners in (1, 4, 8, 12)
        for episode_length in (32, 512)
        for batch_size in (128, 512)
        for intensity in (1, 4)
    )
    return names


def validate_evidence(root: Path) -> tuple[int, int, tuple[str, ...]]:
    """Validate matrix coverage, per-document provenance, gates, and profiles."""

    if not root.is_dir():
        raise ValueError(f"evidence directory does not exist: {root}")

    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read {path.name}: {error}") from error
        if (
            isinstance(document, dict)
            and document.get("schema_version") == 1
            and document.get("benchmark")
        ):
            documents.append((path, document))
    if not documents:
        raise ValueError("no benchmark JSON documents found")

    expected_names = expected_document_names()
    observed_names = {path.name for path, _ in documents}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing or unexpected:
        raise ValueError(f"matrix mismatch: missing={missing}, unexpected={unexpected}")

    failed = []
    profile_names = set()
    commits = set()
    for path, document in documents:
        environment = document.get("environment")
        if not isinstance(environment, dict):
            failed.append(f"{path.name}:environment")
        else:
            commit = environment.get("git_commit")
            cuda_device_count = environment.get("cuda_device_count")
            environment_valid = (
                isinstance(commit, str)
                and _FULL_GIT_COMMIT.fullmatch(commit) is not None
                and environment.get("git_dirty") is False
                and environment.get("cuda_available") is True
                and isinstance(cuda_device_count, int)
                and not isinstance(cuda_device_count, bool)
                and cuda_device_count >= 2
            )
            if environment_valid:
                commits.add(commit)
            else:
                failed.append(f"{path.name}:environment")

        results = document.get("results")
        if not isinstance(results, list) or not results:
            failed.append(f"{path.name}:missing-results")
            continue
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                failed.append(f"{path.name}:result[{index}]")
                continue
            gates = result.get("gates", {})
            if not isinstance(gates, dict) or gates.get("all_passed") is not True:
                failed.append(f"{path.name}:result[{index}]")
            members = result.get("member_results", [])
            if not isinstance(members, list):
                failed.append(f"{path.name}:result[{index}]:member-results")
                members = []
            for member_index, member in enumerate(members):
                if (
                    not isinstance(member, dict)
                    or not isinstance(member.get("gates"), dict)
                    or member["gates"].get("all_passed") is not True
                ):
                    failed.append(f"{path.name}:result[{index}]:member[{member_index}]")
            profile = result.get("profile")
            if not isinstance(profile, str) or not profile:
                failed.append(f"{path.name}:result[{index}]:missing-profile")
                continue
            profile_name = Path(profile).name
            profile_names.add(profile_name)
            profile_path = root / "profiles" / profile_name
            if not profile_path.is_file() or profile_path.stat().st_size == 0:
                failed.append(f"{path.name}:result[{index}]:profile-not-found")

    if failed:
        raise ValueError("failed gates: " + ", ".join(failed))
    if len(profile_names) != EXPECTED_PROFILE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_PROFILE_COUNT} unique profiles, "
            f"found {len(profile_names)}"
        )
    return len(documents), len(profile_names), tuple(sorted(commits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        document_count, profile_count, commits = validate_evidence(args.artifact_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(
        f"validated {document_count} benchmark documents and "
        f"{profile_count} unique profiles"
    )
    print(f"clean commits: {', '.join(commits)}")


if __name__ == "__main__":
    main()
