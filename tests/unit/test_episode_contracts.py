from __future__ import annotations

import pickle
from dataclasses import replace

import pytest

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    EpisodeValidationError,
    FlatEpisodeCodec,
    FrozenVersions,
    SchemaMismatchError,
)


def make_episode(
    codec: FlatEpisodeCodec,
    episode_id: str = "member-0/runner-0/0/0",
    transitions: list[object] | None = None,
) -> EpisodeEnvelope:
    payload = codec.encode(transitions or [{"step": 0}, {"step": 1}])
    count = len(payload.encoded_transitions)
    return EpisodeEnvelope(
        episode_id=episode_id,
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=0,
        behavior_versions=FrozenVersions({"default_policy": 3}),
        env_steps=count,
        agent_steps=count,
        terminated=True,
        truncated=False,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )


def test_flat_payload_and_versions_are_immutable_and_pickle_safe() -> None:
    codec = FlatEpisodeCodec()
    source_transition = {"values": [1, 2]}
    source_versions = {"default_policy": 7}
    payload = codec.encode([source_transition])
    episode = EpisodeEnvelope(
        episode_id="member-0/runner-0/0/0",
        schema_version=codec.schema_version,
        producer_member_id="member-0",
        runner_id="runner-0",
        runner_generation=0,
        local_episode_seq=0,
        behavior_versions=source_versions,
        env_steps=1,
        agent_steps=1,
        terminated=False,
        truncated=True,
        estimated_bytes=payload.estimated_bytes,
        payload=payload,
    )

    source_transition["values"].append(3)
    source_versions["default_policy"] = 99

    assert codec.get_transition(episode, 0) == {"values": [1, 2]}
    assert dict(episode.behavior_versions) == {"default_policy": 7}
    assert pickle.loads(pickle.dumps(episode)) == episode


def test_codec_rejects_schema_and_metadata_mismatches() -> None:
    codec = FlatEpisodeCodec()
    episode = make_episode(codec)

    with pytest.raises(SchemaMismatchError, match="expected schema 1"):
        codec.validate(replace(episode, schema_version=2))
    with pytest.raises(EpisodeValidationError, match="transition count"):
        codec.validate(replace(episode, agent_steps=episode.agent_steps + 1))
    with pytest.raises(EpisodeValidationError, match="codec estimate"):
        codec.validate(replace(episode, estimated_bytes=episode.estimated_bytes + 1))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_id", "", "episode_id"),
        ("runner_generation", -1, "runner_generation"),
        ("env_steps", 0, "env_steps"),
        ("agent_steps", 0, "agent_steps"),
        ("estimated_bytes", 0, "estimated_bytes"),
    ],
)
def test_envelope_rejects_invalid_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    codec = FlatEpisodeCodec()
    episode = make_episode(codec)

    with pytest.raises(EpisodeValidationError, match=message):
        replace(episode, **{field: value})


def test_envelope_requires_a_completed_episode_and_behavior_version() -> None:
    codec = FlatEpisodeCodec()
    episode = make_episode(codec)

    with pytest.raises(EpisodeValidationError, match="terminated or truncated"):
        replace(episode, terminated=False, truncated=False)
    with pytest.raises(EpisodeValidationError, match="must not be empty"):
        replace(episode, behavior_versions=FrozenVersions({}))
