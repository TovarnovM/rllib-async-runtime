"""Validated configuration for one asynchronous SAC member."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from ray.rllib.algorithms.sac import SACConfig


@dataclass(frozen=True, slots=True)
class SharedReplayDescriptor:
    """Serializable lookup information for one named population replay actor."""

    actor_name: str
    namespace: str

    def __post_init__(self) -> None:
        for name, value in (
            ("actor_name", self.actor_name),
            ("namespace", self.namespace),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"shared replay {name} must be a non-empty string")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | SharedReplayDescriptor | None,
    ) -> SharedReplayDescriptor | None:
        if values is None or isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise TypeError("shared replay descriptor must be a mapping")
        unknown = set(values) - {"actor_name", "namespace"}
        if unknown:
            raise ValueError(f"unknown shared replay settings {sorted(unknown)!r}")
        try:
            return cls(
                actor_name=values["actor_name"],
                namespace=values["namespace"],
            )
        except KeyError as error:
            raise ValueError(
                "shared replay descriptor requires actor_name and namespace"
            ) from error


@dataclass(frozen=True, slots=True)
class AsyncSACRuntimeConfig:
    """Runtime-only settings kept separate from RLlib's SAC configuration."""

    member_id: str = "member-0"
    runner_count: int = 4
    max_episode_steps: int = 200
    replay_capacity_transitions: int = 100_000
    replay_capacity_bytes: int = 512 * 1024 * 1024
    replay_journal_capacity: int = 4_096
    replay_sync_max_bytes: int = 16 * 1024 * 1024
    pending_commit_high_watermark: int = 16
    pending_commit_low_watermark: int = 8
    batch_size: int = 256
    batch_queue_capacity: int = 4
    learner_updates_per_tick: int = 1
    publication_interval_updates: int = 1
    evaluation_interval_env_steps: int = 5_000
    evaluation_num_episodes: int = 4
    report_interval_s: float = 1.0
    event_poll_timeout_s: float = 0.01
    shutdown_timeout_s: float = 30.0
    seed: int = 0
    num_cpus_per_replay: float = 1.0
    num_cpus_per_runner: float = 1.0
    num_cpus_per_evaluation_runner: float = 1.0
    num_cpus_per_learner: float = 1.0
    num_gpus_per_learner: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.member_id, str)
            or not self.member_id
            or self.member_id in {".", ".."}
            or "/" in self.member_id
            or "\\" in self.member_id
        ):
            raise ValueError("member_id must be a non-empty path segment")
        if (
            not isinstance(self.runner_count, int)
            or isinstance(self.runner_count, bool)
            or not 1 <= self.runner_count <= 16
        ):
            raise ValueError("runner_count must be between 1 and 16")
        for name in (
            "max_episode_steps",
            "replay_capacity_transitions",
            "replay_capacity_bytes",
            "replay_journal_capacity",
            "replay_sync_max_bytes",
            "pending_commit_high_watermark",
            "batch_size",
            "learner_updates_per_tick",
            "publication_interval_updates",
        ):
            self._positive_int(name, getattr(self, name))
        if (
            not isinstance(self.batch_queue_capacity, int)
            or isinstance(self.batch_queue_capacity, bool)
            or self.batch_queue_capacity < 0
        ):
            raise ValueError("batch_queue_capacity must be a non-negative integer")
        if (
            not isinstance(self.pending_commit_low_watermark, int)
            or isinstance(self.pending_commit_low_watermark, bool)
            or self.pending_commit_low_watermark < 0
            or self.pending_commit_low_watermark >= self.pending_commit_high_watermark
        ):
            raise ValueError(
                "pending_commit_low_watermark must be non-negative and below high"
            )
        if (
            not isinstance(self.evaluation_interval_env_steps, int)
            or isinstance(self.evaluation_interval_env_steps, bool)
            or self.evaluation_interval_env_steps < 0
        ):
            raise ValueError(
                "evaluation_interval_env_steps must be a non-negative integer"
            )
        if (
            not isinstance(self.evaluation_num_episodes, int)
            or isinstance(self.evaluation_num_episodes, bool)
            or not 0 <= self.evaluation_num_episodes <= 16
        ):
            raise ValueError("evaluation_num_episodes must be between 0 and 16")
        if (self.evaluation_interval_env_steps == 0) != (
            self.evaluation_num_episodes == 0
        ):
            raise ValueError(
                "evaluation interval and episode count must both be zero or positive"
            )
        for name in (
            "report_interval_s",
            "shutdown_timeout_s",
        ):
            self._positive_float(name, getattr(self, name))
        if (
            not isinstance(self.event_poll_timeout_s, int | float)
            or isinstance(self.event_poll_timeout_s, bool)
            or not math.isfinite(self.event_poll_timeout_s)
            or self.event_poll_timeout_s < 0
        ):
            raise ValueError("event_poll_timeout_s must be finite and non-negative")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        for name in (
            "num_cpus_per_replay",
            "num_cpus_per_runner",
            "num_cpus_per_evaluation_runner",
            "num_cpus_per_learner",
            "num_gpus_per_learner",
        ):
            self._non_negative_float(name, getattr(self, name))
        if self.num_gpus_per_learner not in {0.0, 1.0}:
            raise ValueError("num_gpus_per_learner must be 0 or 1")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        sac_config: SACConfig,
    ) -> AsyncSACRuntimeConfig:
        """Build a strict runtime config, defaulting batch size from SAC."""

        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("runtime config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown runtime settings {sorted(unknown)!r}")
        resolved = dict(values)
        resolved.setdefault(
            "batch_size",
            int(sac_config.train_batch_size_per_learner),
        )
        resolved.setdefault(
            "num_gpus_per_learner",
            float(sac_config.num_gpus_per_learner),
        )
        return cls(**resolved)

    @staticmethod
    def _positive_int(name: str, value: object) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _positive_float(name: str, value: object) -> None:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _non_negative_float(name: str, value: object) -> None:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
