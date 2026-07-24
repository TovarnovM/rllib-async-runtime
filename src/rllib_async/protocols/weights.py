"""Versioned inference-weight publication contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from rllib_async.protocols.episodes import FrozenVersions


@dataclass(frozen=True, slots=True)
class WeightsDescriptor:
    """One immutable-by-contract inference-weight publication."""

    member_id: str
    module_versions: Mapping[str, int]
    learner_updates: int
    published_at_monotonic: float
    state: object

    def __post_init__(self) -> None:
        if not isinstance(self.member_id, str) or not self.member_id:
            raise ValueError("member_id must be a non-empty string")
        versions = self.module_versions
        if not isinstance(versions, FrozenVersions):
            try:
                versions = FrozenVersions(versions)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "module_versions must map module IDs to non-negative versions"
                ) from error
            object.__setattr__(self, "module_versions", versions)
        if not versions:
            raise ValueError("module_versions must not be empty")
        if (
            not isinstance(self.learner_updates, int)
            or isinstance(self.learner_updates, bool)
            or self.learner_updates < 0
        ):
            raise ValueError("learner_updates must be a non-negative integer")
        if (
            not isinstance(self.published_at_monotonic, int | float)
            or isinstance(self.published_at_monotonic, bool)
            or not math.isfinite(self.published_at_monotonic)
            or self.published_at_monotonic < 0
        ):
            raise ValueError("published_at_monotonic must be finite and non-negative")
