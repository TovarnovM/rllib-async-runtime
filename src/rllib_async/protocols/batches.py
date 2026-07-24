"""Learner-local batch construction contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

BatchT = TypeVar("BatchT")


@runtime_checkable
class BatchCollator(Protocol[BatchT]):
    """Convert sampled transitions into one learner-consumable CPU batch."""

    def collate(self, transitions: Sequence[object]) -> BatchT: ...
