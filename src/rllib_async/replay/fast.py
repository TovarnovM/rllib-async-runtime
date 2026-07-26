"""Learner-local replay with reader-safe background index publication."""

from __future__ import annotations

import random
import threading
import time
from bisect import bisect_right
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rllib_async.protocols.episodes import (
    EpisodeCodec,
    EpisodeEnvelope,
    ModuleEpisodeCodec,
    MultiModuleTransition,
)
from rllib_async.protocols.replay import (
    ReplayCursor,
    ReplayDelta,
    ReplaySnapshot,
)
from rllib_async.replay.reference import (
    CursorMismatchError,
    FullResyncRequiredError,
    ReplayError,
)


class IndexRebuildError(ReplayError):
    """The background sampling-index rebuild failed."""


class ReplayClosedError(ReplayError):
    """The learner-local replay has been closed."""


@dataclass(frozen=True, slots=True)
class FastReplayStats:
    """A point-in-time snapshot of learner-local replay state."""

    cursor: ReplayCursor | None
    active_cursor: ReplayCursor | None
    episode_count: int
    total_transitions: int
    total_estimated_bytes: int
    active_producer_episode_counts: tuple[tuple[str, int], ...]
    active_producer_transition_counts: tuple[tuple[str, int], ...]
    active_module_transition_counts: tuple[tuple[str, int], ...]
    delta_lag_mutations: int
    delta_lag_agent_steps: int
    rebuild_in_progress: bool
    completed_rebuilds: int
    discarded_rebuilds: int
    rebuild_failures: int
    last_rebuild_ms: float
    rebuild_ms_total: float
    full_resyncs: int
    closed: bool


@dataclass(frozen=True, slots=True)
class _EpisodeRecord:
    episode: EpisodeEnvelope
    transition_count: int


@dataclass(frozen=True, slots=True)
class _TargetState:
    cursor: ReplayCursor
    records: Mapping[str, _EpisodeRecord]
    total_transitions: int
    total_estimated_bytes: int


@dataclass(frozen=True, slots=True)
class _SamplingIndex:
    episode_ids: tuple[str, ...]
    cumulative_lengths: tuple[int, ...]
    total_transitions: int


@dataclass(frozen=True, slots=True)
class _ModuleSamplingIndex:
    episode_indices: tuple[int, ...]
    cumulative_lengths: tuple[int, ...]
    total_transitions: int


@dataclass(frozen=True, slots=True)
class _MaterializedView:
    cursor: ReplayCursor
    episodes: tuple[EpisodeEnvelope, ...]
    sampling_index: _SamplingIndex
    module_sampling_indices: Mapping[str, _ModuleSamplingIndex]
    total_estimated_bytes: int


@dataclass(frozen=True, slots=True)
class _BuildRequest:
    revision: int
    cursor: ReplayCursor
    records: tuple[_EpisodeRecord, ...]
    total_transitions: int
    total_estimated_bytes: int


class FastReplay:
    """Learner-local payload store with an asynchronously rebuilt index.

    One writer validates snapshots and deltas against a logical target state.
    Readers capture a strong reference to one immutable active view. A
    background worker coalesces updates, builds the latest sampling index, and
    publishes it only if its revision is still current. Python object ownership
    therefore acts as the read lease: an old view and its evicted payloads stay
    alive until every in-flight sampling call releases its local reference.
    """

    def __init__(self, codec: EpisodeCodec) -> None:
        self._codec = codec
        self._writer_lock = threading.Lock()
        self._condition = threading.Condition()
        self._records: OrderedDict[str, _EpisodeRecord] = OrderedDict()
        self._target: _TargetState | None = None
        self._active_view: _MaterializedView | None = None
        self._target_revision = 0
        self._active_revision = 0
        self._delta_lag_agent_steps = 0
        self._rebuild_running = False
        self._rebuild_thread: threading.Thread | None = None
        self._rebuild_error: Exception | None = None
        self._completed_rebuilds = 0
        self._discarded_rebuilds = 0
        self._rebuild_failures = 0
        self._last_rebuild_ms = 0.0
        self._rebuild_ms_total = 0.0
        self._full_resyncs = 0
        self._closed = False

    @property
    def cursor(self) -> ReplayCursor | None:
        with self._condition:
            return self._target.cursor if self._target is not None else None

    @property
    def active_cursor(self) -> ReplayCursor | None:
        with self._condition:
            return self._active_view.cursor if self._active_view is not None else None

    @property
    def episode_ids(self) -> tuple[str, ...]:
        with self._condition:
            if self._target is None:
                return ()
            return tuple(self._target.records)

    @property
    def total_transitions(self) -> int:
        with self._condition:
            return self._target.total_transitions if self._target is not None else 0

    @property
    def total_estimated_bytes(self) -> int:
        with self._condition:
            return self._target.total_estimated_bytes if self._target is not None else 0

    @property
    def module_ids(self) -> tuple[str, ...]:
        self._module_codec()
        with self._condition:
            if self._active_view is None:
                return ()
            return tuple(self._active_view.module_sampling_indices)

    @property
    def module_transition_counts(self) -> tuple[tuple[str, int], ...]:
        self._module_codec()
        with self._condition:
            if self._active_view is None:
                return ()
            return tuple(
                (
                    module_id,
                    index.total_transitions,
                )
                for module_id, index in (
                    self._active_view.module_sampling_indices.items()
                )
            )

    def load_snapshot(self, snapshot: ReplaySnapshot) -> None:
        """Validate and synchronously publish a complete bootstrap/resync."""
        with self._writer_lock:
            with self._condition:
                self._ensure_open_locked()

            target, records = self._target_from_snapshot(snapshot)
            request = _BuildRequest(
                revision=0,
                cursor=target.cursor,
                records=tuple(target.records.values()),
                total_transitions=target.total_transitions,
                total_estimated_bytes=target.total_estimated_bytes,
            )
            view = self._build_view(request)

            with self._condition:
                self._ensure_open_locked()
                self._target_revision += 1
                self._active_revision = self._target_revision
                self._records = records
                self._target = target
                self._active_view = view
                self._delta_lag_agent_steps = 0
                self._rebuild_error = None
                self._condition.notify_all()

    def apply_delta(self, delta: ReplayDelta) -> None:
        """Validate a delta and enqueue its index revision for publication."""
        with self._writer_lock:
            with self._condition:
                self._ensure_open_locked()
                if delta.full_resync_required:
                    self._full_resyncs += 1
                    raise FullResyncRequiredError(
                        "authoritative replay requires a snapshot"
                    )
                if self._target is None:
                    raise CursorMismatchError("load a snapshot before applying deltas")
                target = self._target

            if delta.base_cursor != target.cursor:
                raise CursorMismatchError(
                    f"local cursor {target.cursor!r} does not match "
                    f"delta base {delta.base_cursor!r}"
                )

            manifest_ids = deque(target.records)
            present_ids = set(manifest_ids)
            added_records: dict[str, _EpisodeRecord] = {}
            total_transitions = target.total_transitions
            total_estimated_bytes = target.total_estimated_bytes
            added_agent_steps = 0
            expected_mutation_seq = target.cursor.mutation_seq
            for transaction in delta.transactions:
                expected_mutation_seq += 1
                if transaction.mutation_seq != expected_mutation_seq:
                    raise CursorMismatchError(
                        f"expected mutation {expected_mutation_seq}, "
                        f"got {transaction.mutation_seq}"
                    )
                self._codec.validate(transaction.added)
                episode_id = transaction.added.episode_id
                if episode_id in present_ids:
                    raise ReplayError(f"delta adds existing episode_id {episode_id!r}")
                transition_count = self._codec.transition_count(transaction.added)
                record = _EpisodeRecord(transaction.added, transition_count)
                added_records[episode_id] = record
                manifest_ids.append(episode_id)
                present_ids.add(episode_id)
                total_transitions += transition_count
                total_estimated_bytes += transaction.added.estimated_bytes
                added_agent_steps += transaction.added.agent_steps

                for evicted_id in transaction.evicted_episode_ids:
                    if evicted_id not in present_ids:
                        raise ReplayError(
                            f"delta evicts unknown episode_id {evicted_id!r}"
                        )
                    if manifest_ids[0] != evicted_id:
                        raise ReplayError(
                            f"delta transaction {transaction.mutation_seq} "
                            "does not evict FIFO"
                        )
                    manifest_ids.popleft()
                    present_ids.remove(evicted_id)
                    evicted = added_records.get(evicted_id)
                    if evicted is None:
                        evicted = target.records[evicted_id]
                    total_transitions -= evicted.transition_count
                    total_estimated_bytes -= evicted.episode.estimated_bytes

            if delta.next_cursor.store_generation != target.cursor.store_generation:
                raise CursorMismatchError("delta changes store generation")
            if delta.next_cursor.mutation_seq != expected_mutation_seq:
                raise CursorMismatchError(
                    "delta next cursor does not match its transaction suffix"
                )
            if not delta.transactions:
                return

            with self._condition:
                self._ensure_open_locked()
                for transaction in delta.transactions:
                    episode_id = transaction.added.episode_id
                    self._records[episode_id] = added_records[episode_id]
                    for evicted_id in transaction.evicted_episode_ids:
                        del self._records[evicted_id]
                candidate = _TargetState(
                    cursor=delta.next_cursor,
                    records=MappingProxyType(self._records),
                    total_transitions=total_transitions,
                    total_estimated_bytes=total_estimated_bytes,
                )
                self._target_revision += 1
                self._target = candidate
                self._delta_lag_agent_steps += added_agent_steps
                self._rebuild_error = None
                self._schedule_rebuild_locked()
                self._condition.notify_all()

    def apply_deltas(self, deltas: Sequence[ReplayDelta]) -> None:
        for delta in deltas:
            self.apply_delta(delta)

    def get_snapshot(self) -> ReplaySnapshot:
        with self._condition:
            self._ensure_open_locked()
            if self._target is None:
                raise ReplayError("load a snapshot before reading the local view")
            target = self._target
            return ReplaySnapshot(
                cursor=target.cursor,
                episodes=tuple(record.episode for record in target.records.values()),
                total_transitions=target.total_transitions,
                total_estimated_bytes=target.total_estimated_bytes,
            )

    def get_stats(self) -> FastReplayStats:
        with self._condition:
            target = self._target
            active = self._active_view
            lag_mutations = self._delta_lag_mutations_locked(target, active)
            producer_episode_counts: Counter[str] = Counter()
            producer_transition_counts: Counter[str] = Counter()
            if active is not None:
                previous_cumulative_length = 0
                for episode, cumulative_length in zip(
                    active.episodes,
                    active.sampling_index.cumulative_lengths,
                    strict=True,
                ):
                    transition_count = cumulative_length - previous_cumulative_length
                    previous_cumulative_length = cumulative_length
                    producer_episode_counts[episode.producer_member_id] += 1
                    producer_transition_counts[episode.producer_member_id] += (
                        transition_count
                    )
            return FastReplayStats(
                cursor=target.cursor if target is not None else None,
                active_cursor=active.cursor if active is not None else None,
                episode_count=len(target.records) if target is not None else 0,
                total_transitions=(
                    target.total_transitions if target is not None else 0
                ),
                total_estimated_bytes=(
                    target.total_estimated_bytes if target is not None else 0
                ),
                active_producer_episode_counts=tuple(
                    sorted(producer_episode_counts.items())
                ),
                active_producer_transition_counts=tuple(
                    sorted(producer_transition_counts.items())
                ),
                active_module_transition_counts=(
                    ()
                    if active is None
                    else tuple(
                        (
                            module_id,
                            index.total_transitions,
                        )
                        for module_id, index in (active.module_sampling_indices.items())
                    )
                ),
                delta_lag_mutations=lag_mutations,
                delta_lag_agent_steps=self._delta_lag_agent_steps,
                rebuild_in_progress=self._rebuild_running,
                completed_rebuilds=self._completed_rebuilds,
                discarded_rebuilds=self._discarded_rebuilds,
                rebuild_failures=self._rebuild_failures,
                last_rebuild_ms=self._last_rebuild_ms,
                rebuild_ms_total=self._rebuild_ms_total,
                full_resyncs=self._full_resyncs,
                closed=self._closed,
            )

    def wait_for_idle(self, timeout: float | None = None) -> None:
        """Wait until the active sampling view reaches the accepted cursor."""
        deadline = self._deadline(timeout)
        with self._condition:
            while self._active_revision != self._target_revision:
                if self._rebuild_error is not None and not self._rebuild_running:
                    error = self._rebuild_error
                    raise IndexRebuildError(
                        f"background index rebuild failed: {error}"
                    ) from error
                if self._closed and not self._rebuild_running:
                    raise ReplayClosedError(
                        "replay closed before the target index was published"
                    )
                remaining = self._remaining(deadline)
                if remaining == 0:
                    raise TimeoutError("timed out waiting for replay index rebuild")
                self._condition.wait(remaining)

    def wait_for_transitions(
        self,
        minimum: int = 1,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Wait until the active sampling view contains enough transitions."""
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            raise ValueError("minimum must be a positive integer")
        deadline = self._deadline(timeout)
        with self._condition:
            while True:
                if self._rebuild_error is not None and not self._rebuild_running:
                    error = self._rebuild_error
                    raise IndexRebuildError(
                        f"background index rebuild failed: {error}"
                    ) from error
                if (
                    self._active_view is not None
                    and self._active_view.sampling_index.total_transitions >= minimum
                ):
                    return True
                if self._closed:
                    return False
                remaining = self._remaining(deadline)
                if remaining == 0:
                    return False
                self._condition.wait(remaining)

    def wait_for_module_transitions(
        self,
        module_id: str,
        minimum: int = 1,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Wait until one module-specific active view contains enough transitions."""

        self._module_codec()
        if not isinstance(module_id, str) or not module_id:
            raise ValueError("module_id must be a non-empty string")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            raise ValueError("minimum must be a positive integer")
        deadline = self._deadline(timeout)
        with self._condition:
            while True:
                if self._rebuild_error is not None and not self._rebuild_running:
                    error = self._rebuild_error
                    raise IndexRebuildError(
                        f"background index rebuild failed: {error}"
                    ) from error
                if self._active_view is not None:
                    index = self._active_view.module_sampling_indices.get(module_id)
                    if index is not None and index.total_transitions >= minimum:
                        return True
                if self._closed:
                    return False
                remaining = self._remaining(deadline)
                if remaining == 0:
                    return False
                self._condition.wait(remaining)

    def sample_coordinates(
        self,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[str, int]]:
        view = self._capture_active_view()
        return [
            (view.sampling_index.episode_ids[episode_index], transition_index)
            for episode_index, transition_index in self._sample_positions(
                view,
                batch_size,
                rng=rng,
            )
        ]

    def sample(
        self,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[object]:
        view = self._capture_active_view()
        return [
            self._codec.get_transition(
                view.episodes[episode_index],
                transition_index,
            )
            for episode_index, transition_index in self._sample_positions(
                view,
                batch_size,
                rng=rng,
            )
        ]

    def sample_module_coordinates(
        self,
        module_id: str,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[str, int]]:
        self._module_codec()
        view = self._capture_active_view()
        return [
            (
                view.episodes[episode_index].episode_id,
                transition_index,
            )
            for episode_index, transition_index in self._sample_module_positions(
                view,
                module_id,
                batch_size,
                rng=rng,
            )
        ]

    def sample_module(
        self,
        module_id: str,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[MultiModuleTransition]:
        codec = self._module_codec()
        view = self._capture_active_view()
        return [
            codec.get_module_transition(
                view.episodes[episode_index],
                module_id,
                transition_index,
            )
            for episode_index, transition_index in self._sample_module_positions(
                view,
                module_id,
                batch_size,
                rng=rng,
            )
        ]

    def close(
        self,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Stop background publication, optionally draining the latest build."""
        deadline = self._deadline(timeout)
        wait_error: BaseException | None = None
        with self._writer_lock:
            with self._condition:
                if self._closed:
                    return
            if wait:
                try:
                    self.wait_for_idle(timeout=self._remaining(deadline))
                except BaseException as error:
                    wait_error = error
            with self._condition:
                self._closed = True
                thread = self._rebuild_thread
                self._condition.notify_all()

        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(self._remaining(deadline))
            if thread.is_alive() and wait_error is None:
                wait_error = TimeoutError(
                    "timed out waiting for replay rebuild worker shutdown"
                )
        if wait_error is not None:
            raise wait_error

    def __enter__(self) -> FastReplay:
        with self._condition:
            self._ensure_open_locked()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _target_from_snapshot(
        self,
        snapshot: ReplaySnapshot,
    ) -> tuple[_TargetState, OrderedDict[str, _EpisodeRecord]]:
        records: OrderedDict[str, _EpisodeRecord] = OrderedDict()
        total_transitions = 0
        total_estimated_bytes = 0
        for episode in snapshot.episodes:
            self._codec.validate(episode)
            if episode.episode_id in records:
                raise ReplayError(
                    "materialized view contains duplicate episode_id "
                    f"{episode.episode_id!r}"
                )
            transition_count = self._codec.transition_count(episode)
            records[episode.episode_id] = _EpisodeRecord(
                episode,
                transition_count,
            )
            total_transitions += transition_count
            total_estimated_bytes += episode.estimated_bytes

        if total_transitions != snapshot.total_transitions:
            raise ReplayError("snapshot transition total is inconsistent")
        if total_estimated_bytes != snapshot.total_estimated_bytes:
            raise ReplayError("snapshot byte total is inconsistent")
        return (
            _TargetState(
                cursor=snapshot.cursor,
                records=MappingProxyType(records),
                total_transitions=total_transitions,
                total_estimated_bytes=total_estimated_bytes,
            ),
            records,
        )

    def _build_view(self, request: _BuildRequest) -> _MaterializedView:
        episode_ids: list[str] = []
        episodes: list[EpisodeEnvelope] = []
        cumulative_lengths: list[int] = []
        total_transitions = 0
        module_episode_indices: dict[str, list[int]] = {}
        module_cumulative_lengths: dict[str, list[int]] = {}
        module_totals: dict[str, int] = {}
        module_codec = (
            self._codec if isinstance(self._codec, ModuleEpisodeCodec) else None
        )

        for episode_index, record in enumerate(request.records):
            episodes.append(record.episode)
            episode_ids.append(record.episode.episode_id)
            total_transitions += record.transition_count
            cumulative_lengths.append(total_transitions)
            if module_codec is not None:
                episode_module_total = 0
                for module_id in module_codec.module_ids(record.episode):
                    module_count = module_codec.module_transition_count(
                        record.episode,
                        module_id,
                    )
                    if module_count < 1:
                        raise ReplayError(
                            "module-specific episode views cannot be empty"
                        )
                    episode_module_total += module_count
                    module_episode_indices.setdefault(module_id, []).append(
                        episode_index
                    )
                    module_totals[module_id] = (
                        module_totals.get(module_id, 0) + module_count
                    )
                    module_cumulative_lengths.setdefault(module_id, []).append(
                        module_totals[module_id]
                    )
                if episode_module_total != record.transition_count:
                    raise ReplayError(
                        "module transition totals do not match the episode total"
                    )

        if total_transitions != request.total_transitions:
            raise ReplayError("sampling index transition total is inconsistent")
        module_sampling_indices = {
            module_id: _ModuleSamplingIndex(
                episode_indices=tuple(module_episode_indices[module_id]),
                cumulative_lengths=tuple(module_cumulative_lengths[module_id]),
                total_transitions=module_totals[module_id],
            )
            for module_id in sorted(module_totals)
        }
        return _MaterializedView(
            cursor=request.cursor,
            episodes=tuple(episodes),
            sampling_index=_SamplingIndex(
                episode_ids=tuple(episode_ids),
                cumulative_lengths=tuple(cumulative_lengths),
                total_transitions=total_transitions,
            ),
            module_sampling_indices=MappingProxyType(module_sampling_indices),
            total_estimated_bytes=request.total_estimated_bytes,
        )

    def _schedule_rebuild_locked(self) -> None:
        if (
            self._closed
            or self._rebuild_running
            or self._active_revision == self._target_revision
        ):
            return
        self._rebuild_running = True
        thread = threading.Thread(
            target=self._rebuild_loop,
            name=f"fast-replay-rebuild-{id(self):x}",
            daemon=True,
        )
        self._rebuild_thread = thread
        try:
            thread.start()
        except Exception as error:
            self._rebuild_running = False
            self._rebuild_thread = None
            self._rebuild_error = error
            self._rebuild_failures += 1
            self._condition.notify_all()

    def _rebuild_loop(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    self._rebuild_running = False
                    self._condition.notify_all()
                    return
                if self._active_revision == self._target_revision:
                    self._rebuild_running = False
                    self._condition.notify_all()
                    return
                assert self._target is not None
                request = _BuildRequest(
                    revision=self._target_revision,
                    cursor=self._target.cursor,
                    records=tuple(self._target.records.values()),
                    total_transitions=self._target.total_transitions,
                    total_estimated_bytes=self._target.total_estimated_bytes,
                )

            started = time.perf_counter()
            try:
                view = self._build_view(request)
            except Exception as error:
                elapsed_ms = (time.perf_counter() - started) * 1_000
                with self._condition:
                    self._rebuild_failures += 1
                    self._rebuild_ms_total += elapsed_ms
                    if request.revision != self._target_revision:
                        continue
                    self._rebuild_error = error
                    self._rebuild_running = False
                    self._condition.notify_all()
                    return

            elapsed_ms = (time.perf_counter() - started) * 1_000
            with self._condition:
                self._rebuild_ms_total += elapsed_ms
                if self._closed:
                    self._rebuild_running = False
                    self._condition.notify_all()
                    return
                if request.revision != self._target_revision:
                    self._discarded_rebuilds += 1
                    continue
                self._active_view = view
                self._active_revision = request.revision
                self._delta_lag_agent_steps = 0
                self._rebuild_error = None
                self._completed_rebuilds += 1
                self._last_rebuild_ms = elapsed_ms
                self._rebuild_running = False
                self._condition.notify_all()
                return

    def _capture_active_view(self) -> _MaterializedView:
        with self._condition:
            self._ensure_open_locked()
            if self._active_view is None:
                raise ReplayError("load a snapshot before reading the local view")
            return self._active_view

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise ReplayClosedError("learner-local replay is closed")

    @staticmethod
    def _delta_lag_mutations_locked(
        target: _TargetState | None,
        active: _MaterializedView | None,
    ) -> int:
        if target is None or active is None:
            return 0
        if target.cursor.store_generation != active.cursor.store_generation:
            return target.cursor.mutation_seq
        return max(0, target.cursor.mutation_seq - active.cursor.mutation_seq)

    @staticmethod
    def _sample_positions(
        view: _MaterializedView,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        index = view.sampling_index
        if index.total_transitions == 0:
            raise ReplayError("cannot sample an empty replay")

        positions: list[tuple[int, int]] = []
        for _ in range(batch_size):
            flat_index = rng.randrange(index.total_transitions)
            episode_index = bisect_right(index.cumulative_lengths, flat_index)
            episode_start = (
                index.cumulative_lengths[episode_index - 1] if episode_index else 0
            )
            positions.append((episode_index, flat_index - episode_start))
        return positions

    @staticmethod
    def _sample_module_positions(
        view: _MaterializedView,
        module_id: str,
        batch_size: int,
        *,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        if not isinstance(module_id, str) or not module_id:
            raise ValueError("module_id must be a non-empty string")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        index = view.module_sampling_indices.get(module_id)
        if index is None:
            raise KeyError(module_id)
        positions: list[tuple[int, int]] = []
        for _ in range(batch_size):
            flat_index = rng.randrange(index.total_transitions)
            module_episode_index = bisect_right(
                index.cumulative_lengths,
                flat_index,
            )
            episode_start = (
                index.cumulative_lengths[module_episode_index - 1]
                if module_episode_index
                else 0
            )
            positions.append(
                (
                    index.episode_indices[module_episode_index],
                    flat_index - episode_start,
                )
            )
        return positions

    def _module_codec(self) -> ModuleEpisodeCodec:
        if not isinstance(self._codec, ModuleEpisodeCodec):
            raise TypeError("the configured episode codec has no module-specific views")
        return self._codec

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        return time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
