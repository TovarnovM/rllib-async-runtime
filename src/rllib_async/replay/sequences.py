"""Learner-local windows over raw episodes and physical-time target preparation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rllib_async.protocols.episodes import (
    EpisodeEnvelope,
    MultiModuleTransition,
    SequenceEpisodeCodec,
)


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """None burn-in replays the full prefix; bounded burn-in is approximate."""

    learning_length: int = 1
    burn_in: int | None = None

    def __post_init__(self) -> None:
        _positive_int("learning_length", self.learning_length)
        if self.burn_in is not None and (
            type(self.burn_in) is not int or self.burn_in < 0
        ):
            raise ValueError("burn_in must be None or a non-negative integer")


@dataclass(frozen=True, slots=True)
class SequenceWindow:
    """A sampled agent history; the envelope reference is its read lease."""

    episode: EpisodeEnvelope
    module_id: str
    agent_id: str
    learn_start: int
    prefix: tuple[MultiModuleTransition, ...]
    learning: tuple[MultiModuleTransition, ...]
    continuation: tuple[MultiModuleTransition, ...]
    approximate: bool
    target_horizon: int = 1

    @property
    def handle(self) -> tuple[str, str, int]:
        return self.episode.episode_id, self.agent_id, self.learn_start

    @property
    def valid_count(self) -> int:
        return len(self.learning)


@dataclass(frozen=True, slots=True)
class PreparedSequence:
    observations: tuple[object, ...]
    prefix_length: int
    current_indices: tuple[int, ...]
    next_indices: tuple[int, ...]
    actions: tuple[object, ...]
    rewards: tuple[float, ...]
    terminateds: tuple[bool, ...]
    truncateds: tuple[bool, ...]
    n_step: tuple[int, ...]
    decision_horizons: tuple[int, ...]
    final_terminated: bool


def make_sequence_window(
    codec: SequenceEpisodeCodec,
    episode: EpisodeEnvelope,
    module_id: str,
    history: tuple[int, ...],
    start: int,
    *,
    sequence_config: SequenceConfig,
    n_step: int,
) -> SequenceWindow:
    """Decode only the selected window, without altering the retained payload."""

    _positive_int("n_step", n_step)
    prefix_start = (
        0 if sequence_config.burn_in is None
        else max(0, start - sequence_config.burn_in)
    )
    learning_end = min(len(history), start + sequence_config.learning_length)
    continuation_end = min(len(history), learning_end + n_step - 1)
    rows = tuple(
        codec.get_module_transition(episode, module_id, index, validate=False)
        for index in history[prefix_start:continuation_end]
    )
    prefix_length = start - prefix_start
    learning_length = learning_end - start
    first = rows[prefix_length]
    return SequenceWindow(
        episode=episode,
        module_id=module_id,
        agent_id=first.agent_id,
        learn_start=first.agent_t,
        prefix=rows[:prefix_length],
        learning=rows[prefix_length:prefix_length + learning_length],
        continuation=rows[prefix_length + learning_length:],
        approximate=sequence_config.burn_in is not None,
        target_horizon=n_step,
    )


def prepare_sequence(
    window: SequenceWindow, *, gamma: float, n_step: int = 1
) -> PreparedSequence:
    """Compute SMDP returns from raw ticks; never discount the stored replay."""

    _positive_int("n_step", n_step)
    if n_step != window.target_horizon:
        raise ValueError("target horizon differs from the sampled sequence continuation")
    if isinstance(gamma, bool) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and between zero and one")
    rows = window.prefix + window.learning + window.continuation
    if not window.learning:
        raise ValueError("a sequence must contain at least one learning decision")
    prefix_length = len(window.prefix)
    current_indices = tuple(range(prefix_length, prefix_length + len(window.learning)))
    rewards, exponents, horizons, terminals, truncations, next_indices = (
        [], [], [], [], [], []
    )
    for start in current_indices:
        total_reward = 0.0
        elapsed = 0
        horizon = 0
        for index in range(start, min(len(rows), start + n_step)):
            data = rows[index].data
            for offset, reward in enumerate(data["raw_rewards"]):
                total_reward += gamma ** (elapsed + offset) * reward
            elapsed += data["duration"]
            horizon += 1
            terminal = data["terminateds"] or data["subtask_terminated"]
            truncated = data["truncateds"]
            if terminal or truncated:
                break
        rewards.append(total_reward)
        exponents.append(elapsed)
        horizons.append(horizon)
        terminals.append(bool(terminal))
        truncations.append(bool(truncated))
        next_indices.append(start + horizon)
    last = rows[-1].data
    return PreparedSequence(
        observations=tuple(row.data["obs"] for row in rows) + (last["new_obs"],),
        prefix_length=prefix_length,
        current_indices=current_indices,
        next_indices=tuple(next_indices),
        actions=tuple(row.data["actions"] for row in window.learning),
        rewards=tuple(rewards),
        terminateds=tuple(terminals),
        truncateds=tuple(truncations),
        n_step=tuple(exponents),
        decision_horizons=tuple(horizons),
        final_terminated=bool(last["terminateds"] or last["subtask_terminated"]),
    )


def collate_sequences(
    windows: Sequence[SequenceWindow],
    *,
    gamma: float,
    n_step: int = 1,
    validate: bool = False,
) -> dict[str, Any]:
    """Return only valid loss rows plus ragged histories for recurrent encoders.

    ``n_step`` in the resulting batch is the physical discount exponent expected
    by RLlib; ``decision_horizons`` retains the separate sampling-horizon meaning.
    There is no padding and neither prefix nor continuation gets a loss row.
    """

    if not windows:
        raise ValueError("sequence collation requires at least one window")
    if len({window.module_id for window in windows}) != 1:
        raise ValueError("sequence batches cannot mix module semantics")
    prepared = [prepare_sequence(window, gamma=gamma, n_step=n_step) for window in windows]
    observations: list[object] = []
    current, following, lengths, prefixes, history_terminals = [], [], [], [], []
    for sequence in prepared:
        offset = len(observations)
        observations.extend(sequence.observations)
        current.extend(offset + index for index in sequence.current_indices)
        following.extend(offset + index for index in sequence.next_indices)
        lengths.append(len(sequence.observations))
        prefixes.append(sequence.prefix_length)
        history_terminals.extend([False] * (len(sequence.observations) - 1))
        history_terminals.append(sequence.final_terminated)
    terminals = np.asarray(
        [value for sequence in prepared for value in sequence.terminateds], dtype=np.bool_
    )
    batch: dict[str, Any] = {
        "obs": _stack_observations(
            [observations[index] for index in current], validate=validate
        ),
        "new_obs": _stack_observations(
            [observations[index] for index in following],
            validate=validate, terminated=terminals,
        ),
        "actions": np.ascontiguousarray(np.stack(
            [action for sequence in prepared for action in sequence.actions]
        )),
        "rewards": np.asarray(
            [value for sequence in prepared for value in sequence.rewards], dtype=np.float32
        ),
        "terminateds": terminals,
        "truncateds": np.asarray(
            [value for sequence in prepared for value in sequence.truncateds], dtype=np.bool_
        ),
        "n_step": np.asarray(
            [value for sequence in prepared for value in sequence.n_step], dtype=np.int64
        ),
        "decision_horizons": np.asarray(
            [value for sequence in prepared for value in sequence.decision_horizons],
            dtype=np.int64,
        ),
        "weights": np.ones(len(current), dtype=np.float32),
        "recurrent": {
            "obs": _stack_observations(
                observations, validate=validate, terminated=history_terminals
            ),
            "seq_lens": np.asarray(lengths, dtype=np.int64),
            "prefix_lens": np.asarray(prefixes, dtype=np.int64),
            "current_indices": np.asarray(current, dtype=np.int64),
            "next_indices": np.asarray(following, dtype=np.int64),
        },
    }
    return batch


class SequenceBatchCollator:
    """One consumer's discounting and optional first successful batch check."""

    def __init__(
        self, *, gamma: float, n_step: int = 1, validate_startup: bool = False
    ) -> None:
        self._gamma = gamma
        self._n_step = n_step
        self._validate_next_batch = validate_startup

    def collate(self, windows: Sequence[SequenceWindow]) -> dict[str, Any]:
        batch = collate_sequences(
            windows, gamma=self._gamma, n_step=self._n_step,
            validate=self._validate_next_batch,
        )
        self._validate_next_batch = False
        return batch


def _stack_observations(
    observations: Sequence[object], *, validate: bool,
    terminated: Sequence[bool] | None = None,
) -> object:
    first = observations[0]
    if isinstance(first, Mapping):
        if "node_features" in first:
            # Import lazily: graph batching itself uses the ordinary replay collator.
            from rllib_async.gnn.graph_batch import pack_graph_observations

            graphs = [_without_graph_padding(item) for item in observations]
            return pack_graph_observations(graphs, validate=validate, terminated=terminated)
        if validate and any(set(item) != set(first) for item in observations):
            raise ValueError("observation mappings must have identical keys")
        return {
            key: _stack_observations(
                [item[key] for item in observations], validate=validate,
                terminated=terminated,
            )
            for key in first
        }
    result = np.ascontiguousarray(np.stack(observations))
    if validate and (result.dtype.kind not in "biuf" or not np.isfinite(result).all()):
        raise ValueError("observations must contain finite real arrays")
    return result


def _without_graph_padding(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    if "node_count" not in observation:
        return observation
    graph = dict(observation)
    node_count = int(graph.pop("node_count"))
    edge_count = int(graph.pop("edge_count"))
    graph["node_features"] = graph["node_features"][:node_count]
    graph["edge_index"] = graph["edge_index"][:, :edge_count]
    if "edge_features" in graph:
        graph["edge_features"] = graph["edge_features"][:edge_count]
    return graph


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
