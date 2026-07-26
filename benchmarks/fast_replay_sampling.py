"""Compare direct and queued FastReplay sampling/collation throughput."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.common import (
    benchmark_document,
    graph_codec,
    make_flat_episode,
    make_graph_episode,
    profiled_call,
    publish_document,
    require_gates,
)
from rllib_async.examples import SHARED_GNN_MODULE_ID
from rllib_async.gnn import GraphBatchCollator
from rllib_async.protocols import FlatEpisodeCodec
from rllib_async.replay import (
    BatchProducer,
    EpisodeStore,
    FastReplay,
    FlatBatchCollator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", choices=("flat", "gnn", "all"), default="all")
    parser.add_argument("--mode", choices=("direct", "queued", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=1_000)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_replay(
    args: argparse.Namespace,
    payload_kind: str,
) -> tuple[FastReplay, object]:
    codec = FlatEpisodeCodec() if payload_kind == "flat" else graph_codec()
    maker = make_flat_episode if payload_kind == "flat" else make_graph_episode
    episodes = [
        maker(sequence, args.episode_length, codec) for sequence in range(args.episodes)
    ]
    capacity_transitions = args.episodes * args.episode_length
    capacity_bytes = sum(episode.estimated_bytes for episode in episodes)
    store = EpisodeStore(
        codec,
        capacity_transitions=capacity_transitions,
        capacity_bytes=capacity_bytes,
        journal_capacity=max(args.episodes, 1),
        store_generation=f"phase11-{payload_kind}-sampling",
    )
    for episode in episodes:
        store.commit_episode(episode)
    replay = FastReplay(codec)
    replay.load_snapshot(store.get_snapshot())
    collator = (
        FlatBatchCollator()
        if payload_kind == "flat"
        else GraphBatchCollator(module_id=SHARED_GNN_MODULE_ID)
    )
    return replay, collator


def run_mode(
    args: argparse.Namespace,
    payload_kind: str,
    mode: str,
    replay: FastReplay,
    collator: object,
) -> dict[str, object]:
    queue_capacity = 0 if mode == "direct" else args.queue_capacity
    producer = BatchProducer(
        replay,
        collator,
        batch_size=args.batch_size,
        queue_capacity=queue_capacity,
        seed=args.seed,
    )
    profile_path = (
        args.profile_dir
        / (
            f"fast_replay_{payload_kind}_{mode}_"
            f"{args.episodes}x{args.episode_length}_"
            f"b{args.batch_size}_n{args.batches}.prof"
        )
        if args.profile_dir is not None
        else None
    )
    try:
        producer.start()
        producer.get(timeout=5)

        def consume() -> None:
            for _ in range(args.batches):
                producer.get(timeout=5)

        started = time.perf_counter()
        profiled_call(consume, profile_path)
        duration_s = time.perf_counter() - started
        stats = producer.get_stats()
        gates = {
            "queue_bounded": stats.queue_high_watermark <= stats.queue_capacity,
            "mode_matches_queue": stats.prefetch_enabled == (mode == "queued"),
            "no_failures": stats.producer_failures == 0,
            "all_batches_consumed": stats.batches_consumed >= args.batches,
            "build_time_measured": (
                stats.batch_builds >= stats.batches_consumed
                and stats.batch_build_ms_p95 >= 0
            ),
            "data_wait_measured": (
                stats.data_wait_calls >= stats.batches_consumed
                and stats.data_wait_s >= 0
            ),
        }
        gates["all_passed"] = all(gates.values())
        require_gates(gates)
        return {
            "payload": payload_kind,
            "mode": mode,
            "duration_s": duration_s,
            "batches_per_s": args.batches / duration_s,
            "transitions_per_s": (args.batches * args.batch_size / duration_s),
            "producer": asdict(stats),
            "fast_replay": asdict(replay.get_stats()),
            "gates": gates,
            "profile": profile_path,
        }
    finally:
        producer.stop(timeout=5)


def main() -> None:
    args = parse_args()
    for name in (
        "episodes",
        "episode_length",
        "batch_size",
        "batches",
        "queue_capacity",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    payloads = ("flat", "gnn") if args.payload == "all" else (args.payload,)
    modes = ("direct", "queued") if args.mode == "all" else (args.mode,)
    results = []
    for payload in payloads:
        replay, collator = build_replay(args, payload)
        try:
            results.extend(
                run_mode(args, payload, mode, replay, collator) for mode in modes
            )
        finally:
            replay.close(timeout=5)
    publish_document(
        benchmark_document(
            "fast_replay_sampling",
            {
                "payload": args.payload,
                "mode": args.mode,
                "episodes": args.episodes,
                "episode_length": args.episode_length,
                "batch_size": args.batch_size,
                "batches": args.batches,
                "queue_capacity": args.queue_capacity,
                "seed": args.seed,
            },
            results,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
