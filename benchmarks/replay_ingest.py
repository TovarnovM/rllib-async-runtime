"""Profile authoritative in-process replay commit and FIFO retention."""

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
from rllib_async.protocols import FlatEpisodeCodec
from rllib_async.replay import EpisodeStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", choices=("flat", "gnn", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--capacity-transitions", type=int, default=4_096)
    parser.add_argument("--capacity-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--journal-capacity", type=int, default=256)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_payload(args: argparse.Namespace, payload_kind: str) -> dict[str, object]:
    codec = FlatEpisodeCodec() if payload_kind == "flat" else graph_codec()
    maker = make_flat_episode if payload_kind == "flat" else make_graph_episode
    build_started = time.perf_counter()
    episodes = [
        maker(sequence, args.episode_length, codec) for sequence in range(args.episodes)
    ]
    payload_build_s = time.perf_counter() - build_started
    store = EpisodeStore(
        codec,
        capacity_transitions=args.capacity_transitions,
        capacity_bytes=args.capacity_bytes,
        journal_capacity=args.journal_capacity,
        store_generation=f"phase11-{payload_kind}-ingest",
    )

    def commit_all() -> None:
        for episode in episodes:
            acknowledgement = store.commit_episode(episode)
            if not acknowledgement.committed:
                raise RuntimeError(
                    f"benchmark episode {episode.episode_id!r} was not committed"
                )

    profile_path = (
        args.profile_dir
        / (f"replay_ingest_{payload_kind}_{args.episodes}x{args.episode_length}.prof")
        if args.profile_dir is not None
        else None
    )
    started = time.perf_counter()
    profiled_call(commit_all, profile_path)
    duration_s = time.perf_counter() - started
    stats = store.get_stats()
    gates = {
        "transition_capacity": (stats.total_transitions <= args.capacity_transitions),
        "byte_capacity": stats.total_estimated_bytes <= args.capacity_bytes,
        "journal_capacity": stats.journal_entries <= args.journal_capacity,
        "all_commits_accounted": stats.committed_episodes == args.episodes,
        "no_rejections": stats.rejected_commits == 0,
    }
    gates["all_passed"] = all(gates.values())
    require_gates(gates)
    return {
        "payload": payload_kind,
        "payload_build_s": payload_build_s,
        "commit_duration_s": duration_s,
        "episodes_per_s": args.episodes / duration_s,
        "transitions_per_s": (args.episodes * args.episode_length / duration_s),
        "stats": asdict(stats),
        "gates": gates,
        "profile": profile_path,
    }


def main() -> None:
    args = parse_args()
    for name in (
        "episodes",
        "episode_length",
        "capacity_transitions",
        "capacity_bytes",
        "journal_capacity",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    payloads = ("flat", "gnn") if args.payload == "all" else (args.payload,)
    results = [run_payload(args, payload) for payload in payloads]
    publish_document(
        benchmark_document(
            "replay_ingest",
            {
                "payload": args.payload,
                "episodes": args.episodes,
                "episode_length": args.episode_length,
                "capacity_transitions": args.capacity_transitions,
                "capacity_bytes": args.capacity_bytes,
                "journal_capacity": args.journal_capacity,
            },
            results,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
