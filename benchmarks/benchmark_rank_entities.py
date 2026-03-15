from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import numpy as np
import polars as pl

from loopr import ExposureLogOddsConfig, LOOPREngine


def build_synthetic_tables(
    *,
    events: int,
    teams_per_event: int,
    matches_per_event: int,
    roster_size: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    now = 1_700_000_000.0

    match_rows: list[dict[str, float | int]] = []
    participant_rows: list[dict[str, int]] = []
    match_id = 1

    for event_id in range(1, events + 1):
        team_ids = [event_id * 10_000 + team for team in range(teams_per_event)]

        for team_offset, team_id in enumerate(team_ids):
            base_entity = event_id * 1_000_000 + team_offset * roster_size
            for roster_index in range(roster_size):
                participant_rows.append(
                    {
                        "event_id": event_id,
                        "group_id": team_id,
                        "entity_id": base_entity + roster_index,
                    }
                )

        for _ in range(matches_per_event):
            winner_idx, loser_idx = rng.choice(
                len(team_ids), size=2, replace=False
            )
            completed_at = now - float(rng.integers(0, 180)) * 86_400.0
            match_rows.append(
                {
                    "event_id": event_id,
                    "match_id": match_id,
                    "winner_id": team_ids[winner_idx],
                    "loser_id": team_ids[loser_idx],
                    "completed_at": completed_at,
                }
            )
            match_id += 1

    return pl.DataFrame(match_rows), pl.DataFrame(participant_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LOOPR rank_entities throughput on synthetic data."
    )
    parser.add_argument("--events", type=int, default=40)
    parser.add_argument("--teams-per-event", type=int, default=32)
    parser.add_argument("--matches-per-event", type=int, default=160)
    parser.add_argument("--roster-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--use-tick-tock-active",
        action="store_true",
        help="Include tick-tock active player resolution in the benchmark.",
    )
    args = parser.parse_args()

    matches, participants = build_synthetic_tables(
        events=args.events,
        teams_per_event=args.teams_per_event,
        matches_per_event=args.matches_per_event,
        roster_size=args.roster_size,
        seed=args.seed,
    )

    config = ExposureLogOddsConfig(
        use_tick_tock_active=args.use_tick_tock_active,
    )

    runtimes: list[float] = []
    stage_samples: dict[str, list[float]] = {}
    ranked_rows = 0

    for _ in range(args.repeats):
        engine = LOOPREngine(config=config, now_ts=1_700_000_000.0)
        started = time.perf_counter()
        rankings = engine.rank_entities(matches, participants)
        runtimes.append(time.perf_counter() - started)
        ranked_rows = rankings.height

        for stage, value in (engine.last_stage_timings or {}).items():
            stage_samples.setdefault(stage, []).append(value)

    summary = {
        "workload": {
            "events": args.events,
            "teams_per_event": args.teams_per_event,
            "matches_per_event": args.matches_per_event,
            "roster_size": args.roster_size,
            "repeats": args.repeats,
            "use_tick_tock_active": args.use_tick_tock_active,
        },
        "rows": {
            "matches": matches.height,
            "participants": participants.height,
            "ranked_entities": ranked_rows,
        },
        "runtime_seconds": {
            "min": min(runtimes),
            "mean": mean(runtimes),
            "max": max(runtimes),
        },
        "stage_seconds_mean": {
            stage: mean(values) for stage, values in sorted(stage_samples.items())
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
