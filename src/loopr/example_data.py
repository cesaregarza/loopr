"""Deterministic example datasets used in documentation and smoke tests."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

SECONDS_PER_DAY = 86_400

_ENTITY_ROWS: list[tuple[int, str, float]] = [
    (1, "Ada", 1.85),
    (2, "Blaze", 1.55),
    (3, "Cora", 1.35),
    (4, "Dax", 1.10),
    (5, "Echo", 0.90),
    (6, "Fern", 0.75),
    (7, "Gray", 0.55),
    (8, "Halo", 0.35),
    (9, "Iris", 0.15),
    (10, "Jett", 0.00),
    (11, "Kira", -0.10),
    (12, "Lux", -0.20),
    (13, "Mako", -0.35),
    (14, "Nova", -0.50),
    (15, "Onyx", -0.65),
    (16, "Pike", -0.80),
    (17, "Quill", -0.95),
    (18, "Rune", -1.10),
    (19, "Sol", -1.25),
    (20, "Tali", -1.40),
]

_EVENTS: list[dict[str, object]] = [
    {
        "event_id": 1,
        "event_name": "Spring Split 1",
        "group_prefix": "S1",
        "rosters": [
            ("Comets", [1, 6, 9, 13, 17]),
            ("Foxes", [2, 5, 10, 14, 18]),
            ("Kings", [3, 7, 11, 15, 19]),
            ("Lynx", [4, 8, 12, 16, 20]),
        ],
    },
    {
        "event_id": 2,
        "event_name": "Spring Split 2",
        "group_prefix": "S2",
        "rosters": [
            ("Comets", [1, 7, 10, 15, 20]),
            ("Foxes", [2, 6, 11, 16, 19]),
            ("Kings", [3, 8, 9, 14, 18]),
            ("Lynx", [4, 5, 12, 13, 17]),
        ],
    },
    {
        "event_id": 3,
        "event_name": "Summer Split 1",
        "group_prefix": "U1",
        "rosters": [
            ("Comets", [1, 5, 11, 16, 18]),
            ("Foxes", [2, 8, 9, 13, 20]),
            ("Kings", [3, 6, 12, 14, 17]),
            ("Lynx", [4, 7, 10, 15, 19]),
        ],
    },
    {
        "event_id": 4,
        "event_name": "Summer Split 2",
        "group_prefix": "U2",
        "rosters": [
            ("Comets", [1, 8, 12, 14, 19]),
            ("Foxes", [2, 7, 9, 16, 17]),
            ("Kings", [3, 5, 10, 13, 20]),
            ("Lynx", [4, 6, 11, 15, 18]),
        ],
    },
]

_ROUND_ROBIN_PAIRINGS: list[tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (1, 2),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _lineup_from_roster(
    roster: list[int],
    *,
    round_index: int,
    group_index: int,
    event_id: int,
) -> list[int]:
    bench_index = (round_index + group_index + event_id) % len(roster)
    return [player_id for idx, player_id in enumerate(roster) if idx != bench_index]


def _simulate_best_of_five(
    lineup_a: list[int],
    lineup_b: list[int],
    skill_map: dict[int, float],
    *,
    rng: np.random.Generator,
) -> tuple[int, int]:
    wins_a = 0
    wins_b = 0
    while wins_a < 3 and wins_b < 3:
        base_a = float(np.mean([skill_map[player_id] for player_id in lineup_a]))
        base_b = float(np.mean([skill_map[player_id] for player_id in lineup_b]))
        perf_a = base_a + float(rng.normal(0.0, 0.18))
        perf_b = base_b + float(rng.normal(0.0, 0.18))
        prob_a = _sigmoid((perf_a - perf_b) / 0.55)
        if rng.random() < prob_a:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a, wins_b


def build_quickstart_frames(
    *,
    seed: int = 7,
    start_timestamp: int = 1_700_000_000,
) -> dict[str, pl.DataFrame]:
    """Build the deterministic quickstart dataset."""
    rng = np.random.default_rng(seed)
    skill_map = {entity_id: skill for entity_id, _, skill in _ENTITY_ROWS}

    entities = pl.DataFrame(
        {
            "entity_id": [row[0] for row in _ENTITY_ROWS],
            "entity_name": [row[1] for row in _ENTITY_ROWS],
        }
    )

    group_rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, int]] = []
    match_rows: list[dict[str, int]] = []
    appearance_rows: list[dict[str, int]] = []

    match_id = 1
    current_ts = start_timestamp

    for event in _EVENTS:
        event_id = int(event["event_id"])
        group_prefix = str(event["group_prefix"])
        raw_rosters = list(event["rosters"])

        rosters: list[dict[str, object]] = []
        for group_index, (group_name, roster) in enumerate(raw_rosters, start=1):
            group_id = event_id * 100 + group_index
            rosters.append(
                {
                    "group_id": group_id,
                    "group_name": f"{group_prefix} {group_name}",
                    "roster": roster,
                }
            )
            group_rows.append(
                {
                    "event_id": event_id,
                    "group_id": group_id,
                    "group_name": f"{group_prefix} {group_name}",
                }
            )
            for entity_id in roster:
                participant_rows.append(
                    {
                        "event_id": event_id,
                        "group_id": group_id,
                        "entity_id": int(entity_id),
                    }
                )

        standings = {
            int(team["group_id"]): {"wins": 0, "game_diff": 0}
            for team in rosters
        }

        for round_index, (left_idx, right_idx) in enumerate(_ROUND_ROBIN_PAIRINGS):
            left_team = rosters[left_idx]
            right_team = rosters[right_idx]
            left_lineup = _lineup_from_roster(
                list(left_team["roster"]),
                round_index=round_index,
                group_index=left_idx,
                event_id=event_id,
            )
            right_lineup = _lineup_from_roster(
                list(right_team["roster"]),
                round_index=round_index,
                group_index=right_idx,
                event_id=event_id,
            )

            left_score, right_score = _simulate_best_of_five(
                left_lineup,
                right_lineup,
                skill_map,
                rng=rng,
            )

            left_group_id = int(left_team["group_id"])
            right_group_id = int(right_team["group_id"])
            if left_score > right_score:
                winner_id, loser_id = left_group_id, right_group_id
                winner_lineup, loser_lineup = left_lineup, right_lineup
                standings[left_group_id]["wins"] += 1
            else:
                winner_id, loser_id = right_group_id, left_group_id
                winner_lineup, loser_lineup = right_lineup, left_lineup
                standings[right_group_id]["wins"] += 1

            standings[left_group_id]["game_diff"] += left_score - right_score
            standings[right_group_id]["game_diff"] += right_score - left_score

            match_rows.append(
                {
                    "event_id": event_id,
                    "match_id": match_id,
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "completed_at": current_ts,
                }
            )
            for entity_id in winner_lineup:
                appearance_rows.append(
                    {
                        "event_id": event_id,
                        "match_id": match_id,
                        "entity_id": int(entity_id),
                        "group_id": winner_id,
                    }
                )
            for entity_id in loser_lineup:
                appearance_rows.append(
                    {
                        "event_id": event_id,
                        "match_id": match_id,
                        "entity_id": int(entity_id),
                        "group_id": loser_id,
                    }
                )

            match_id += 1
            current_ts += 4 * 3_600

        finalists = sorted(
            rosters,
            key=lambda team: (
                standings[int(team["group_id"])]["wins"],
                standings[int(team["group_id"])]["game_diff"],
            ),
            reverse=True,
        )[:2]

        final_left = finalists[0]
        final_right = finalists[1]
        final_left_lineup = _lineup_from_roster(
            list(final_left["roster"]),
            round_index=len(_ROUND_ROBIN_PAIRINGS),
            group_index=0,
            event_id=event_id,
        )
        final_right_lineup = _lineup_from_roster(
            list(final_right["roster"]),
            round_index=len(_ROUND_ROBIN_PAIRINGS),
            group_index=1,
            event_id=event_id,
        )
        left_score, right_score = _simulate_best_of_five(
            final_left_lineup,
            final_right_lineup,
            skill_map,
            rng=rng,
        )

        left_group_id = int(final_left["group_id"])
        right_group_id = int(final_right["group_id"])
        if left_score > right_score:
            winner_id, loser_id = left_group_id, right_group_id
            winner_lineup, loser_lineup = final_left_lineup, final_right_lineup
        else:
            winner_id, loser_id = right_group_id, left_group_id
            winner_lineup, loser_lineup = final_right_lineup, final_left_lineup

        match_rows.append(
            {
                "event_id": event_id,
                "match_id": match_id,
                "winner_id": winner_id,
                "loser_id": loser_id,
                "completed_at": current_ts,
            }
        )
        for entity_id in winner_lineup:
            appearance_rows.append(
                {
                    "event_id": event_id,
                    "match_id": match_id,
                    "entity_id": int(entity_id),
                    "group_id": winner_id,
                }
            )
        for entity_id in loser_lineup:
            appearance_rows.append(
                {
                    "event_id": event_id,
                    "match_id": match_id,
                    "entity_id": int(entity_id),
                    "group_id": loser_id,
                }
            )

        match_id += 1
        current_ts += 6 * 3_600
        current_ts += 12 * SECONDS_PER_DAY

    matches = pl.DataFrame(match_rows).sort(["completed_at", "match_id"])
    participants = pl.DataFrame(participant_rows).sort(
        ["event_id", "group_id", "entity_id"]
    )
    appearances = pl.DataFrame(appearance_rows).sort(
        ["event_id", "match_id", "group_id", "entity_id"]
    )
    groups = pl.DataFrame(group_rows).sort(["event_id", "group_id"])

    return {
        "entities": entities,
        "groups": groups,
        "matches": matches,
        "participants": participants,
        "appearances": appearances,
    }


def write_quickstart_dataset(
    output_dir: str | Path,
    *,
    seed: int = 7,
    start_timestamp: int = 1_700_000_000,
) -> dict[str, Path]:
    """Write the quickstart dataset to CSV files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = build_quickstart_frames(
        seed=seed,
        start_timestamp=start_timestamp,
    )

    written: dict[str, Path] = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.csv"
        frame.write_csv(path)
        written[name] = path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the deterministic LOOPR quickstart dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/quickstart"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-timestamp", type=int, default=1_700_000_000)
    args = parser.parse_args(argv)

    write_quickstart_dataset(
        args.output_dir,
        seed=args.seed,
        start_timestamp=args.start_timestamp,
    )
    return 0
