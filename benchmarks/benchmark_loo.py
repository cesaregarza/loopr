from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from loopr import LOOPREngine
from loopr.example_data import build_quickstart_frames


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


def _window_tournament_filter(
    tournaments: pl.DataFrame,
    *,
    filter_mode: str,
) -> pl.DataFrame:
    if filter_mode == "none":
        return tournaments
    if filter_mode == "ranked_only":
        return tournaments.filter(pl.col("is_ranked") == True)
    if filter_mode == "ranked_quality":
        return tournaments.filter(
            (pl.col("is_ranked") == True)
            & (pl.col("team_count") >= 4)
            & (pl.col("participated_users_count") >= 16)
            & (pl.col("match_count") >= 10)
            & (~pl.col("name").str.contains("(?i)test|debug|demo|practice"))
        )
    raise ValueError(f"Unsupported window filter mode: {filter_mode}")


def _rename_present(frame: pl.DataFrame | None, mapping: dict[str, str]) -> pl.DataFrame | None:
    if frame is None:
        return None
    available = {src: dst for src, dst in mapping.items() if src in frame.columns}
    return frame.rename(available)


def _load_sendou_window(
    *,
    window_dir: Path,
    filter_mode: str,
    limit_tournaments: int | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, float]:
    matches_path = window_dir / "matches.parquet"
    players_path = window_dir / "players.parquet"
    appearances_path = window_dir / "appearance_teams.parquet"
    tournaments_path = window_dir / "tournaments.parquet"

    for path in (
        matches_path,
        players_path,
        appearances_path,
        tournaments_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required Sendou window artifact: {path}")

    tournaments = pl.read_parquet(tournaments_path)
    tournaments = _window_tournament_filter(tournaments, filter_mode=filter_mode)
    tournaments = tournaments.sort("start_time_ms")
    if limit_tournaments is not None:
        tournaments = tournaments.head(limit_tournaments)

    tournament_ids = tournaments.select("tournament_id")
    matches_sendou = pl.read_parquet(matches_path).join(
        tournament_ids, on="tournament_id", how="inner"
    )
    appearances_sendou = pl.read_parquet(appearances_path).join(
        tournament_ids, on="tournament_id", how="inner"
    )
    players_sendou = (
        pl.read_parquet(players_path)
        .select(
            [
                pl.col("player_id").cast(pl.Int64).alias("user_id"),
                pl.col("display_name").alias("username"),
            ]
        )
        .unique(subset=["user_id"])
    )

    participants_loopr = (
        appearances_sendou.select(["tournament_id", "team_id", "user_id"])
        .unique(subset=["tournament_id", "team_id", "user_id"])
        .join(players_sendou, on="user_id", how="left")
    )

    appearances_loopr = _rename_present(
        appearances_sendou,
        {
            "tournament_id": "event_id",
            "team_id": "group_id",
            "user_id": "entity_id",
        },
    )
    matches_loopr = _rename_present(
        matches_sendou,
        {
            "tournament_id": "event_id",
            "winner_team_id": "winner_id",
            "loser_team_id": "loser_id",
            "last_game_finished_at": "completed_at",
            "match_created_at": "created_at",
            "is_bye": "walkover",
        },
    )
    participants_loopr = _rename_present(
        participants_loopr,
        {
            "tournament_id": "event_id",
            "team_id": "group_id",
            "user_id": "entity_id",
        },
    )

    now_ts = float(
        matches_sendou.select(pl.max("match_created_at")).item() or time.time()
    )
    return matches_loopr, participants_loopr, appearances_loopr, now_ts


def load_dataset(
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, float]:
    if args.dataset == "quickstart":
        frames = build_quickstart_frames()
        return (
            frames["matches"],
            frames["participants"],
            frames["appearances"],
            1_700_000_000.0,
        )
    if args.dataset == "sendou_window":
        return _load_sendou_window(
            window_dir=args.window_dir,
            filter_mode=args.window_filter,
            limit_tournaments=args.limit_tournaments,
        )
    matches, participants = build_synthetic_tables(
        events=args.events,
        teams_per_event=args.teams_per_event,
        matches_per_event=args.matches_per_event,
        roster_size=args.roster_size,
        seed=args.seed,
    )
    return matches, participants, None, 1_700_000_000.0


def _parse_limit_token(token: str) -> int | None:
    token = token.strip().lower()
    if token == "none":
        return None
    return int(token)


def _limit_key(limit: int | None) -> str:
    return "none" if limit is None else str(limit)


def _compare_batch_results(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
) -> dict[str, float | int | None]:
    if baseline.is_empty() or candidate.is_empty():
        return {
            "baseline_rows": baseline.height,
            "candidate_rows": candidate.height,
            "top_overlap": None,
            "shared_match_count": 0,
            "score_delta_spearman": None,
            "score_delta_mae": None,
        }

    k = min(baseline.height, candidate.height)
    baseline_top = set(baseline.head(k)["match_id"].to_list())
    candidate_top = set(candidate.head(k)["match_id"].to_list())
    joined = candidate.select(["match_id", "score_delta"]).join(
        baseline.select(["match_id", "score_delta"]),
        on="match_id",
        how="inner",
        suffix="_baseline",
    )

    spearman = None
    if joined.height >= 2:
        stat = spearmanr(
            joined["score_delta"].to_numpy(),
            joined["score_delta_baseline"].to_numpy(),
        ).statistic
        spearman = None if np.isnan(stat) else float(stat)

    mae = None
    if joined.height > 0:
        mae = float(
            np.mean(
                np.abs(
                    joined["score_delta"].to_numpy()
                    - joined["score_delta_baseline"].to_numpy()
                )
            )
        )

    return {
        "baseline_rows": baseline.height,
        "candidate_rows": candidate.height,
        "top_overlap": len(baseline_top & candidate_top) / float(k),
        "shared_match_count": joined.height,
        "score_delta_spearman": spearman,
        "score_delta_mae": mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LOOPR leave-one-match-out analysis."
    )
    parser.add_argument(
        "--dataset",
        choices=["quickstart", "synthetic", "sendou_window"],
        default="synthetic",
    )
    parser.add_argument("--events", type=int, default=40)
    parser.add_argument("--teams-per-event", type=int, default=32)
    parser.add_argument("--matches-per-event", type=int, default=160)
    parser.add_argument("--roster-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "exact_combined",
            "exact_separate",
            "perturb_2",
            "perturb_4",
        ],
    )
    parser.add_argument(
        "--entity-limits",
        nargs="+",
        default=["20", "50"],
        help="Entity batch limits to benchmark. Use 'none' for full exact audit.",
    )
    parser.add_argument(
        "--window-dir",
        type=Path,
        default=Path("/root/dev/sendouq_analysis/data/embeddings_window_540d_all"),
    )
    parser.add_argument(
        "--window-filter",
        choices=["none", "ranked_only", "ranked_quality"],
        default="ranked_quality",
    )
    parser.add_argument("--limit-tournaments", type=int, default=None)
    args = parser.parse_args()

    entity_limits = [_parse_limit_token(token) for token in args.entity_limits]
    requested_variants = list(dict.fromkeys(args.variants))
    baseline_injected = False
    if any(variant != "exact_combined" for variant in requested_variants):
        if "exact_combined" not in requested_variants:
            requested_variants = ["exact_combined", *requested_variants]
            baseline_injected = True

    matches, participants, appearances, now_ts = load_dataset(args)
    prep_times: list[float] = []
    cache_bytes: list[int] = []
    variant_single_impact_times: dict[str, list[float]] = {
        variant: [] for variant in requested_variants
    }
    variant_batch_times: dict[str, dict[str, list[float]]] = {
        variant: {_limit_key(limit): [] for limit in entity_limits}
        for variant in requested_variants
    }
    variant_last_batches: dict[str, dict[str, pl.DataFrame]] = {
        variant: {} for variant in requested_variants
    }
    skipped_variants: dict[str, str] = {}

    heavy_entity = None
    heavy_match_count = 0
    ranked_rows = 0

    for _ in range(args.repeats):
        engine = LOOPREngine(now_ts=now_ts)
        engine.rank_entities(matches, participants, appearances=appearances)
        ranked_rows = len(engine.last_result.ids)

        started = time.perf_counter()
        engine.prepare_loo_analyzer()
        prep_times.append(time.perf_counter() - started)

        analyzer = engine.get_loo_analyzer()
        cache_bytes.append(analyzer.estimate_cache_bytes())

        heavy_entity, refs = max(
            analyzer._entity_match_index.items(),
            key=lambda item: len(item[1]),
        )
        heavy_match_count = len(refs)
        match_id = refs[0].match_id

        for variant in requested_variants:
            if variant in skipped_variants:
                continue

            try:
                started = time.perf_counter()
                analyzer.impact_of_match_on_entity_variant(
                    match_id,
                    heavy_entity,
                    variant=variant,
                )
                variant_single_impact_times[variant].append(
                    time.perf_counter() - started
                )

                for limit in entity_limits:
                    limit_key = _limit_key(limit)
                    started = time.perf_counter()
                    batch_df = analyzer.analyze_entity_matches_variant(
                        entity_id=heavy_entity,
                        variant=variant,
                        limit=limit,
                        parallel=True,
                        max_workers=args.max_workers,
                    )
                    variant_batch_times[variant][limit_key].append(
                        time.perf_counter() - started
                    )
                    variant_last_batches[variant][limit_key] = batch_df
            except RuntimeError as exc:
                skipped_variants[variant] = str(exc)

    variant_summary: dict[str, object] = {}
    baseline_batches = variant_last_batches.get("exact_combined", {})
    for variant in requested_variants:
        if variant in skipped_variants:
            variant_summary[variant] = {
                "available": False,
                "reason": skipped_variants[variant],
            }
            continue

        batches = {
            limit_key: mean(times)
            for limit_key, times in variant_batch_times[variant].items()
            if times
        }
        summary_entry: dict[str, object] = {
            "available": True,
            "single_impact_seconds_mean": mean(
                variant_single_impact_times[variant]
            ),
            "entity_batch_seconds_mean": batches,
        }
        if variant != "exact_combined" and baseline_batches:
            summary_entry["comparison_to_exact_combined"] = {
                limit_key: _compare_batch_results(
                    baseline_batches[limit_key],
                    variant_last_batches[variant][limit_key],
                )
                for limit_key in baseline_batches
                if limit_key in variant_last_batches[variant]
            }
        variant_summary[variant] = summary_entry

    summary = {
        "workload": {
            "dataset": args.dataset,
            "events": args.events if args.dataset == "synthetic" else None,
            "teams_per_event": args.teams_per_event
            if args.dataset == "synthetic"
            else None,
            "matches_per_event": args.matches_per_event
            if args.dataset == "synthetic"
            else None,
            "roster_size": args.roster_size if args.dataset == "synthetic" else None,
            "window_dir": str(args.window_dir)
            if args.dataset == "sendou_window"
            else None,
            "window_filter": args.window_filter
            if args.dataset == "sendou_window"
            else None,
            "limit_tournaments": args.limit_tournaments
            if args.dataset == "sendou_window"
            else None,
            "entity_limits": [_limit_key(limit) for limit in entity_limits],
            "variants": requested_variants,
            "baseline_injected": baseline_injected,
            "repeats": args.repeats,
            "max_workers": args.max_workers,
        },
        "rows": {
            "matches": matches.height,
            "participants": participants.height,
            "appearances": 0 if appearances is None else appearances.height,
            "ranked_entities": int(ranked_rows),
        },
        "loo": {
            "heavy_entity_id": int(heavy_entity) if heavy_entity is not None else None,
            "heavy_entity_match_count": heavy_match_count,
            "prepare_seconds_mean": mean(prep_times),
            "cache_bytes_mean": int(mean(cache_bytes)),
            "variants": variant_summary,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
