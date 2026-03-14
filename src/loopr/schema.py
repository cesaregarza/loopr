"""Schema adapters for domain-agnostic LOOPR inputs."""

from __future__ import annotations

import polars as pl

_MATCH_ALIASES = {
    "tournament_id": ("event_id",),
    "winner_team_id": ("winner_id", "winner_group_id"),
    "loser_team_id": ("loser_id", "loser_group_id"),
    "last_game_finished_at": ("completed_at", "event_ts", "timestamp", "ts"),
    "match_created_at": ("created_at",),
    "is_bye": ("walkover", "is_walkover", "bye"),
}

_PARTICIPANT_ALIASES = {
    "tournament_id": ("event_id",),
    "team_id": ("group_id", "side_id", "competitor_id"),
    "user_id": ("entity_id", "participant_id", "player_id"),
}

_APPEARANCE_ALIASES = {
    "tournament_id": ("event_id",),
    "team_id": ("group_id", "side_id", "competitor_id"),
    "user_id": ("entity_id", "participant_id", "player_id"),
}


def _rename_aliases(
    dataframe: pl.DataFrame | None,
    aliases: dict[str, tuple[str, ...]],
) -> pl.DataFrame | None:
    if dataframe is None:
        return None

    rename_map: dict[str, str] = {}
    columns = set(dataframe.columns)

    for canonical, candidates in aliases.items():
        if canonical in columns:
            continue
        for candidate in candidates:
            if candidate in columns:
                rename_map[candidate] = canonical
                break

    return dataframe.rename(rename_map) if rename_map else dataframe


def normalize_matches_schema(matches: pl.DataFrame) -> pl.DataFrame:
    """Rename neutral match columns to LOOPR's internal schema."""
    normalized = _rename_aliases(matches, _MATCH_ALIASES)
    return normalized if normalized is not None else matches


def normalize_participants_schema(participants: pl.DataFrame) -> pl.DataFrame:
    """Rename neutral participant columns to LOOPR's internal schema."""
    normalized = _rename_aliases(participants, _PARTICIPANT_ALIASES)
    return normalized if normalized is not None else participants


def normalize_appearances_schema(
    appearances: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Rename neutral appearance columns to LOOPR's internal schema."""
    return _rename_aliases(appearances, _APPEARANCE_ALIASES)


def normalize_rank_inputs(
    matches: pl.DataFrame,
    participants: pl.DataFrame,
    appearances: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Normalize all ranking inputs to LOOPR's internal schema."""
    return (
        normalize_matches_schema(matches),
        normalize_participants_schema(participants),
        normalize_appearances_schema(appearances),
    )
