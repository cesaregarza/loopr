"""Neutral-schema validation and preparation for LOOPR inputs."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_MATCH_RENAMES = {
    "event_id": "tournament_id",
    "winner_id": "winner_team_id",
    "loser_id": "loser_team_id",
    "completed_at": "last_game_finished_at",
    "created_at": "match_created_at",
    "walkover": "is_bye",
}

_PARTICIPANT_RENAMES = {
    "event_id": "tournament_id",
    "group_id": "team_id",
    "entity_id": "user_id",
}

_APPEARANCE_RENAMES = {
    "event_id": "tournament_id",
    "group_id": "team_id",
    "entity_id": "user_id",
}


@dataclass(frozen=True)
class NormalizedRankingInputs:
    """Prepared ranking inputs used across LOOPR internals."""

    matches: pl.DataFrame
    participants: pl.DataFrame
    appearances: pl.DataFrame | None = None


def _require_columns(
    dataframe: pl.DataFrame,
    required: tuple[str, ...],
    label: str,
) -> None:
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def _rename_for_internal_use(
    dataframe: pl.DataFrame,
    rename_map: dict[str, str],
) -> pl.DataFrame:
    present = {
        source: target
        for source, target in rename_map.items()
        if source in dataframe.columns
    }
    return dataframe.rename(present) if present else dataframe


def prepare_matches_frame(matches: pl.DataFrame) -> pl.DataFrame:
    """Validate neutral match columns and rename them for internal use."""
    _require_columns(
        matches,
        ("event_id", "match_id", "winner_id", "loser_id"),
        "matches",
    )
    return _rename_for_internal_use(matches, _MATCH_RENAMES)


def prepare_participants_frame(participants: pl.DataFrame) -> pl.DataFrame:
    """Validate neutral participant columns and rename them for internal use."""
    _require_columns(
        participants,
        ("event_id", "group_id", "entity_id"),
        "participants",
    )
    return _rename_for_internal_use(participants, _PARTICIPANT_RENAMES)


def prepare_appearances_frame(
    appearances: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Validate neutral appearance columns and rename them for internal use."""
    if appearances is None:
        return None

    _require_columns(
        appearances,
        ("event_id", "match_id", "entity_id"),
        "appearances",
    )
    return _rename_for_internal_use(appearances, _APPEARANCE_RENAMES)


def prepare_rank_inputs(
    matches: pl.DataFrame,
    participants: pl.DataFrame,
    appearances: pl.DataFrame | None = None,
) -> NormalizedRankingInputs:
    """Validate neutral inputs and rename them into the internal schema."""
    return NormalizedRankingInputs(
        matches=prepare_matches_frame(matches),
        participants=prepare_participants_frame(participants),
        appearances=prepare_appearances_frame(appearances),
    )
