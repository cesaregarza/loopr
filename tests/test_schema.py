import polars as pl

from loopr import (
    normalize_appearances_schema,
    normalize_matches_schema,
    normalize_participants_schema,
)


def test_match_schema_aliases_are_normalized():
    matches = pl.DataFrame(
        {
            "event_id": [1],
            "winner_id": [10],
            "loser_id": [20],
            "completed_at": [123],
            "walkover": [False],
        }
    )
    normalized = normalize_matches_schema(matches)
    assert {"tournament_id", "winner_team_id", "loser_team_id"}.issubset(
        normalized.columns
    )
    assert "last_game_finished_at" in normalized.columns
    assert "is_bye" in normalized.columns


def test_participant_schema_aliases_are_normalized():
    participants = pl.DataFrame(
        {"event_id": [1], "group_id": [10], "entity_id": [999]}
    )
    normalized = normalize_participants_schema(participants)
    assert normalized.columns == ["tournament_id", "team_id", "user_id"]


def test_appearance_schema_aliases_are_normalized():
    appearances = pl.DataFrame(
        {
            "event_id": [1],
            "match_id": [5],
            "group_id": [10],
            "entity_id": [999],
        }
    )
    normalized = normalize_appearances_schema(appearances)
    assert {"tournament_id", "match_id", "team_id", "user_id"} == set(
        normalized.columns
    )
