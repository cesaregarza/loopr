import polars as pl
import pytest

from loopr import prepare_rank_inputs


def test_prepare_rank_inputs_renames_neutral_columns_for_internal_use():
    prepared = prepare_rank_inputs(
        pl.DataFrame(
            {
                "event_id": [1],
                "match_id": [10],
                "winner_id": [100],
                "loser_id": [200],
                "completed_at": [123],
                "walkover": [False],
            }
        ),
        pl.DataFrame(
            {"event_id": [1], "group_id": [10], "entity_id": [999]}
        ),
        pl.DataFrame(
            {
                "event_id": [1],
                "match_id": [5],
                "group_id": [10],
                "entity_id": [999],
            }
        ),
    )

    assert {"tournament_id", "winner_team_id", "loser_team_id"}.issubset(
        prepared.matches.columns
    )
    assert "last_game_finished_at" in prepared.matches.columns
    assert "is_bye" in prepared.matches.columns
    assert prepared.participants.columns == ["tournament_id", "team_id", "user_id"]
    assert {"tournament_id", "match_id", "team_id", "user_id"} == set(
        prepared.appearances.columns
    )


def test_prepare_rank_inputs_requires_neutral_match_columns():
    with pytest.raises(ValueError, match="matches is missing required columns"):
        prepare_rank_inputs(
            pl.DataFrame({"winner_id": [10], "loser_id": [20]}),
            pl.DataFrame(
                {"event_id": [1], "group_id": [10], "entity_id": [999]}
            ),
        )


def test_prepare_rank_inputs_requires_neutral_participant_columns():
    with pytest.raises(
        ValueError, match="participants is missing required columns"
    ):
        prepare_rank_inputs(
            pl.DataFrame(
                {
                    "event_id": [1],
                    "match_id": [10],
                    "winner_id": [10],
                    "loser_id": [20],
                }
            ),
            pl.DataFrame({"event_id": [1], "entity_id": [999]}),
        )


def test_prepare_rank_inputs_requires_neutral_appearance_columns():
    with pytest.raises(
        ValueError, match="appearances is missing required columns"
    ):
        prepare_rank_inputs(
            pl.DataFrame(
                {
                    "event_id": [1],
                    "match_id": [10],
                    "winner_id": [10],
                    "loser_id": [20],
                }
            ),
            pl.DataFrame(
                {"event_id": [1], "group_id": [10], "entity_id": [999]}
            ),
            pl.DataFrame({"event_id": [1], "match_id": [10]}),
        )
