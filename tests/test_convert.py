import polars as pl

from loopr import convert_matches_dataframe


def test_convert_matches_accepts_neutral_schema():
    matches = pl.DataFrame(
        {
            "event_id": [999],
            "match_id": [1],
            "winner_id": [10],
            "loser_id": [11],
            "completed_at": [1_700_000_000.0],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [999] * 8,
            "group_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "entity_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    df = convert_matches_dataframe(
        matches,
        participants,
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
    )

    assert df.height == 1
    assert set(df["winners"][0]) == {1, 2, 3, 4}
    assert set(df["losers"][0]) == {5, 6, 7, 8}


def test_convert_matches_neutral_appearances_override_rosters():
    matches = pl.DataFrame(
        {
            "event_id": [999],
            "match_id": [1],
            "winner_id": [10],
            "loser_id": [11],
            "completed_at": [1_700_000_000.0],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [999] * 8,
            "group_id": [10, 10, 10, 10, 11, 11, 11, 11],
            "entity_id": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    appearances = pl.DataFrame(
        {
            "event_id": [999] * 4,
            "match_id": [1] * 4,
            "entity_id": [1, 2, 5, 6],
            "group_id": [10, 10, 11, 11],
        }
    )

    df = convert_matches_dataframe(
        matches,
        participants,
        tournament_influence={},
        now_timestamp=1_700_000_100.0,
        decay_rate=0.0,
        appearances=appearances,
    )

    assert set(df["winners"][0]) == {1, 2}
    assert set(df["losers"][0]) == {5, 6}
