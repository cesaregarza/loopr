import polars as pl
import pytest


@pytest.fixture
def single_match_neutral_tables():
    return {
        "matches": pl.DataFrame(
            {
                "event_id": [999],
                "match_id": [1],
                "winner_id": [10],
                "loser_id": [11],
                "completed_at": [1_700_000_000.0],
            }
        ),
        "participants": pl.DataFrame(
            {
                "event_id": [999] * 8,
                "group_id": [10, 10, 10, 10, 11, 11, 11, 11],
                "entity_id": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ),
        "appearances": pl.DataFrame(
            {
                "event_id": [999] * 4,
                "match_id": [1] * 4,
                "entity_id": [1, 2, 5, 6],
                "group_id": [10, 10, 11, 11],
            }
        ),
    }


@pytest.fixture
def multi_event_neutral_tables():
    matches = pl.DataFrame(
        {
            "event_id": [1, 1, 2],
            "match_id": [10, 11, 20],
            "winner_id": [100, 100, 100],
            "loser_id": [200, 300, 300],
            "completed_at": [1_700_000_000, 1_700_000_100, 1_700_000_200],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1] * 12 + [2] * 8,
            "group_id": [100] * 4 + [200] * 4 + [300] * 4 + [100] * 4 + [300] * 4,
            "entity_id": list(range(1, 21)),
        }
    )
    return {"matches": matches, "participants": participants, "appearances": None}
