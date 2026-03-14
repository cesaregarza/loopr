import polars as pl

from loopr import LOOPREngine


def test_loopr_engine_ranks_entities_with_neutral_schema():
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

    engine = LOOPREngine()
    rankings = engine.rank_entities(matches, participants)

    assert "entity_id" in rankings.columns
    assert rankings.height > 0
    assert "player_rank" in rankings.columns
    assert rankings["player_rank"].is_sorted(descending=True)
