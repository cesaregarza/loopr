from loopr import LOOPREngine


def test_loopr_engine_ranks_entities_with_neutral_schema(multi_event_neutral_tables):
    engine = LOOPREngine()
    rankings = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )

    assert "entity_id" in rankings.columns
    assert rankings.height > 0
    assert "player_rank" in rankings.columns
    assert rankings["player_rank"].is_sorted(descending=True)
