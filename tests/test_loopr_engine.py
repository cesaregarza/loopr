from loopr import LOOPREngine, rank_entities


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
    assert engine.last_result is not None
    assert engine.last_result.stage_timings is not None
    assert engine.last_result.stage_timings["total"] > 0.0
    assert engine.last_result.teleport is not None
    assert abs(engine.last_result.teleport.sum() - 1.0) < 1e-10


def test_top_level_rank_entities_matches_engine_output(
    multi_event_neutral_tables,
):
    engine = LOOPREngine(now_ts=1_700_000_000)
    via_engine = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
        appearances=multi_event_neutral_tables["appearances"],
    )

    via_helper = rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
        appearances=multi_event_neutral_tables["appearances"],
        now_ts=1_700_000_000,
    )

    assert via_helper.equals(via_engine)
