from loopr import LOOPREngine, TTLEngine, TickTockEngine, prepare_rank_inputs


def test_prepare_rank_inputs_returns_explicit_container(single_match_neutral_tables):
    prepared = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )

    assert "tournament_id" in prepared.matches.columns
    assert "team_id" in prepared.participants.columns
    assert prepared.appearances is not None


def test_tick_tock_engine_smoke(multi_event_neutral_tables):
    engine = TickTockEngine()
    rankings = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )

    assert "entity_id" in rankings.columns
    assert "score" in rankings.columns
    assert rankings.height > 0


def test_ttl_engine_smoke(multi_event_neutral_tables):
    engine = TTLEngine()
    rankings = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )

    assert "entity_id" in rankings.columns
    assert "score" in rankings.columns
    assert "quality_mass" in rankings.columns
    assert rankings.height > 0


def test_loo_analyzer_smoke(multi_event_neutral_tables):
    engine = LOOPREngine()
    engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )
    engine.prepare_loo_analyzer()

    impact = engine.analyze_match_impact(match_id=10, entity_id=1)

    assert impact["ok"] is True
    assert impact["match_id"] == 10
    assert impact["entity_id"] == 1
