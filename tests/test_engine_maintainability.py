import numpy as np

from loopr import LOOPREngine, TTLEngine, TickTockEngine, prepare_rank_inputs


def test_prepare_rank_inputs_returns_explicit_container(single_match_neutral_tables):
    prepared = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )

    assert "tournament_id" in prepared.matches.columns
    assert "team_id" in prepared.participants.columns
    assert prepared.players.equals(prepared.participants)


def test_legacy_and_neutral_inputs_produce_same_loopr_rankings(
    multi_event_neutral_tables,
    multi_event_legacy_tables,
):
    engine = LOOPREngine()
    neutral = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    ).rename({"entity_id": "id"}).sort("id")
    legacy = engine.rank_players(
        multi_event_legacy_tables["matches"],
        multi_event_legacy_tables["participants"],
    ).sort("id")

    assert neutral.columns == legacy.columns
    for column in neutral.columns:
        left = neutral[column]
        right = legacy[column]
        if left.dtype.is_float():
            np.testing.assert_allclose(left.to_numpy(), right.to_numpy())
        else:
            assert left.to_list() == right.to_list()


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

    impact = engine.analyze_match_impact(match_id=10, player_id=1)

    assert impact["ok"] is True
    assert impact["match_id"] == 10
    assert impact["player_id"] == 1
