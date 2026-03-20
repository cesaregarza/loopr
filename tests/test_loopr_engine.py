import polars as pl
import pytest

from loopr import ExposureLogOddsConfig, LOOPREngine, rank_entities


def test_loopr_engine_ranks_entities_with_neutral_schema(multi_event_neutral_tables):
    engine = LOOPREngine()
    rankings = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
    )

    assert "entity_id" in rankings.columns
    assert rankings.height > 0
    assert "player_rank" in rankings.columns
    assert rankings["score"].is_sorted(descending=True)
    assert rankings["player_rank"].to_list() == list(
        range(1, rankings.height + 1)
    )
    assert engine.last_result is not None
    assert engine.last_result.stage_timings is not None
    assert engine.last_result.stage_timings["total"] > 0.0
    assert engine.last_result.teleport is not None
    assert abs(engine.last_result.teleport.sum() - 1.0) < 1e-10


def test_top_level_rank_entities_matches_engine_output(
    multi_event_neutral_tables,
):
    engine = LOOPREngine(
        config=ExposureLogOddsConfig(use_tick_tock_active=False),
        now_ts=1_700_000_000,
    )
    via_engine = engine.rank_entities(
        multi_event_neutral_tables["matches"],
        multi_event_neutral_tables["participants"],
        appearances=multi_event_neutral_tables["appearances"],
    )

    with pytest.warns(UserWarning, match="disconnected components"):
        via_helper = rank_entities(
            multi_event_neutral_tables["matches"],
            multi_event_neutral_tables["participants"],
            appearances=multi_event_neutral_tables["appearances"],
            config=ExposureLogOddsConfig(use_tick_tock_active=False),
            now_ts=1_700_000_000,
        )

    assert via_helper.equals(via_engine)


def test_loopr_engine_keeps_largest_component_by_share_mass():
    matches = pl.DataFrame(
        {
            "event_id": [1, 2, 2, 2],
            "match_id": [10, 20, 21, 22],
            "winner_id": [100, 300, 300, 400],
            "loser_id": [200, 400, 400, 300],
            "completed_at": [1.0, 1.0, 1.0, 1.0],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 1, 1, 2, 2],
            "group_id": [100, 100, 200, 200, 300, 400],
            "entity_id": [1, 2, 3, 4, 5, 6],
        }
    )

    engine = LOOPREngine(
        config=ExposureLogOddsConfig(use_tick_tock_active=False),
        now_ts=1_700_000_000,
    )
    rankings = engine.rank_entities(matches, participants)
    report = engine.last_connectivity_report

    assert set(rankings["entity_id"].to_list()) == {5, 6}
    assert report is not None
    assert report["component_policy"] == "keep_largest"
    assert report["component_count"] == 2
    assert report["largest_component_share_fraction"] == pytest.approx(6 / 7)
    assert report["largest_component_weight_fraction"] == pytest.approx(0.6)
    assert report["largest_component_node_fraction"] == pytest.approx(2 / 6)
    assert report["disconnected_share_fraction"] == pytest.approx(1 / 7)
    assert report["kept_match_count"] == 3
    assert report["dropped_match_count"] == 1
    assert report["kept_entity_count"] == 2
    assert report["dropped_entity_count"] == 4
    assert report["warning_message"] is not None
    assert "Largest component retains 85.7% of share mass." in report["warning_message"]


def test_loopr_engine_allow_policy_keeps_all_components():
    matches = pl.DataFrame(
        {
            "event_id": [1, 2],
            "match_id": [10, 20],
            "winner_id": [100, 300],
            "loser_id": [200, 400],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "group_id": [100, 200, 300, 400],
            "entity_id": [1, 2, 3, 4],
        }
    )

    engine = LOOPREngine(now_ts=1_700_000_000)
    rankings = engine.rank_entities(
        matches,
        participants,
        component_policy="allow",
    )

    assert set(rankings["entity_id"].to_list()) == {1, 2, 3, 4}
    assert engine.last_connectivity_report is not None
    assert engine.last_connectivity_report["component_policy"] == "allow"
    assert engine.last_connectivity_report["warning_message"] is not None
    assert "cross-component ordering is not data-supported" in (
        engine.last_connectivity_report["warning_message"]
    )


def test_loopr_engine_error_policy_rejects_disconnected_graph():
    matches = pl.DataFrame(
        {
            "event_id": [1, 2],
            "match_id": [10, 20],
            "winner_id": [100, 300],
            "loser_id": [200, 400],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "group_id": [100, 200, 300, 400],
            "entity_id": [1, 2, 3, 4],
        }
    )

    engine = LOOPREngine(now_ts=1_700_000_000)
    with pytest.raises(ValueError, match="disconnected components"):
        engine.rank_entities(matches, participants, component_policy="error")


def test_top_level_rank_entities_warns_when_trimming_disconnected_components():
    matches = pl.DataFrame(
        {
            "event_id": [1, 2],
            "match_id": [10, 20],
            "winner_id": [100, 300],
            "loser_id": [200, 400],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "group_id": [100, 200, 300, 400],
            "entity_id": [1, 2, 3, 4],
        }
    )

    with pytest.warns(UserWarning, match="disconnected components"):
        rankings = rank_entities(matches, participants, now_ts=1_700_000_000)

    assert set(rankings["entity_id"].to_list()) == {1, 2}
