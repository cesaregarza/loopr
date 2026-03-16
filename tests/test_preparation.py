import polars as pl
import pytest

from loopr.core.preparation import (
    group_team_members,
    participants_by_tournament,
    prepare_exposure_graph,
    prepare_row_edge_inputs,
    prepare_weighted_positional_results,
    prepare_weighted_matches,
    resolve_match_participants,
    resolve_positional_results,
)
from loopr.schema import prepare_positional_results_frame, prepare_rank_inputs


NOW = 1_700_000_100.0


def test_prepare_weighted_matches_adds_weight_and_timestamp(
    single_match_neutral_tables,
):
    inputs = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
    )

    weighted = prepare_weighted_matches(
        inputs.matches,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
    )

    assert weighted.matches.height == 1
    assert weighted.matches["ts"][0] == single_match_neutral_tables["matches"][
        "completed_at"
    ][0]
    assert weighted.matches["weight"][0] == 1.0


def test_resolve_match_participants_prefers_appearances(
    single_match_neutral_tables,
):
    inputs = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )

    weighted = prepare_weighted_matches(
        inputs.matches,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
    )
    resolved = resolve_match_participants(
        weighted,
        inputs.participants,
        appearances=inputs.appearances,
        include_share=True,
    )

    assert set(resolved.matches["winners"][0]) == {1, 2}
    assert set(resolved.matches["losers"][0]) == {5, 6}
    assert resolved.matches["share"][0] == 0.25


def test_prepare_exposure_graph_builds_metrics_edges_and_node_mapping(
    single_match_neutral_tables,
):
    inputs = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )

    graph = prepare_exposure_graph(
        inputs.matches,
        inputs.participants,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
        appearances=inputs.appearances,
        active_entities=[99],
    )

    assert set(graph.node_ids) == {1, 2, 5, 6, 99}
    assert graph.pair_edges.height == 4
    assert set(graph.entity_metrics["id"].to_list()) == {1, 2, 5, 6}
    assert set(graph.index_mapping.columns) == {"id", "idx"}


def test_prepare_row_edges_and_participants_by_tournament_use_resolved_matches(
    single_match_neutral_tables,
):
    inputs = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )

    row_inputs = prepare_row_edge_inputs(
        inputs.matches,
        inputs.participants,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
        appearances=inputs.appearances,
    )

    assert row_inputs.edges.height == 4
    assert set(row_inputs.node_ids) == {1, 2, 5, 6}
    assert participants_by_tournament(row_inputs.matches) == {999: [1, 2, 5, 6]}


def test_prepare_row_edge_inputs_accepts_grouped_rosters(
    single_match_neutral_tables,
):
    inputs = prepare_rank_inputs(
        single_match_neutral_tables["matches"],
        single_match_neutral_tables["participants"],
        single_match_neutral_tables["appearances"],
    )
    grouped_rosters = group_team_members(inputs.participants)

    row_inputs = prepare_row_edge_inputs(
        inputs.matches,
        inputs.participants,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
        rosters=grouped_rosters,
        appearances=inputs.appearances,
    )

    assert row_inputs.edges.height == 4
    assert set(row_inputs.node_ids) == {1, 2, 5, 6}


def test_prepare_row_edge_inputs_positional_entity_results_expand_ordering():
    positional = prepare_positional_results_frame(
        pl.DataFrame(
            {
                "event_id": [1, 1, 1, 1],
                "match_id": [10, 10, 10, 10],
                "entity_id": [1, 2, 3, 4],
                "placement": [1, 2, 3, 4],
                "completed_at": [NOW, NOW, NOW, NOW],
            }
        )
    )

    row_inputs = prepare_row_edge_inputs(
        positional,
        None,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
        result_mode="positional",
    )

    assert row_inputs.edges.height == 6
    assert set(row_inputs.node_ids) == {1, 2, 3, 4}
    assert {
        tuple(row)
        for row in row_inputs.edges.select(
            ["loser_user_id", "winner_user_id"]
        ).iter_rows()
    } == {
        (2, 1),
        (3, 1),
        (4, 1),
        (3, 2),
        (4, 2),
        (4, 3),
    }


def test_prepare_row_edge_inputs_positional_ties_create_bidirectional_edges():
    positional = prepare_positional_results_frame(
        pl.DataFrame(
            {
                "event_id": [1, 1, 1],
                "match_id": [10, 10, 10],
                "entity_id": [11, 22, 33],
                "placement": [1, 1, 2],
                "completed_at": [NOW, NOW, NOW],
            }
        )
    )

    row_inputs = prepare_row_edge_inputs(
        positional,
        None,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
        result_mode="positional",
    )

    assert {
        tuple(row)
        for row in row_inputs.edges.select(
            ["loser_user_id", "winner_user_id"]
        ).iter_rows()
    } == {(11, 22), (22, 11), (33, 11), (33, 22)}


def test_resolve_positional_results_group_mode_prefers_appearances():
    positional = prepare_positional_results_frame(
        pl.DataFrame(
            {
                "event_id": [1, 1, 1],
                "match_id": [10, 10, 10],
                "group_id": [100, 200, 300],
                "placement": [1, 2, 3],
                "completed_at": [NOW, NOW, NOW],
            }
        )
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 1, 1, 1, 1],
            "group_id": [100, 100, 200, 200, 300, 300],
            "entity_id": [1, 2, 3, 4, 5, 6],
        }
    )
    appearances = pl.DataFrame(
        {
            "event_id": [1, 1, 1, 1],
            "match_id": [10, 10, 10, 10],
            "group_id": [100, 200, 300, 300],
            "entity_id": [1, 3, 5, 6],
        }
    )

    weighted = prepare_weighted_positional_results(
        positional,
        tournament_influence={},
        now_timestamp=NOW,
        decay_rate=0.0,
        beta=0.0,
    )
    prepared_inputs = prepare_rank_inputs(
        pl.DataFrame(
            {
                "event_id": [1],
                "match_id": [999],
                "winner_id": [100],
                "loser_id": [200],
            }
        ),
        participants,
        appearances,
    )
    resolved = resolve_positional_results(
        weighted,
        prepared_inputs.participants,
        appearances=prepared_inputs.appearances,
        include_share=True,
    )

    assert resolved.matches.height == 3
    assert set(resolved.matches["winners"][0]) == {1}
    assert set(resolved.matches["losers"][0]) == {3}
    assert set(resolved.matches["losers"][1]) == {5, 6}


def test_prepare_weighted_positional_results_rejects_single_finisher_after_filtering():
    positional = prepare_positional_results_frame(
        pl.DataFrame(
            {
                "event_id": [1, 1],
                "match_id": [10, 10],
                "entity_id": [1, 2],
                "placement": [1, 2],
                "walkover": [False, True],
            }
        )
    )

    with pytest.raises(ValueError, match="at least two finishers"):
        prepare_weighted_positional_results(
            positional,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
            beta=0.0,
        )
