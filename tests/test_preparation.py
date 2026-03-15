import polars as pl

from loopr.core.preparation import (
    participants_by_tournament,
    prepare_exposure_graph,
    prepare_row_edge_inputs,
    prepare_weighted_matches,
    resolve_match_participants,
)
from loopr.schema import prepare_rank_inputs


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
