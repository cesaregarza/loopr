"""Tests for edge building and normalization (core/edges.py)."""

import numpy as np
import polars as pl
import pytest

from loopr.core.edges import (
    build_exposure_triplets,
    build_player_edges,
    build_team_edges,
    compute_denominators,
    edges_to_triplets,
    normalize_edges,
)
from loopr.core.smoothing import ConstantSmoothing, NoSmoothing, WinsProportional

NOW = 1_700_000_000.0


@pytest.fixture
def basic_matches():
    """Two matches in one tournament."""
    return pl.DataFrame(
        {
            "event_id": [1, 1],
            "match_id": [10, 11],
            "winner_id": [100, 100],
            "loser_id": [200, 300],
            "completed_at": [NOW - 86400, NOW],
        }
    )


@pytest.fixture
def basic_players():
    return pl.DataFrame(
        {
            "event_id": [1, 1, 1, 1, 1, 1],
            "group_id": [100, 100, 200, 200, 300, 300],
            "entity_id": [1, 2, 3, 4, 5, 6],
        }
    )


class TestBuildPlayerEdges:
    def test_produces_edges(self, basic_matches, basic_players):
        edges = build_player_edges(
            basic_matches,
            basic_players,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
        )
        assert not edges.is_empty()
        assert "loser_user_id" in edges.columns
        assert "winner_user_id" in edges.columns
        assert "weight_sum" in edges.columns

    def test_edges_have_correct_direction(self, basic_matches, basic_players):
        """Edges should go from loser to winner."""
        edges = build_player_edges(
            basic_matches,
            basic_players,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
        )
        losers = set(edges["loser_user_id"].unique().to_list())
        winners = set(edges["winner_user_id"].unique().to_list())
        # Players 3,4 (team 200) and 5,6 (team 300) are losers
        assert losers.intersection({3, 4, 5, 6})
        # Players 1,2 (team 100) are winners
        assert winners.intersection({1, 2})

    def test_decay_reduces_old_match_weight(self, basic_matches, basic_players):
        """Older matches should get lower weight with positive decay."""
        edges_no_decay = build_player_edges(
            basic_matches,
            basic_players,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
        )
        edges_with_decay = build_player_edges(
            basic_matches,
            basic_players,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.01,
        )
        # Total weight should be lower with decay
        assert edges_with_decay["weight_sum"].sum() < edges_no_decay[
            "weight_sum"
        ].sum()

    def test_empty_matches_returns_empty(self, basic_players):
        empty = pl.DataFrame(
            {
                "event_id": pl.Series([], dtype=pl.Int64),
                "match_id": pl.Series([], dtype=pl.Int64),
                "winner_id": pl.Series([], dtype=pl.Int64),
                "loser_id": pl.Series([], dtype=pl.Int64),
                "completed_at": pl.Series([], dtype=pl.Float64),
            }
        )
        edges = build_player_edges(
            empty,
            basic_players,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
        )
        assert edges.is_empty()

    def test_positional_entity_results_build_all_implied_edges(self):
        edges = build_player_edges(
            pl.DataFrame(
                {
                    "event_id": [1, 1, 1, 1],
                    "match_id": [10, 10, 10, 10],
                    "entity_id": [1, 2, 3, 4],
                    "placement": [1, 2, 3, 4],
                    "completed_at": [NOW, NOW, NOW, NOW],
                }
            ),
            None,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
            result_mode="positional",
        )

        assert {
            tuple(row)
            for row in edges.select(
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

    def test_positional_ties_create_bidirectional_edges(self):
        edges = build_player_edges(
            pl.DataFrame(
                {
                    "event_id": [1, 1, 1],
                    "match_id": [10, 10, 10],
                    "entity_id": [10, 20, 30],
                    "placement": [1, 1, 2],
                    "completed_at": [NOW, NOW, NOW],
                }
            ),
            None,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
            result_mode="positional",
        )

        assert {
            tuple(row)
            for row in edges.select(
                ["loser_user_id", "winner_user_id"]
            ).iter_rows()
        } == {(10, 20), (20, 10), (30, 10), (30, 20)}


class TestBuildTeamEdges:
    def test_produces_team_edges(self, basic_matches):
        edges = build_team_edges(
            basic_matches,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
        )
        assert not edges.is_empty()
        assert "loser_team_id" in edges.columns
        assert "winner_team_id" in edges.columns

    def test_correct_teams(self, basic_matches):
        edges = build_team_edges(
            basic_matches,
            tournament_influence={},
            now_timestamp=NOW,
            decay_rate=0.0,
        )
        losers = set(edges["loser_team_id"].to_list())
        winners = set(edges["winner_team_id"].to_list())
        assert 100 in winners
        assert losers.intersection({200, 300})


class TestComputeDenominators:
    def test_no_smoothing(self):
        edges = pl.DataFrame(
            {
                "loser_user_id": [1, 1, 2],
                "winner_user_id": [2, 3, 3],
                "weight_sum": [1.0, 2.0, 3.0],
            }
        )
        denoms = compute_denominators(edges, NoSmoothing())
        # Player 1 total loss = 3.0, Player 2 total loss = 3.0
        row1 = denoms.filter(pl.col("loser_user_id") == 1)
        assert row1["denom"][0] == pytest.approx(3.0)

    def test_with_constant_smoothing(self):
        edges = pl.DataFrame(
            {
                "loser_user_id": [1],
                "winner_user_id": [2],
                "weight_sum": [5.0],
            }
        )
        denoms = compute_denominators(edges, ConstantSmoothing(epsilon=0.1))
        assert denoms["denom"][0] == pytest.approx(5.1)

    def test_lambda_column_is_difference(self):
        edges = pl.DataFrame(
            {
                "loser_user_id": [1],
                "winner_user_id": [2],
                "weight_sum": [5.0],
            }
        )
        denoms = compute_denominators(edges, ConstantSmoothing(epsilon=0.1))
        assert denoms["lambda"][0] == pytest.approx(0.1)


class TestNormalizeEdges:
    def test_normalized_weights_sum_to_one_per_source(self):
        edges = pl.DataFrame(
            {
                "loser_user_id": [1, 1, 2],
                "winner_user_id": [2, 3, 3],
                "weight_sum": [1.0, 2.0, 3.0],
            }
        )
        denoms = compute_denominators(edges, NoSmoothing())
        normalized = normalize_edges(edges, denoms)
        # Player 1's normalized weights should sum to 1.0
        p1 = normalized.filter(pl.col("loser_user_id") == 1)
        assert p1["normalized_weight"].sum() == pytest.approx(1.0)


class TestBuildExposureTriplets:
    def test_basic_triplets(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1, 2]],
                "losers": [[3, 4]],
                "share": [0.25],
            }
        )
        node_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
        rows, cols, weights = build_exposure_triplets(mdf, node_to_idx)
        # 2 winners × 2 losers = 4 pairs
        assert len(rows) == 4
        assert all(w > 0 for w in weights)

    def test_empty_mapping_returns_empty(self):
        mdf = pl.DataFrame(
            {"winners": [[1]], "losers": [[2]], "share": [0.5]}
        )
        rows, cols, weights = build_exposure_triplets(mdf, {})
        assert len(rows) == 0

    def test_duplicate_pairs_are_aggregated(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1], [1]],
                "losers": [[2], [2]],
                "share": [0.2, 0.3],
            }
        )
        rows, cols, weights = build_exposure_triplets(
            mdf,
            {1: 0, 2: 1},
        )
        assert len(rows) == 1
        np.testing.assert_array_equal(rows, [0])
        np.testing.assert_array_equal(cols, [1])
        np.testing.assert_allclose(weights, [0.5])

    def test_accepts_index_mapping_dataframe(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1]],
                "losers": [[2]],
                "share": [0.5],
            }
        )
        index_mapping = pl.DataFrame({"id": [1, 2], "idx": [0, 1]})
        rows, cols, weights = build_exposure_triplets(
            mdf,
            index_mapping=index_mapping,
        )
        np.testing.assert_array_equal(rows, [0])
        np.testing.assert_array_equal(cols, [1])
        np.testing.assert_allclose(weights, [0.5])


class TestEdgesToTriplets:
    def test_basic_conversion(self):
        edges = pl.DataFrame(
            {
                "src": [1, 2],
                "dst": [2, 3],
                "w": [0.5, 0.3],
            }
        )
        node_to_idx = {1: 0, 2: 1, 3: 2}
        rows, cols, weights = edges_to_triplets(
            edges, node_to_idx, "src", "dst", "w"
        )
        assert len(rows) == 2
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_array_equal(cols, [1, 2])
        np.testing.assert_allclose(weights, [0.5, 0.3])

    def test_skips_edges_with_missing_mapping(self):
        edges = pl.DataFrame(
            {
                "src": [1, 2],
                "dst": [2, 99],
                "w": [0.5, 0.3],
            }
        )
        rows, cols, weights = edges_to_triplets(
            edges,
            {1: 0, 2: 1},
            "src",
            "dst",
            "w",
        )
        np.testing.assert_array_equal(rows, [0])
        np.testing.assert_array_equal(cols, [1])
        np.testing.assert_allclose(weights, [0.5])
