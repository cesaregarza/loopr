"""Tests for tournament influence and normalization (core/influence.py)."""

import numpy as np
import polars as pl
import pytest

from loopr.core.influence import (
    aggregate_multi_round_influence,
    compute_retrospective_strength,
    compute_tournament_influence,
    normalize_influence,
)


class TestComputeTournamentInfluence:
    def test_arithmetic_mean(self):
        pr = np.array([0.4, 0.3, 0.2, 0.1])
        participants = {1: [0, 1], 2: [2, 3]}
        result = compute_tournament_influence(pr, participants, method="arithmetic")
        assert result[1] == pytest.approx(0.35)  # mean(0.4, 0.3)
        assert result[2] == pytest.approx(0.15)  # mean(0.2, 0.1)

    def test_geometric_mean(self):
        pr = np.array([0.4, 0.1])
        participants = {1: [0, 1]}
        result = compute_tournament_influence(pr, participants, method="geometric")
        # geometric mean ~ exp(mean(log(values + epsilon)))
        expected = np.exp(np.mean(np.log(np.array([0.4, 0.1]) + 1e-10)))
        assert result[1] == pytest.approx(expected, rel=1e-6)

    def test_top_20_sum_with_few_participants(self):
        pr = np.array([0.5, 0.3, 0.2])
        participants = {1: [0, 1, 2]}
        result = compute_tournament_influence(pr, participants, method="top_20_sum")
        # All 3 < 20, so it's mean of all
        assert result[1] == pytest.approx(np.mean([0.5, 0.3, 0.2]))

    def test_empty_participants_get_default(self):
        pr = np.array([0.5, 0.3])
        participants = {1: []}
        result = compute_tournament_influence(pr, participants, method="arithmetic")
        assert result[1] == 1.0

    def test_unknown_method_raises(self):
        pr = np.array([0.5])
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            compute_tournament_influence(pr, {1: [0]}, method="bogus")

    def test_stronger_tournament_gets_higher_influence(self):
        pr = np.array([0.5, 0.4, 0.05, 0.05])
        participants = {1: [0, 1], 2: [2, 3]}
        result = compute_tournament_influence(pr, participants, method="arithmetic")
        assert result[1] > result[2]


class TestNormalizeInfluence:
    def test_none_method_passes_through(self):
        influence = {1: 0.5, 2: 1.5}
        result = normalize_influence(influence, method="none")
        assert result == influence

    def test_minmax_ranges(self):
        influence = {1: 1.0, 2: 5.0, 3: 10.0}
        result = normalize_influence(influence, method="minmax")
        values = list(result.values())
        assert min(values) == pytest.approx(0.5)
        assert max(values) == pytest.approx(2.0)

    def test_zscore_centered(self):
        influence = {1: 1.0, 2: 2.0, 3: 3.0}
        result = normalize_influence(influence, method="zscore")
        # z-scored then mapped to 1.0 ± 0.5*z
        values = list(result.values())
        assert all(v > 0 for v in values)

    def test_log_normalization(self):
        influence = {1: 1.0, 2: 10.0}
        result = normalize_influence(influence, method="log")
        assert result[2] > result[1]

    def test_equal_values_give_ones(self):
        influence = {1: 5.0, 2: 5.0, 3: 5.0}
        result = normalize_influence(influence, method="minmax")
        for v in result.values():
            assert v == pytest.approx(1.0)

    def test_empty_dict(self):
        assert normalize_influence({}, method="minmax") == {}

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_influence({1: 1.0}, method="bogus")


class TestAggregateMultiRoundInfluence:
    def test_single_round(self):
        influences = [{1: 2.0, 2: 3.0}]
        result = aggregate_multi_round_influence(influences)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_uniform_weights(self):
        influences = [{1: 2.0}, {1: 4.0}]
        result = aggregate_multi_round_influence(influences)
        assert result[1] == pytest.approx(3.0)

    def test_custom_weights(self):
        influences = [{1: 2.0}, {1: 8.0}]
        result = aggregate_multi_round_influence(influences, weights=[1.0, 3.0])
        # weighted avg: (1/4)*2 + (3/4)*8 = 0.5 + 6 = 6.5
        assert result[1] == pytest.approx(6.5)

    def test_partial_tournament_presence(self):
        """Tournament appears only in some rounds."""
        influences = [{1: 2.0, 2: 3.0}, {1: 4.0}]
        result = aggregate_multi_round_influence(influences)
        assert 1 in result
        assert 2 in result
        # Tournament 2 only in round 1, so its average = 3.0
        assert result[2] == pytest.approx(3.0)

    def test_empty_list(self):
        assert aggregate_multi_round_influence([]) == {}


class TestComputeRetrospectiveStrength:
    def test_basic_strength(self):
        edges = pl.DataFrame(
            {
                "winner_user_id": [1, 1, 2],
                "loser_user_id": [2, 3, 3],
                "weight_sum": [1.0, 1.0, 1.0],
            }
        )
        pr = np.array([0.5, 0.3, 0.2])
        node_to_idx = {1: 0, 2: 1, 3: 2}
        strength = compute_retrospective_strength(edges, pr, node_to_idx)
        # Player 1 beat players 2 and 3 → should have highest strength
        assert strength[0] > strength[1]
        assert strength[0] > 0
