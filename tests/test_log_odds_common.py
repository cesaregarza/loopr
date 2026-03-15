"""Tests for shared log-odds helpers (algorithms/_log_odds_common.py)."""

import numpy as np
import polars as pl
import pytest

from loopr.algorithms._log_odds_common import (
    aggregate_entity_metrics,
    appeared_entity_ids,
    build_index_mapping,
    last_activity_from_metrics,
    merged_node_ids,
    reporting_exposure,
    resolve_lambda,
    teleport_from_share,
)


class TestBuildIndexMapping:
    def test_basic_mapping(self):
        mapping = build_index_mapping({10: 0, 20: 1, 30: 2})
        assert mapping.height == 3
        assert set(mapping["id"].to_list()) == {10, 20, 30}
        assert set(mapping["idx"].to_list()) == {0, 1, 2}

    def test_filters_none_keys(self):
        mapping = build_index_mapping({10: 0, None: 1, 20: 2})
        assert mapping.height == 2

    def test_empty_dict(self):
        mapping = build_index_mapping({})
        assert mapping.is_empty()


class TestAggregateEntityMetrics:
    def test_aggregates_winners_and_losers(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1, 2]],
                "losers": [[3]],
                "share": [0.5],
                "weight": [1.0],
                "ts": [100.0],
            }
        )
        result = aggregate_entity_metrics(mdf)
        assert result.height == 3
        ids = set(result["id"].to_list())
        assert ids == {1, 2, 3}

    def test_sums_share_across_matches(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1], [1]],
                "losers": [[2], [2]],
                "share": [0.3, 0.7],
                "weight": [1.0, 1.0],
                "ts": [100.0, 200.0],
            }
        )
        result = aggregate_entity_metrics(mdf)
        p1 = result.filter(pl.col("id") == 1)
        assert p1["share"][0] == pytest.approx(1.0)

    def test_max_timestamp(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1], [1]],
                "losers": [[2], [2]],
                "share": [0.5, 0.5],
                "weight": [1.0, 1.0],
                "ts": [100.0, 200.0],
            }
        )
        result = aggregate_entity_metrics(mdf)
        p1 = result.filter(pl.col("id") == 1)
        assert p1["ts"][0] == pytest.approx(200.0)

    def test_empty_dataframe(self):
        mdf = pl.DataFrame(
            schema={
                "winners": pl.List(pl.Int64),
                "losers": pl.List(pl.Int64),
                "share": pl.Float64,
                "weight": pl.Float64,
                "ts": pl.Float64,
            }
        )
        result = aggregate_entity_metrics(mdf)
        assert result.is_empty()


class TestAppearedEntityIds:
    def test_finds_all_ids(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1, 2], [3]],
                "losers": [[4], [5, 6]],
            }
        )
        ids = appeared_entity_ids(mdf)
        assert ids == {1, 2, 3, 4, 5, 6}

    def test_empty_dataframe(self):
        mdf = pl.DataFrame(
            schema={
                "winners": pl.List(pl.Int64),
                "losers": pl.List(pl.Int64),
            }
        )
        assert appeared_entity_ids(mdf) == set()


class TestMergedNodeIds:
    def test_merges_appeared_and_active(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1, 2]],
                "losers": [[3]],
            }
        )
        result = merged_node_ids(mdf, active_ids=[4, 5])
        assert set(result) == {1, 2, 3, 4, 5}

    def test_deterministic_order(self):
        mdf = pl.DataFrame(
            {
                "winners": [[3, 1]],
                "losers": [[2]],
            }
        )
        result = merged_node_ids(mdf)
        assert result == sorted(result)


class TestTeleportFromShare:
    def test_sums_to_one(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1, 2]],
                "losers": [[3]],
                "share": [0.5],
            }
        )
        node_to_idx = {1: 0, 2: 1, 3: 2}
        rho = teleport_from_share(mdf, node_to_idx)
        np.testing.assert_allclose(rho.sum(), 1.0, atol=1e-10)

    def test_all_positive(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1]],
                "losers": [[2]],
                "share": [0.5],
            }
        )
        node_to_idx = {1: 0, 2: 1}
        rho = teleport_from_share(mdf, node_to_idx)
        assert np.all(rho > 0)


class TestReportingExposure:
    def test_basic_exposure(self):
        mdf = pl.DataFrame(
            {
                "winners": [[1]],
                "losers": [[2]],
                "weight": [3.0],
            }
        )
        node_to_idx = {1: 0, 2: 1}
        exposure = reporting_exposure(mdf, node_to_idx)
        assert exposure[0] > 0
        assert exposure[1] > 0


class TestResolveLambda:
    def test_auto_mode(self):
        win_pr = np.array([0.4, 0.3, 0.2, 0.1])
        rho = np.array([0.25, 0.25, 0.25, 0.25])
        lam = resolve_lambda(
            win_pr, rho, lambda_mode="auto", fixed_lambda=None, fallback=1e-4
        )
        assert lam > 0
        # Auto lambda ≈ 0.025 * median(win_pr) / median(rho)
        expected = 0.025 * np.median(win_pr) / np.median(rho)
        assert lam == pytest.approx(expected)

    def test_fixed_mode(self):
        win_pr = np.ones(4) / 4
        rho = np.ones(4) / 4
        lam = resolve_lambda(
            win_pr, rho, lambda_mode="fixed", fixed_lambda=0.05, fallback=1e-4
        )
        assert lam == pytest.approx(0.05)

    def test_fixed_lambda_overrides_mode(self):
        """If fixed_lambda is set, it's used regardless of mode."""
        win_pr = np.ones(4) / 4
        rho = np.ones(4) / 4
        lam = resolve_lambda(
            win_pr, rho, lambda_mode="auto", fixed_lambda=0.123, fallback=1e-4
        )
        assert lam == pytest.approx(0.123)

    def test_fallback_mode(self):
        win_pr = np.ones(4) / 4
        rho = np.ones(4) / 4
        lam = resolve_lambda(
            win_pr,
            rho,
            lambda_mode="some_unknown",
            fixed_lambda=None,
            fallback=0.007,
        )
        assert lam == pytest.approx(0.007)


class TestLastActivityFromMetrics:
    def test_fills_default_for_missing(self):
        metrics = pl.DataFrame(
            {"id": [1, 2], "ts": [100.0, 200.0]}
        )
        mapping = pl.DataFrame({"id": [1, 2, 3], "idx": [0, 1, 2]})
        result = last_activity_from_metrics(metrics, mapping, 3, 999.0)
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(200.0)
        assert result[2] == pytest.approx(999.0)  # default for missing
