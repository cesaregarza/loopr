"""Tests for configuration and result dataclasses."""

import numpy as np
import polars as pl
import pytest

from loopr.core.config import (
    DecayConfig,
    EngineConfig,
    ExposureLogOddsConfig,
    PageRankConfig,
    TickTockConfig,
    merge_configs,
)
from loopr.core.results import ExposureLogOddsResult, RankResult, TickTockResult


class TestDecayConfig:
    def test_default_half_life(self):
        cfg = DecayConfig()
        assert cfg.half_life_days == 30.0

    def test_decay_rate_from_half_life(self):
        cfg = DecayConfig(half_life_days=30.0)
        expected = np.log(2) / 30.0
        assert cfg.decay_rate == pytest.approx(expected)

    def test_zero_half_life_gives_zero_rate(self):
        cfg = DecayConfig(half_life_days=0.0)
        assert cfg.decay_rate == 0.0

    def test_frozen(self):
        cfg = DecayConfig()
        with pytest.raises(AttributeError):
            cfg.half_life_days = 60.0


class TestPageRankConfig:
    def test_defaults(self):
        cfg = PageRankConfig()
        assert cfg.alpha == 0.85
        assert cfg.tol == 1e-8
        assert cfg.max_iter == 200
        assert cfg.orientation == "row"
        assert cfg.redistribute_dangling is True


class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.beta == 1.0
        assert cfg.min_exposure is None
        assert cfg.verbose is False


class TestTickTockConfig:
    def test_nested_configs(self):
        cfg = TickTockConfig()
        assert cfg.max_ticks == 5
        assert isinstance(cfg.engine, EngineConfig)
        assert isinstance(cfg.pagerank, PageRankConfig)
        assert isinstance(cfg.decay, DecayConfig)


class TestExposureLogOddsConfig:
    def test_defaults(self):
        cfg = ExposureLogOddsConfig()
        assert cfg.lambda_mode == "auto"
        assert cfg.apply_log_transform is True
        assert cfg.use_tick_tock_active is True


class TestMergeConfigs:
    def test_merges_two_dicts(self):
        result = merge_configs({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_later_overrides_earlier(self):
        result = merge_configs({"a": 1}, {"a": 2})
        assert result == {"a": 2}


class TestRankResult:
    def test_to_dataframe(self):
        result = RankResult(
            scores=np.array([0.5, 0.3, 0.2]),
            ids=[1, 2, 3],
        )
        df = result.to_dataframe()
        assert "entity_id" in df.columns
        assert "score" in df.columns
        assert df.height == 3

    def test_to_dataframe_custom_columns(self):
        result = RankResult(
            scores=np.array([0.5, 0.3]),
            ids=[10, 20],
        )
        df = result.to_dataframe(id_column="entity_id", score_column="rating")
        assert "entity_id" in df.columns
        assert "rating" in df.columns

    def test_to_dataframe_includes_optional_fields(self):
        result = RankResult(
            scores=np.array([0.5, 0.3]),
            ids=[1, 2],
            win_pagerank=np.array([0.6, 0.4]),
            loss_pagerank=np.array([0.3, 0.7]),
            exposure=np.array([10.0, 5.0]),
            teleport=np.array([0.5, 0.5]),
        )
        df = result.to_dataframe()
        assert "win_pr" in df.columns
        assert "loss_pr" in df.columns
        assert "exposure" in df.columns
        assert "teleport" in df.columns

    def test_get_top_n(self):
        result = RankResult(
            scores=np.array([0.1, 0.5, 0.3, 0.8, 0.2]),
            ids=[1, 2, 3, 4, 5],
        )
        top = result.get_top_n(3)
        assert top.height == 3
        assert top["entity_id"][0] == 4  # highest score


class TestTickTockResult:
    def test_extends_rank_result(self):
        result = TickTockResult(
            scores=np.array([0.5]),
            ids=[1],
            tournament_influence={1: 1.5},
        )
        assert result.tournament_influence[1] == 1.5
        assert isinstance(result, RankResult)


class TestExposureLogOddsResult:
    def test_extends_rank_result(self):
        result = ExposureLogOddsResult(
            scores=np.array([0.5]),
            ids=[1],
            active_mask=np.array([True]),
        )
        assert result.active_mask[0] is np.True_
        assert isinstance(result, RankResult)
