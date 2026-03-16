"""Tests for rating backends (algorithms/backends/)."""

import numpy as np
import polars as pl
import pytest

from loopr.algorithms.backends.log_odds import LogOddsBackend
from loopr.algorithms.backends.row_pr import RowPRBackend
from loopr.core.config import ExposureLogOddsConfig, TickTockConfig
from loopr.core.protocols import RatingBackend

NOW = 1_700_000_000.0
DAY = 86400.0


@pytest.fixture
def backend_test_data():
    matches = pl.DataFrame(
        {
            "event_id": [1, 1],
            "match_id": [10, 11],
            "winner_id": [100, 100],
            "loser_id": [200, 300],
            "completed_at": [NOW - DAY, NOW],
        }
    )
    players = pl.DataFrame(
        {
            "event_id": [1] * 6,
            "group_id": [100, 100, 200, 200, 300, 300],
            "entity_id": [1, 2, 3, 4, 5, 6],
        }
    )
    return matches, players


class TestLogOddsBackend:
    def test_satisfies_protocol(self):
        backend = LogOddsBackend()
        assert isinstance(backend, RatingBackend)

    def test_compute_returns_expected_columns(self, backend_test_data):
        matches, players = backend_test_data
        backend = LogOddsBackend()
        result = backend.compute(matches, players, [], {})
        assert "id" in result.columns
        assert "score" in result.columns
        assert "quality_mass" in result.columns
        assert "win_pr" in result.columns
        assert "loss_pr" in result.columns

    def test_quality_mass_between_zero_and_one(self, backend_test_data):
        matches, players = backend_test_data
        backend = LogOddsBackend()
        result = backend.compute(matches, players, [], {})
        qm = result["quality_mass"].to_numpy()
        assert np.all(qm >= 0)
        assert np.all(qm <= 1)

    def test_winners_have_positive_scores(self, backend_test_data):
        matches, players = backend_test_data
        backend = LogOddsBackend()
        result = backend.compute(matches, players, [], {})
        scores = dict(zip(result["id"].to_list(), result["score"].to_list()))
        # Players 1,2 (team 100) won all matches
        assert scores[1] > 0
        assert scores[2] > 0

    def test_fixed_lambda_mode(self, backend_test_data):
        matches, players = backend_test_data
        backend = LogOddsBackend(lambda_mode="fixed", fixed_lambda=0.01)
        result = backend.compute(matches, players, [], {})
        assert result.height > 0
        assert result["lambda_used"][0] == pytest.approx(0.01)

    def test_none_players_returns_empty(self):
        matches = pl.DataFrame(
            {
                "event_id": [1],
                "match_id": [10],
                "winner_id": [100],
                "loser_id": [200],
                "completed_at": [NOW],
            }
        )
        backend = LogOddsBackend()
        result = backend.compute(matches, None, [], {})
        assert result.is_empty()


class TestRowPRBackend:
    def test_satisfies_protocol(self):
        backend = RowPRBackend()
        assert isinstance(backend, RatingBackend)

    def test_compute_returns_expected_columns(self, backend_test_data):
        matches, players = backend_test_data
        backend = RowPRBackend()
        result = backend.compute(matches, players, [], {})
        assert "id" in result.columns
        assert "score" in result.columns
        assert "quality_mass" in result.columns

    def test_pagerank_sums_to_one(self, backend_test_data):
        matches, players = backend_test_data
        backend = RowPRBackend()
        result = backend.compute(matches, players, [], {})
        pr = result["pagerank"].to_numpy()
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-6)

    def test_uniform_teleport_mode(self, backend_test_data):
        matches, players = backend_test_data
        backend = RowPRBackend(teleport_mode="uniform")
        result = backend.compute(matches, players, [], {})
        assert result.height > 0

    def test_none_players_returns_empty(self):
        matches = pl.DataFrame(
            {
                "event_id": [1],
                "match_id": [10],
                "winner_id": [100],
                "loser_id": [200],
                "completed_at": [NOW],
            }
        )
        backend = RowPRBackend()
        result = backend.compute(matches, None, [], {})
        assert result.is_empty()

    def test_winners_rank_higher(self, backend_test_data):
        matches, players = backend_test_data
        backend = RowPRBackend()
        result = backend.compute(matches, players, [], {})
        scores = dict(zip(result["id"].to_list(), result["score"].to_list()))
        # Players 1,2 (team 100) won all → should rank highest
        assert scores[1] > scores[5]
        assert scores[2] > scores[6]


class TestBackendConfigDefaults:
    """Verify backends initialized without args use config defaults."""

    def test_log_odds_defaults_match_config(self):
        cfg = ExposureLogOddsConfig()
        backend = LogOddsBackend()
        assert backend.decay_rate == pytest.approx(cfg.decay.decay_rate)
        assert backend.beta == cfg.engine.beta
        assert backend.alpha == cfg.pagerank.alpha
        assert backend.lambda_mode == cfg.lambda_mode
        assert backend.fixed_lambda == cfg.fixed_lambda
        assert backend.pagerank_tol == cfg.pagerank.tol
        assert backend.pagerank_max_iter == cfg.pagerank.max_iter

    def test_log_odds_explicit_config(self):
        from loopr.core.config import DecayConfig

        cfg = ExposureLogOddsConfig(decay=DecayConfig(half_life_days=180.0))
        backend = LogOddsBackend(config=cfg)
        assert backend.decay_rate == pytest.approx(cfg.decay.decay_rate)

    def test_log_odds_keyword_overrides_config(self):
        backend = LogOddsBackend(alpha=0.99)
        assert backend.alpha == 0.99

    def test_row_pr_defaults_match_config(self):
        cfg = TickTockConfig()
        backend = RowPRBackend()
        assert backend.decay_rate == pytest.approx(cfg.decay.decay_rate)
        assert backend.beta == cfg.engine.beta
        assert backend.alpha == cfg.pagerank.alpha
        assert backend.teleport_mode == cfg.teleport_mode
        assert backend.smoothing_gamma == cfg.engine.gamma
        assert backend.smoothing_cap_ratio == cfg.engine.cap_ratio
        assert backend.pagerank_tol == cfg.pagerank.tol
        assert backend.pagerank_max_iter == cfg.pagerank.max_iter

    def test_row_pr_explicit_config(self):
        from loopr.core.config import DecayConfig

        cfg = TickTockConfig(decay=DecayConfig(half_life_days=180.0))
        backend = RowPRBackend(config=cfg)
        assert backend.decay_rate == pytest.approx(cfg.decay.decay_rate)

    def test_row_pr_keyword_overrides_config(self):
        backend = RowPRBackend(alpha=0.99)
        assert backend.alpha == 0.99
