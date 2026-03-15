"""Tests for smoothing strategies (core/smoothing.py)."""

import numpy as np
import pytest

from loopr.core.smoothing import (
    AdaptiveSmoothing,
    ConstantSmoothing,
    HybridSmoothing,
    NoSmoothing,
    SmoothingStrategy,
    WinsProportional,
    get_smoothing_strategy,
)


@pytest.fixture
def sample_weights():
    loss = np.array([10.0, 5.0, 0.0, 20.0])
    win = np.array([5.0, 10.0, 8.0, 2.0])
    return loss, win


class TestNoSmoothing:
    def test_returns_raw_loss_weights(self, sample_weights):
        loss, win = sample_weights
        result = NoSmoothing().denom(loss, win)
        np.testing.assert_array_equal(result, loss)

    def test_satisfies_protocol(self):
        assert isinstance(NoSmoothing(), SmoothingStrategy)


class TestWinsProportional:
    def test_basic_formula(self, sample_weights):
        loss, win = sample_weights
        gamma = 0.02
        result = WinsProportional(gamma=gamma).denom(loss, win)
        # With cap_ratio=1.0 (default), the smoothing term is capped at loss_weights
        raw = loss + gamma * win
        capped_lambda = np.minimum(raw - loss, 1.0 * loss)
        expected = loss + capped_lambda
        np.testing.assert_allclose(result, expected)

    def test_no_cap_when_infinite(self, sample_weights):
        loss, win = sample_weights
        gamma = 0.5
        result = WinsProportional(gamma=gamma, cap_ratio=float("inf")).denom(
            loss, win
        )
        np.testing.assert_allclose(result, loss + gamma * win)

    def test_result_at_least_as_large_as_loss(self, sample_weights):
        loss, win = sample_weights
        result = WinsProportional().denom(loss, win)
        assert np.all(result >= loss)


class TestConstantSmoothing:
    def test_adds_epsilon(self, sample_weights):
        loss, win = sample_weights
        eps = 0.001
        result = ConstantSmoothing(epsilon=eps).denom(loss, win)
        np.testing.assert_allclose(result, loss + eps)

    def test_zero_loss_gets_smoothed(self):
        loss = np.array([0.0])
        win = np.array([5.0])
        result = ConstantSmoothing(epsilon=0.1).denom(loss, win)
        assert result[0] == pytest.approx(0.1)


class TestAdaptiveSmoothing:
    def test_decreases_with_volume(self):
        """Players with more games should get less smoothing."""
        strategy = AdaptiveSmoothing(base_smooth=0.1, volume_scale=100.0)
        # Low volume
        loss_low = np.array([2.0])
        win_low = np.array([3.0])
        denom_low = strategy.denom(loss_low, win_low)
        smoothing_low = denom_low[0] - loss_low[0]

        # High volume
        loss_high = np.array([200.0])
        win_high = np.array([300.0])
        denom_high = strategy.denom(loss_high, win_high)
        smoothing_high = denom_high[0] - loss_high[0]

        # Smoothing fraction should be lower for high-volume
        assert (smoothing_low / win_low[0]) > (smoothing_high / win_high[0])


class TestHybridSmoothing:
    def test_combines_constant_and_proportional(self, sample_weights):
        loss, win = sample_weights
        eps = 1e-6
        gamma = 0.01
        result = HybridSmoothing(epsilon=eps, gamma=gamma).denom(loss, win)
        np.testing.assert_allclose(result, loss + eps + gamma * win)


class TestSmoothingFactory:
    @pytest.mark.parametrize(
        "mode,cls",
        [
            ("none", NoSmoothing),
            ("wins_proportional", WinsProportional),
            ("constant", ConstantSmoothing),
            ("adaptive", AdaptiveSmoothing),
            ("hybrid", HybridSmoothing),
        ],
    )
    def test_factory_returns_correct_type(self, mode, cls):
        strategy = get_smoothing_strategy(mode)
        assert isinstance(strategy, cls)

    def test_factory_passes_kwargs(self):
        strategy = get_smoothing_strategy("constant", epsilon=0.5)
        assert strategy.epsilon == 0.5

    def test_factory_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown smoothing mode"):
            get_smoothing_strategy("nonexistent")

    def test_factory_ignores_irrelevant_kwargs(self):
        # Should not error even with extra kwargs
        strategy = get_smoothing_strategy(
            "constant", epsilon=0.5, bogus_param=999
        )
        assert strategy.epsilon == 0.5
