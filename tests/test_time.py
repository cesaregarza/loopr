"""Tests for time utilities (core/time.py)."""

import numpy as np
import polars as pl
import pytest

from loopr.core.time import (
    Clock,
    apply_inactivity_decay,
    compute_decay_factor,
    create_time_windows,
    filter_by_recency,
)

NOW = 1_700_000_000.0
DAY = 86400.0


class TestClock:
    def test_custom_timestamp(self):
        clock = Clock(now_timestamp=NOW)
        assert clock.now == NOW

    def test_days_ago(self):
        clock = Clock(now_timestamp=NOW)
        assert clock.days_ago(1) == pytest.approx(NOW - DAY)
        assert clock.days_ago(0) == pytest.approx(NOW)

    def test_days_since(self):
        clock = Clock(now_timestamp=NOW)
        ts = NOW - 3 * DAY
        assert clock.days_since(ts) == pytest.approx(3.0)

    def test_auto_timestamp(self):
        clock = Clock()
        assert clock.now > 0


class TestComputeDecayFactor:
    def test_current_event_has_no_decay(self):
        factors = compute_decay_factor([NOW], NOW, half_life_days=30.0)
        np.testing.assert_allclose(factors, [1.0])

    def test_at_half_life_decays_to_half(self):
        ts = NOW - 30 * DAY  # exactly 30 days ago
        factors = compute_decay_factor([ts], NOW, half_life_days=30.0)
        np.testing.assert_allclose(factors, [0.5], atol=1e-10)

    def test_two_half_lives_gives_quarter(self):
        ts = NOW - 60 * DAY
        factors = compute_decay_factor([ts], NOW, half_life_days=30.0)
        np.testing.assert_allclose(factors, [0.25], atol=1e-10)

    def test_zero_half_life_gives_no_decay(self):
        ts = NOW - 100 * DAY
        factors = compute_decay_factor([ts], NOW, half_life_days=0.0)
        np.testing.assert_allclose(factors, [1.0])

    def test_multiple_timestamps(self):
        timestamps = [NOW, NOW - 30 * DAY, NOW - 60 * DAY]
        factors = compute_decay_factor(timestamps, NOW, half_life_days=30.0)
        np.testing.assert_allclose(factors, [1.0, 0.5, 0.25], atol=1e-10)


class TestApplyInactivityDecay:
    def test_active_players_not_decayed(self):
        scores = np.array([1.0, 2.0, 3.0])
        last_activity = np.array([NOW, NOW, NOW])
        result = apply_inactivity_decay(scores, last_activity, NOW)
        np.testing.assert_allclose(result, scores)

    def test_within_delay_not_decayed(self):
        scores = np.array([1.0])
        last_activity = np.array([NOW - 20 * DAY])  # 20 days inactive < 30 day delay
        result = apply_inactivity_decay(
            scores, last_activity, NOW, delay_days=30.0
        )
        np.testing.assert_allclose(result, scores)

    def test_past_delay_is_decayed(self):
        scores = np.array([1.0])
        last_activity = np.array([NOW - 60 * DAY])  # 60 days inactive
        result = apply_inactivity_decay(
            scores, last_activity, NOW, delay_days=30.0, decay_rate=0.01
        )
        # 30 days of decay at rate 0.01
        expected = np.exp(-0.01 * 30.0)
        np.testing.assert_allclose(result, [expected], atol=1e-10)

    def test_mixed_activity(self):
        scores = np.array([10.0, 10.0])
        last_activity = np.array([NOW, NOW - 130 * DAY])
        result = apply_inactivity_decay(
            scores, last_activity, NOW, delay_days=30.0, decay_rate=0.01
        )
        assert result[0] == pytest.approx(10.0)
        assert result[1] < 10.0


class TestFilterByRecency:
    def test_filters_old_records(self):
        df = pl.DataFrame(
            {
                "ts": [NOW, NOW - 10 * DAY, NOW - 50 * DAY, NOW - 100 * DAY],
                "val": [1, 2, 3, 4],
            }
        )
        result = filter_by_recency(df, "ts", NOW, max_days=30)
        assert result.height == 2
        assert set(result["val"].to_list()) == {1, 2}


class TestCreateTimeWindows:
    def test_non_overlapping_windows(self):
        start = 0.0
        end = 10 * DAY
        windows = create_time_windows(start, end, window_days=2.0)
        assert len(windows) == 5
        assert windows[0] == (0.0, 2 * DAY)
        assert windows[-1][1] == end

    def test_overlapping_with_stride(self):
        start = 0.0
        end = 6 * DAY
        windows = create_time_windows(
            start, end, window_days=4.0, stride_days=2.0
        )
        # Strides: 0, 2d, 4d → windows at [0,4d], [2d,6d], [4d,6d]
        assert len(windows) == 3

    def test_single_window(self):
        windows = create_time_windows(0, DAY, window_days=10.0)
        assert len(windows) == 1
