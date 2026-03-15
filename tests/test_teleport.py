"""Tests for teleport vector strategies (core/teleport.py)."""

import numpy as np
import polars as pl
import pytest

from loopr.core.teleport import (
    ActivePlayersTeleport,
    CustomTeleport,
    UniformTeleport,
    VolumeInverseTeleport,
    uniform,
    volume_inverse,
)


@pytest.fixture
def sample_edges():
    return pl.DataFrame(
        {
            "from_node": [1, 1, 1, 2, 2, 3],
            "to_node": [2, 3, 4, 3, 4, 4],
            "weight": [1.0] * 6,
        }
    )


class TestUniformTeleport:
    def test_sums_to_one(self):
        nodes = [1, 2, 3, 4, 5]
        result = UniformTeleport()(nodes, pl.DataFrame(), "")
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_all_equal(self):
        nodes = [10, 20, 30]
        result = UniformTeleport()(nodes, pl.DataFrame(), "")
        np.testing.assert_allclose(result, np.ones(3) / 3)

    def test_single_node(self):
        result = UniformTeleport()([42], pl.DataFrame(), "")
        np.testing.assert_allclose(result, [1.0])


class TestVolumeInverseTeleport:
    def test_sums_to_one(self, sample_edges):
        nodes = [1, 2, 3, 4]
        result = VolumeInverseTeleport()(nodes, sample_edges, "from_node")
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_low_degree_gets_higher_teleport(self, sample_edges):
        nodes = [1, 2, 3, 4]
        result = VolumeInverseTeleport()(nodes, sample_edges, "from_node")
        # Node 1 has 3 edges, node 3 has 1 edge → node 3 should have higher teleport
        assert result[nodes.index(3)] > result[nodes.index(1)]

    def test_missing_node_gets_default_weight(self, sample_edges):
        """A node with no edges gets count=1 default."""
        nodes = [1, 2, 3, 99]  # 99 not in edges
        result = VolumeInverseTeleport()(nodes, sample_edges, "from_node")
        np.testing.assert_allclose(result.sum(), 1.0)
        assert result[3] > 0  # node 99 still gets some probability


class TestCustomTeleport:
    def test_respects_provided_weights(self):
        weights = {1: 10.0, 2: 5.0, 3: 1.0}
        nodes = [1, 2, 3]
        result = CustomTeleport(weights)(nodes, pl.DataFrame(), "")
        np.testing.assert_allclose(result.sum(), 1.0)
        assert result[0] > result[1] > result[2]

    def test_missing_node_gets_default_1(self):
        weights = {1: 10.0}
        nodes = [1, 2]
        result = CustomTeleport(weights)(nodes, pl.DataFrame(), "")
        np.testing.assert_allclose(result.sum(), 1.0)
        # Node 2 gets default weight 1.0 vs node 1's 10.0
        assert result[0] > result[1]


class TestActivePlayersTeleport:
    def test_currently_returns_uniform(self):
        """ActivePlayersTeleport is a stub that falls back to uniform."""
        nodes = [1, 2, 3]
        result = ActivePlayersTeleport()(nodes, pl.DataFrame(), "")
        np.testing.assert_allclose(result, np.ones(3) / 3)


class TestConvenienceFunctions:
    def test_uniform_function(self):
        result = uniform([1, 2, 3, 4])
        np.testing.assert_allclose(result, np.ones(4) / 4)

    def test_volume_inverse_function(self, sample_edges):
        nodes = [1, 2, 3]
        result = volume_inverse(nodes, sample_edges, "from_node")
        np.testing.assert_allclose(result.sum(), 1.0)
