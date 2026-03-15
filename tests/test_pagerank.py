"""Tests for the PageRank solver (core/pagerank.py)."""

import numpy as np
import pytest

from loopr.core.config import PageRankConfig
from loopr.core.pagerank import pagerank_dense, pagerank_sparse


# ── helpers ──────────────────────────────────────────────────────────────────


def _simple_chain(n: int = 4):
    """Build a simple chain graph: 0→1→2→…→(n-1)→0."""
    rows = np.arange(n)
    cols = np.roll(np.arange(n), -1)
    weights = np.ones(n, dtype=float)
    return rows, cols, weights, n


def _star_graph():
    """Center node 0 receives edges from 1,2,3 (hub-spoke)."""
    rows = np.array([1, 2, 3])
    cols = np.array([0, 0, 0])
    weights = np.ones(3, dtype=float)
    return rows, cols, weights, 4


def _disconnected_pair():
    """Two disconnected components: 0↔1 and 2↔3."""
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 0, 3, 2])
    weights = np.ones(4, dtype=float)
    return rows, cols, weights, 4


# ── basic properties ─────────────────────────────────────────────────────────


class TestPageRankSparse:
    def test_sums_to_one(self):
        rows, cols, w, n = _simple_chain()
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85)
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-8)

    def test_uniform_on_symmetric_graph(self):
        """A symmetric cycle should yield uniform PageRank."""
        rows, cols, w, n = _simple_chain()
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85)
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        np.testing.assert_allclose(pr, np.ones(n) / n, atol=1e-6)

    def test_star_hub_gets_highest_rank(self):
        rows, cols, w, n = _star_graph()
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85, orientation="row")
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        assert pr[0] > pr[1]
        assert pr[0] > pr[2]
        assert pr[0] > pr[3]

    def test_disconnected_components(self):
        rows, cols, w, n = _disconnected_pair()
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85)
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-8)
        # Symmetric components: each pair should be equal
        np.testing.assert_allclose(pr[0], pr[1], atol=1e-8)
        np.testing.assert_allclose(pr[2], pr[3], atol=1e-8)

    def test_dangling_nodes_redistributed(self):
        """Node 2 has no outgoing edges → dangling mass redistributed."""
        rows = np.array([0, 1])
        cols = np.array([1, 2])
        w = np.ones(2, dtype=float)
        n = 3
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85, redistribute_dangling=True)
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-8)
        assert all(pr > 0), "All nodes should have positive rank with dangling redistribution"

    def test_col_stochastic_orientation(self):
        rows, cols, w, n = _star_graph()
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85, orientation="col")
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-8)
        assert all(pr > 0)

    def test_alpha_zero_gives_teleport(self):
        """With alpha=0, result should equal the teleport vector."""
        rows, cols, w, n = _simple_chain()
        teleport = np.array([0.4, 0.3, 0.2, 0.1])
        cfg = PageRankConfig(alpha=0.0)
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        np.testing.assert_allclose(pr, teleport / teleport.sum(), atol=1e-8)

    def test_higher_alpha_amplifies_structure(self):
        """Higher alpha should make the hub score higher relative to leaves."""
        rows, cols, w, n = _star_graph()
        teleport = np.ones(n) / n

        pr_low = pagerank_sparse(
            rows, cols, w, n, teleport, PageRankConfig(alpha=0.5)
        )
        pr_high = pagerank_sparse(
            rows, cols, w, n, teleport, PageRankConfig(alpha=0.95)
        )
        # Hub advantage should be larger with higher alpha
        assert (pr_high[0] - pr_high[1]) > (pr_low[0] - pr_low[1])

    def test_weighted_edges(self):
        """Heavier edge weight shifts rank when a node has multiple outgoing edges."""
        # Node 0 has two outgoing edges: 0→1 (weight 9) and 0→2 (weight 1)
        # Node 1 should receive more of node 0's rank than node 2
        rows = np.array([0, 0, 1, 2])
        cols = np.array([1, 2, 0, 0])
        n = 3
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85)

        w = np.array([9.0, 1.0, 1.0, 1.0])
        pr = pagerank_sparse(rows, cols, w, n, teleport, cfg)
        assert pr[1] > pr[2], "Node 1 should rank higher (receives more weight from node 0)"


class TestPageRankDense:
    def test_matches_sparse_on_chain(self):
        rows, cols, w, n = _simple_chain()
        teleport = np.ones(n) / n
        cfg = PageRankConfig(alpha=0.85)

        pr_sparse = pagerank_sparse(rows, cols, w, n, teleport, cfg)

        adj = np.zeros((n, n))
        for r, c, weight in zip(rows, cols, w):
            adj[r, c] += weight
        pr_dense = pagerank_dense(adj, teleport, cfg)

        np.testing.assert_allclose(pr_sparse, pr_dense, atol=1e-6)

    def test_col_stochastic_dense(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        teleport = np.ones(3) / 3
        cfg = PageRankConfig(alpha=0.85, orientation="col")
        pr = pagerank_dense(adj, teleport, cfg)
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-8)
