"""Unified PageRank solver supporting both row and column stochastic orientations."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from loopr.core.config import PageRankConfig


def pagerank_sparse(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
    teleport: np.ndarray,
    cfg: PageRankConfig,
) -> np.ndarray:
    """Generic PageRank on sparse triplets.

    Supports row/col orientation with the same code path.

    Args:
        rows: Source node indices.
        cols: Target node indices.
        weights: Edge weights.
        num_nodes: Number of nodes.
        teleport: Teleport probability vector (must sum to 1).
        cfg: PageRank configuration.

    Returns:
        PageRank scores vector that sums to 1.
    """
    adjacency_matrix = sp.csr_matrix(
        (weights, (rows, cols)), shape=(num_nodes, num_nodes)
    )
    return pagerank_from_adjacency(adjacency_matrix, teleport, cfg)


def pagerank_from_adjacency(
    adjacency_matrix: sp.spmatrix,
    teleport: np.ndarray,
    cfg: PageRankConfig,
) -> np.ndarray:
    """Generic PageRank on a pre-built sparse adjacency matrix."""
    adjacency_matrix = adjacency_matrix.tocsr()

    sums = np.asarray(
        adjacency_matrix.sum(axis=1 if cfg.orientation == "row" else 0)
    ).ravel()

    inverse_sums = np.zeros_like(sums)
    nonzero_mask = sums > 0
    inverse_sums[nonzero_mask] = 1.0 / sums[nonzero_mask]

    # Pre-compute transpose once instead of per iteration
    if cfg.orientation == "row":
        _mat = adjacency_matrix.T.tocsr()
    else:
        _mat = adjacency_matrix

    def multiply(vector: np.ndarray) -> np.ndarray:
        return _mat.dot(vector * inverse_sums)

    rank_vector = teleport / teleport.sum()
    alpha = cfg.alpha

    for _ in range(cfg.max_iter):
        matrix_product = multiply(rank_vector)

        dangling_mass = (
            alpha * rank_vector[~nonzero_mask].sum()
            if cfg.redistribute_dangling
            else 0.0
        )

        new_rank = (
            alpha * matrix_product
            + (1 - alpha) * teleport
            + dangling_mass * teleport
        )
        new_rank /= new_rank.sum()

        if np.linalg.norm(new_rank - rank_vector, 1) < cfg.tol:
            return new_rank
        rank_vector = new_rank

    return rank_vector


def pagerank_dense(
    adjacency: np.ndarray,
    teleport: np.ndarray,
    cfg: PageRankConfig,
) -> np.ndarray:
    """PageRank on dense adjacency matrix (for backward compatibility).

    Args:
        adjacency: Dense adjacency matrix.
        teleport: Teleport probability vector.
        cfg: PageRank configuration.

    Returns:
        PageRank scores vector.
    """
    num_nodes = adjacency.shape[0]

    if cfg.orientation == "row":
        row_sums = adjacency.sum(axis=1)
        transition_matrix = np.zeros_like(adjacency)
        nonzero_rows = row_sums > 0
        transition_matrix[nonzero_rows] = (
            adjacency[nonzero_rows] / row_sums[nonzero_rows, np.newaxis]
        )
    else:
        col_sums = adjacency.sum(axis=0)
        transition_matrix = np.zeros_like(adjacency)
        nonzero_cols = col_sums > 0
        transition_matrix[:, nonzero_cols] = (
            adjacency[:, nonzero_cols] / col_sums[nonzero_cols]
        )

    rank_vector = teleport / teleport.sum()
    alpha = cfg.alpha

    for _ in range(cfg.max_iter):
        if cfg.orientation == "row":
            new_rank = (
                alpha * transition_matrix.T @ rank_vector
                + (1 - alpha) * teleport
            )
        else:
            new_rank = (
                alpha * transition_matrix @ rank_vector + (1 - alpha) * teleport
            )

        if cfg.redistribute_dangling:
            if cfg.orientation == "row":
                dangling_mass = alpha * rank_vector[row_sums == 0].sum()
            else:
                dangling_mass = alpha * rank_vector[col_sums == 0].sum()
            new_rank += dangling_mass * teleport

        new_rank /= new_rank.sum()

        if np.linalg.norm(new_rank - rank_vector, 1) < cfg.tol:
            return new_rank
        rank_vector = new_rank

    return rank_vector
