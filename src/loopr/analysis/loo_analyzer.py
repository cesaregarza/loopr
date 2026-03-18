"""Leave-One-Match-Out (LOO) Analyzer for ExposureLogOddsEngine.

This module implements exact, efficient leave-one-match-out impact analysis
using low-rank PageRank updates via Sherman-Morrison-Woodbury formula.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import _loo_numba

logger = logging.getLogger(__name__)

EPS = 1e-15
DANGLING_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class EntityMatchRef:
    """Compact reference from an entity to one of its matches."""

    match_id: int
    is_win: bool


@dataclass(slots=True)
class CachedMatch:
    """Compact per-match cache for exact LOO updates."""

    winner_indices: np.ndarray
    loser_indices: np.ndarray
    participant_indices: np.ndarray
    pair_weight: float
    sigma: float

    def triplets(self, graph: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Materialize sparse triplets for the requested graph on demand."""
        if self.winner_indices.size == 0 or self.loser_indices.size == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            return empty_i32, empty_i32, empty_f64

        if graph == "win":
            rows = np.repeat(
                self.winner_indices, self.loser_indices.size
            ).astype(np.int32, copy=False)
            cols = np.tile(
                self.loser_indices, self.winner_indices.size
            ).astype(np.int32, copy=False)
        elif graph == "loss":
            rows = np.repeat(
                self.loser_indices, self.winner_indices.size
            ).astype(np.int32, copy=False)
            cols = np.tile(
                self.winner_indices, self.loser_indices.size
            ).astype(np.int32, copy=False)
        else:
            raise ValueError(f"Unknown graph {graph}")

        weights = np.full(rows.size, self.pair_weight, dtype=np.float64)
        return rows, cols, weights

    def delta_rho(self, n: int, *, use_numba: bool = False) -> np.ndarray | None:
        """Materialize the dense teleport delta only when exact solve needs it."""
        if self.sigma == 0.0 or self.participant_indices.size == 0:
            return None

        if use_numba and _loo_numba.NUMBA_AVAILABLE:
            return _loo_numba.delta_rho_numba(
                self.participant_indices,
                self.sigma,
                n,
            )

        delta = np.zeros(n, dtype=np.float64)
        delta[self.participant_indices] -= self.sigma
        return delta

    def estimated_bytes(self) -> int:
        """Approximate in-memory footprint of the cached record."""
        return (
            self.winner_indices.nbytes
            + self.loser_indices.nbytes
            + self.participant_indices.nbytes
            + 16
        )


# -------------------------
# Linear Solver Backend
# -------------------------


class LinearSolveBackend:
    """
    Pre-factorized solver for (I - alpha A) X = B with reusable solves.
    method:
      - 'splu'   : sparse LU (fastest, most robust)
      - 'gmres'  : GMRES with ILU preconditioner (less memory if LU is large)
    """

    def __init__(
        self, adjacency_csc: sp.csc_matrix, alpha: float, method: str = "splu"
    ):
        self.num_nodes = adjacency_csc.shape[0]
        system_matrix = (
            sp.eye(self.num_nodes, format="csc") - alpha * adjacency_csc
        )

        if method == "splu":
            self._lu = spla.splu(system_matrix)
            self._solve = lambda rhs: self._lu.solve(rhs)
        elif method == "gmres":
            ilu = spla.spilu(system_matrix, drop_tol=1e-4, fill_factor=10)
            preconditioner = spla.LinearOperator(
                system_matrix.shape, matvec=lambda x: ilu.solve(x)
            )

            def _solve(rhs):
                if rhs.ndim == 1:
                    rhs = rhs[:, None]
                solution = np.empty_like(rhs, dtype=float)
                for col_idx in range(rhs.shape[1]):
                    x, info = spla.gmres(
                        system_matrix,
                        rhs[:, col_idx],
                        M=preconditioner,
                        tol=1e-10,
                        restart=50,
                        maxiter=200,
                    )
                    if info != 0:
                        logger.warning(
                            f"GMRES did not fully converge (col {col_idx}, info={info})"
                        )
                    solution[:, col_idx] = x
                return solution

            self._solve = _solve
        else:
            raise ValueError(f"Unknown method {method}")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return X solving (I - alpha A) X = rhs. Accepts (n,) or (n,k)."""
        return self._solve(rhs)


# -------------------------
# Matrix Construction Helpers
# -------------------------


def _normalize_and_fill_dangling(adjacency_csr, teleport_vector):
    """Efficiently normalize columns and fill dangling with teleport vector."""
    adjacency = adjacency_csr.tocsc()
    col_sums = np.asarray(adjacency.sum(axis=0)).ravel()
    counts = np.diff(adjacency.indptr)
    inverse_sums = np.zeros_like(col_sums, dtype=float)
    nonzero_mask = col_sums > 0
    inverse_sums[nonzero_mask] = 1.0 / col_sums[nonzero_mask]
    adjacency.data *= inverse_sums.repeat(counts)

    if (~nonzero_mask).any():
        dangling_cols = np.where(~nonzero_mask)[0]
        data = np.tile(teleport_vector, len(dangling_cols))
        row_idx = np.tile(np.arange(len(teleport_vector)), len(dangling_cols))
        col_idx = np.repeat(dangling_cols, len(teleport_vector))
        dangling_matrix = sp.csc_matrix(
            (data, (row_idx, col_idx)), shape=adjacency.shape
        )
        adjacency = adjacency + dangling_matrix
    return adjacency, col_sums


def get_sparse_matrices(
    engine,
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    teleport_vector: np.ndarray,
) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray, np.ndarray]:
    """
    Build sparse adjacency matrices from engine state with dangling redistribution.

    Returns:
        A_win: Win graph adjacency (column-stochastic with dangling → teleport)
        A_loss: Loss graph adjacency (column-stochastic with dangling → teleport)
        T_win: Raw column sums for win graph
        T_loss: Raw column sums for loss graph
    """
    from loopr.core import build_exposure_triplets

    rows, cols, data = build_exposure_triplets(matches_df, node_to_idx)
    num_nodes = len(node_to_idx)

    adjacency_win_raw = sp.csr_matrix(
        (data, (rows, cols)), shape=(num_nodes, num_nodes)
    )
    adjacency_win, col_sums_win = _normalize_and_fill_dangling(
        adjacency_win_raw, teleport_vector
    )

    adjacency_loss_raw = sp.csr_matrix(
        (data, (cols, rows)), shape=(num_nodes, num_nodes)
    )
    adjacency_loss, col_sums_loss = _normalize_and_fill_dangling(
        adjacency_loss_raw, teleport_vector
    )

    return adjacency_win, adjacency_loss, col_sums_win, col_sums_loss


def exposures_for_match(
    match_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    graph: str = "win",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns triplets (rows_i, cols_j, raw_weights) for a specific match.

    Args:
        match_id: Match identifier
        matches_df: Processed matches with winners/losers lists
        players_df: Player data (not used in current implementation)
        node_to_idx: Mapping from player IDs to node indices
        graph: "win" or "loss" to specify which graph

    Returns:
        rows_i: Destination node indices
        cols_j: Source node indices
        raw_weights: Raw edge weights for this match
    """
    # Get match data
    match = matches_df.filter(pl.col("match_id") == match_id)
    if match.is_empty():
        return np.array([]), np.array([]), np.array([])

    match_data = match.to_dicts()[0]

    # Extract match weight (includes time decay and tournament influence)
    # Check for different possible column names
    if "weight" in match_data:
        match_weight = float(match_data["weight"])
    elif "w_m" in match_data:
        match_weight = float(match_data["w_m"])
    else:
        # Fallback: try to find any weight-like column
        logger.warning(f"No weight column found for match {match_id}")
        return np.array([]), np.array([]), np.array([])

    # Get winner and loser player IDs
    winners = match_data.get("winners", [])
    losers = match_data.get("losers", [])

    # Ensure they are lists
    if not isinstance(winners, list):
        winners = [winners] if winners is not None else []
    if not isinstance(losers, list):
        losers = [losers] if losers is not None else []

    # Map to node indices
    winner_indices = [node_to_idx[w] for w in winners if w in node_to_idx]
    loser_indices = [node_to_idx[l] for l in losers if l in node_to_idx]

    if graph == "win":
        # Win graph: edges from losers to winners
        rows_i = []
        cols_j = []
        raw_weights = []

        for winner_idx in winner_indices:
            for loser_idx in loser_indices:
                rows_i.append(winner_idx)
                cols_j.append(loser_idx)
                # Raw weight per edge (distributed across all pairs)
                raw_weights.append(match_weight / (len(winners) * len(losers)))

        return (
            np.array(rows_i, dtype=np.int32),
            np.array(cols_j, dtype=np.int32),
            np.array(raw_weights, dtype=np.float64),
        )

    else:  # graph == "loss"
        # Loss graph: edges from winners to losers
        rows_i = []
        cols_j = []
        raw_weights = []

        for loser_idx in loser_indices:
            for winner_idx in winner_indices:
                rows_i.append(loser_idx)
                cols_j.append(winner_idx)
                raw_weights.append(match_weight / (len(winners) * len(losers)))

        return (
            np.array(rows_i, dtype=np.int32),
            np.array(cols_j, dtype=np.int32),
            np.array(raw_weights, dtype=np.float64),
        )


def _compute_total_exposure_mass(matches_df: pl.DataFrame) -> float:
    """
    Compute total exposure mass E = sum over matches of (share * num_participants).
    """
    total = 0.0
    for row in matches_df.iter_rows(named=True):
        share = float(row.get("share", 0.0))
        winners = row.get("winners", []) or []
        losers = row.get("losers", []) or []
        if not isinstance(winners, list):
            winners = [winners] if winners is not None else []
        if not isinstance(losers, list):
            losers = [losers] if losers is not None else []
        total += share * (len(winners) + len(losers))
    return max(total, 1e-12)


def delta_rho_for_match(
    match_id: int,
    matches_df: pl.DataFrame,
    players_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    total_exposure: float,
) -> np.ndarray | None:
    """
    Returns the change in teleport vector if this match is removed.

    Computes change in normalized space: delta = -sigma for each participant,
    where sigma = share / total_exposure.

    Args:
        match_id: Match identifier
        matches_df: Processed matches with share values
        players_df: Player data (not used in current implementation)
        node_to_idx: Mapping from player IDs to node indices
        total_exposure: Total exposure mass E

    Returns:
        Delta rho vector (negative values since we're removing)
    """
    match = matches_df.filter(pl.col("match_id") == match_id)
    if match.is_empty():
        return np.zeros(len(node_to_idx))

    match_data = match.to_dicts()[0]

    # Get share value for this match
    share = float(match_data.get("share", 0.0))
    if share == 0.0:
        return None  # No teleport change

    # Get all players involved
    winners = match_data.get("winners", [])
    losers = match_data.get("losers", [])

    if not isinstance(winners, list):
        winners = [winners] if winners is not None else []
    if not isinstance(losers, list):
        losers = [losers] if losers is not None else []

    # Compute change in normalized space
    sigma = share / total_exposure
    delta_rho = np.zeros(len(node_to_idx))

    # Each participant loses sigma in the normalized space
    for player_id in winners + losers:
        if player_id in node_to_idx:
            idx = node_to_idx[player_id]
            delta_rho[idx] -= sigma

    return delta_rho


# -------------------------
# Block Resolvent Solver
# -------------------------


def block_resolvent_fixed_point(
    A_csc: sp.csc_matrix,
    alpha: float,
    U: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 400,
) -> np.ndarray:
    """
    Solve (I - alpha A) X = U for multiple RHS using fixed-point iteration.

    Uses contractive fixed-point: X_{t+1} = alpha * A @ X_t + U
    Converges linearly since ||alpha*A||_1 <= alpha < 1

    Args:
        A_csc: (n,n) CSC sparse, column-stochastic
        alpha: Damping factor
        U: (n,m) dense RHS matrix
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        X = (I - alpha A)^{-1} U
    """
    n, m = U.shape
    X = np.zeros((n, m), dtype=U.dtype)

    for it in range(max_iter):
        X_new = alpha * (A_csc @ X) + U

        # Relative 1-norm stopping criterion
        num = np.linalg.norm(X_new - X, ord=1)
        den = 1.0 + np.linalg.norm(X_new, ord=1)

        if num < tol * den:
            return X_new

        X = X_new

    logger.warning(
        f"Fixed-point iteration did not converge in {max_iter} iterations"
    )
    return X


def block_resolvent_neumann(
    A_csc: sp.csc_matrix,
    alpha: float,
    U: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """
    Truncated Neumann approximation of (I - alpha A)^-1 U.

    This is a benchmark-oriented approximation path for explanation ranking.
    It trades exactness for a fixed number of sparse matvecs.
    """
    if U.ndim == 1:
        U = U[:, None]

    X = U.astype(float, copy=True)
    term = X.copy()
    for _ in range(1, max(steps, 1)):
        term = alpha * (A_csc @ term)
        X += term
    return X


# -------------------------
# Column Update Construction
# -------------------------


def build_U_alpha_for_graph(
    A_csc: sp.csc_matrix,
    T_col_raw: np.ndarray,
    rho: np.ndarray,
    rho_new: np.ndarray,  # New teleport after removal
    alpha: float,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    n: int,
) -> tuple[List[int], np.ndarray]:
    """
    Build per-source-column updates u_j for a match removal.

    Args:
        A_csc: Current adjacency matrix (column-stochastic)
        T_col_raw: Raw column sums before normalization
        rho: Teleport vector
        alpha: Damping factor
        rows, cols, weights: Match triplets
        n: Number of nodes

    Returns:
        j_list: List of affected source columns
        U: (n, k) matrix with columns alpha*u_j
    """
    # Group match contributions by source column j
    by_col = {}
    for i, j, w in zip(rows, cols, weights):
        if T_col_raw[j] <= 0:
            # Raw column sum was zero; skip
            continue
        by_col.setdefault(int(j), []).append((int(i), float(w)))

    j_list = sorted(by_col.keys())
    k = len(j_list)

    if k == 0:
        return j_list, np.zeros((n, 0), dtype=float)

    U = np.zeros((n, k), dtype=float)

    for c, j in enumerate(j_list):
        entries = by_col[j]
        Tj = float(T_col_raw[j])

        # Current normalized column a_j (dense)
        aj = np.zeros(n, dtype=float)
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        aj[A_csc.indices[start:end]] = A_csc.data[start:end]

        sum_w = sum(w for (_i, w) in entries)
        delta = sum_w / Tj  # Fraction of column mass removed

        if delta >= 1.0 - DANGLING_EPS:
            # Column becomes dangling -> replace with rho_new
            u = rho_new - aj
        else:
            # Normal update
            rj = np.zeros(n, dtype=float)
            for i, w in entries:
                rj[i] += w / Tj
            scale = 1.0 / (1.0 - delta)
            u = scale * (delta * aj - rj)

        U[:, c] = alpha * u

    return j_list, U


# -------------------------
# Exact Rank-k Update
# -------------------------


def loo_update_graph_exact(
    A_csc: sp.csc_matrix,
    s: np.ndarray,
    rho: np.ndarray,
    alpha: float,
    T_col_raw: np.ndarray,
    match_rows: np.ndarray,
    match_cols: np.ndarray,
    match_weights: np.ndarray,
    delta_rho_vec: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 400,
    R_solve=None,  # Pre-factorized solver for (I - alpha A)^{-1}
    solve_strategy: str = "exact",
    combine_rhs: bool = True,
    approx_steps: int = 3,
    check_conditioning: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    LOO update for a single match on a single PageRank vector.

    Returns:
        s_new: Updated PageRank vector
        aux: Dictionary with auxiliary information
    """
    n = A_csc.shape[0]

    # Compute new teleport first if needed
    if delta_rho_vec is not None and np.any(delta_rho_vec):
        rho_tmp = rho + delta_rho_vec
        rho_tmp = np.maximum(rho_tmp, 0.0)  # Enforce non-negativity
        total = float(rho_tmp.sum())
        rho_new = rho_tmp / total if total > 0.0 else rho.copy()
    else:
        rho_new = rho

    # Build per-column updates
    j_list, U = build_U_alpha_for_graph(
        A_csc,
        T_col_raw,
        rho,
        rho_new,
        alpha,
        match_rows,
        match_cols,
        match_weights,
        n,
    )
    k = U.shape[1]
    teleport_requested = delta_rho_vec is not None and np.any(delta_rho_vec)

    if k == 0 and not teleport_requested:
        return s.copy(), {"k": 0, "teleport_applied": False}

    v = None
    teleport_applied = False
    if teleport_requested:
        v = (1.0 - alpha) * (rho_new - rho)
        teleport_norm = (1.0 - alpha) * np.linalg.norm(delta_rho_vec, 1)
        teleport_applied = teleport_norm >= 1e-12

    def _solve_rhs(rhs: np.ndarray) -> np.ndarray:
        if solve_strategy == "exact":
            if R_solve is not None:
                return R_solve(rhs)
            return block_resolvent_fixed_point(
                A_csc, alpha, rhs, tol=tol, max_iter=max_iter
            )
        if solve_strategy == "neumann":
            return block_resolvent_neumann(
                A_csc,
                alpha,
                rhs,
                steps=approx_steps,
            )
        raise ValueError(f"Unknown solve_strategy: {solve_strategy}")

    X = np.zeros((n, 0), dtype=float)
    y = None
    if combine_rhs and k > 0 and teleport_applied:
        solved = _solve_rhs(np.column_stack([U, v.reshape(-1, 1)]))
        X = solved[:, :k]
        y = solved[:, k]
    else:
        if k > 0:
            X = _solve_rhs(U)
        if teleport_applied:
            y = _solve_rhs(v.reshape(-1, 1))[:, 0]

    if k > 0:
        X_rows = X[j_list, :]
        K = np.eye(k, dtype=float) - X_rows
        if check_conditioning:
            try:
                cond = np.linalg.cond(K)
                if not np.isfinite(cond) or cond > 1e8:
                    logger.warning(
                        "K matrix is ill-conditioned: cond=%.3e. Results may be inaccurate.",
                        cond,
                    )
            except Exception:
                pass
        beta = np.linalg.solve(K, s[j_list])
        s_star = s + X @ beta
    else:
        K = np.zeros((0, 0), dtype=float)
        s_star = s.copy()

    if teleport_applied and y is not None:
        if k > 0:
            gamma = np.linalg.solve(K, y[j_list])
            s_new = s_star + y + X @ gamma
        else:
            s_new = s_star + y
        return s_new, {
            "k": k,
            "teleport_applied": True,
            "rho_new": rho_new,
            "j_list": j_list,
            "K": K,
            "solve_strategy": solve_strategy,
            "combine_rhs": combine_rhs,
        }

    return s_star, {
        "k": k,
        "teleport_applied": False,
        "rho_new": rho_new if teleport_requested else rho,
        "j_list": j_list,
        "K": K,
        "solve_strategy": solve_strategy,
        "combine_rhs": combine_rhs,
    }


# -------------------------
# Main LOO Analyzer Class
# -------------------------


class LOOAnalyzer:
    """
    Leave-One-Match-Out analyzer for ExposureLogOddsEngine.

    Computes exact impact of removing a single match on entity scores
    using efficient rank-k updates instead of full PageRank recomputation.
    """

    def _check_pagerank_validity(
        self, name: str, s: np.ndarray, rho: np.ndarray
    ) -> None:
        """
        Check PageRank vector validity and log warnings for invariant violations.

        Args:
            name: Name of the PageRank vector (for logging)
            s: PageRank vector to check
            rho: Teleport vector
        """
        # Check for negative entries
        if np.any(s < -1e-12):
            min_val = float(s.min())
            logger.warning(f"{name} has negative entries: min={min_val:.3e}")

        # Check normalization
        ssum = float(s.sum())
        if not (0.9999 <= ssum <= 1.0001):
            logger.warning(f"{name} not normalized: sum={ssum:.6f}")

        # Check against theoretical lower bound
        lower_bound = (1.0 - self.alpha) * rho
        violations = s < (lower_bound - 1e-6)
        if np.any(violations):
            num_violations = int(violations.sum())
            max_violation = float((lower_bound - s)[violations].max())
            logger.debug(
                f"{name} violates (1-α)ρ bound at {num_violations} entries, max violation={max_violation:.3e}"
            )

    def __init__(
        self, engine, matches_df: pl.DataFrame, players_df: pl.DataFrame
    ):
        """
        Initialize LOO analyzer from engine state.

        Args:
            engine: ExposureLogOddsEngine instance after rank_entities()
            matches_df: Processed matches DataFrame with winners/losers lists
            players_df: Participant DataFrame used for node expansion
        """
        # Core parameters
        self.alpha = float(engine.config.pagerank.alpha)
        self.rho = engine._last_rho.astype(float)
        self.lam = float(engine.last_result.lambda_used)

        # Node mapping
        self.node_ids = engine.last_result.ids
        self.node_to_idx = {pid: i for i, pid in enumerate(self.node_ids)}
        self.n = len(self.node_ids)

        # Current PageRank vectors
        self.s_win = engine.last_result.win_pagerank.astype(float)
        self.s_loss = engine.last_result.loss_pagerank.astype(float)

        # Store actual scores from engine
        self.actual_scores = engine.last_result.scores.astype(float)
        self._baseline_scores = self._compute_score_vector(
            self.s_win, self.s_loss, self.rho
        )
        self._run_validity_checks = engine.logger.isEnabledFor(logging.DEBUG)

        # Build sparse matrices
        logger.info("Building sparse adjacency matrices...")
        A_win, A_loss, T_win, T_loss = get_sparse_matrices(
            engine, matches_df, self.node_to_idx, self.rho
        )
        self.A_win = A_win.tocsc()
        self.A_loss = A_loss.tocsc()
        self.T_win = T_win.astype(float)
        self.T_loss = T_loss.astype(float)

        self.matches_df = matches_df
        self.players_df = players_df

        # Compute total exposure mass for teleport updates
        self._total_exposure = _compute_total_exposure_mass(matches_df)

        # Pre-factorize linear solvers for massive speedup
        logger.info("Pre-factorizing linear solvers...")
        self._win_solver = LinearSolveBackend(
            self.A_win, self.alpha, method="splu"
        )
        self._loss_solver = LinearSolveBackend(
            self.A_loss, self.alpha, method="splu"
        )

        # Build match cache for fast triplet/teleport lookup
        logger.info("Building match cache...")
        self._match_cache, self._entity_match_index = self._build_match_cache()

        logger.info(
            f"LOOAnalyzer initialized: {self.n} nodes, "
            f"{self.A_win.nnz} win edges, {self.A_loss.nnz} loss edges"
        )

    def _build_match_cache(
        self,
    ) -> tuple[
        dict[int, CachedMatch | None], dict[int, tuple[EntityMatchRef, ...]]
    ]:
        """Pre-compute compact per-match records and entity-to-match lookup."""
        cache: dict[int, CachedMatch | None] = {}
        entity_match_index: dict[int, list[EntityMatchRef]] = defaultdict(list)

        for row in self.matches_df.iter_rows(named=True):
            mid = row["match_id"]
            winners = row.get("winners", []) or []
            losers = row.get("losers", []) or []
            if not isinstance(winners, list):
                winners = [winners] if winners is not None else []
            if not isinstance(losers, list):
                losers = [losers] if losers is not None else []
            weight = float(row.get("weight", row.get("w_m", 0.0)))
            share = float(row.get("share", 0.0))

            w_idx = [
                self.node_to_idx[p] for p in winners if p in self.node_to_idx
            ]
            l_idx = [
                self.node_to_idx[p] for p in losers if p in self.node_to_idx
            ]
            if not w_idx or not l_idx:
                cache[mid] = None
                continue

            participant_indices = np.array(w_idx + l_idx, dtype=np.int32)
            pair_weight = weight / (len(w_idx) * len(l_idx))
            sigma = share / self._total_exposure if self._total_exposure > 0 else 0.0

            cache[mid] = CachedMatch(
                winner_indices=np.array(w_idx, dtype=np.int32),
                loser_indices=np.array(l_idx, dtype=np.int32),
                participant_indices=participant_indices,
                pair_weight=float(pair_weight),
                sigma=float(sigma),
            )

            for player_id in winners:
                if player_id in self.node_to_idx:
                    entity_match_index[player_id].append(
                        EntityMatchRef(match_id=mid, is_win=True)
                    )
            for player_id in losers:
                if player_id in self.node_to_idx:
                    entity_match_index[player_id].append(
                        EntityMatchRef(match_id=mid, is_win=False)
                    )

        frozen_index = {
            entity_id: tuple(refs)
            for entity_id, refs in entity_match_index.items()
        }
        return cache, frozen_index

    def _compute_score_at(
        self,
        s_win: np.ndarray,
        s_loss: np.ndarray,
        rho_vec: np.ndarray,
        idx: int,
    ) -> float:
        """Compute one log-odds score from PageRank state."""
        sw = max(float(s_win[idx]), 0.0)
        sl = max(float(s_loss[idx]), 0.0)

        min_floor = 0.5 * (1.0 - self.alpha) * rho_vec[idx]
        w = max(sw, min_floor) + self.lam * rho_vec[idx]
        l = max(sl, min_floor) + self.lam * rho_vec[idx]
        return float(np.log(w / l))

    def _compute_score_vector(
        self,
        s_win: np.ndarray,
        s_loss: np.ndarray,
        rho_vec: np.ndarray,
    ) -> np.ndarray:
        """Compute the full log-odds vector for one PageRank state."""
        sw = np.maximum(s_win.astype(float, copy=False), 0.0)
        sl = np.maximum(s_loss.astype(float, copy=False), 0.0)
        min_floor = 0.5 * (1.0 - self.alpha) * rho_vec
        w = np.maximum(sw, min_floor) + self.lam * rho_vec
        l = np.maximum(sl, min_floor) + self.lam * rho_vec
        return np.log(w / l)

    def estimate_cache_bytes(self) -> int:
        """Approximate the in-memory footprint of LOO cache structures."""
        total = 0
        for match_data in self._match_cache.values():
            if match_data is not None:
                total += match_data.estimated_bytes()
        for refs in self._entity_match_index.values():
            total += len(refs) * 16
        return total

    def exposures_for_match(
        self, match_id: int, graph: str = "win"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get match triplets for specified graph."""
        match_data = self._match_cache.get(match_id)
        if match_data is None:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            return empty_i32, empty_i32, empty_f64
        return match_data.triplets(graph)

    def delta_rho_for_match(self, match_id: int) -> np.ndarray | None:
        """Get teleport vector change for match removal."""
        match_data = self._match_cache.get(match_id)
        if match_data is None:
            return np.zeros(self.n)
        return match_data.delta_rho(self.n)

    def node_index_for_entity(self, entity_id: int) -> Optional[int]:
        """Get node index for an entity ID."""
        return self.node_to_idx.get(entity_id, None)

    def _estimate_match_flux(
        self, match_id: int, entity_id: int, *, use_numba: bool = False
    ) -> float:
        """
        Estimate flux for a match without full LOO computation.

        This uses the current PageRank values to estimate how much flux
        flows through the edges created by this match.

        Args:
            match_id: Match to estimate
            entity_id: Entity of interest

        Returns:
            Estimated flux magnitude
        """
        # Get cached match data
        match_data = self._match_cache.get(match_id)
        if match_data is None:
            return 0.0

        entity_idx = self.node_index_for_entity(entity_id)
        if entity_idx is None:
            return 0.0

        winners = match_data.winner_indices
        losers = match_data.loser_indices
        pair_weight = match_data.pair_weight
        if use_numba and _loo_numba.NUMBA_AVAILABLE:
            return float(
                _loo_numba.estimate_match_flux_numba(
                    entity_idx,
                    winners,
                    losers,
                    pair_weight,
                    self.alpha,
                    self.s_win,
                    self.s_loss,
                    self.T_win,
                    self.T_loss,
                )
            )

        total_flux = 0.0

        if np.any(winners == entity_idx):
            denom = self.T_win[losers]
            valid = denom > 0
            if np.any(valid):
                total_flux += self.alpha * pair_weight * float(
                    np.sum(self.s_win[losers[valid]] / denom[valid])
                )
        elif np.any(losers == entity_idx) and self.T_win[entity_idx] > 0:
            total_flux += (
                winners.size
                * self.alpha
                * self.s_win[entity_idx]
                * (pair_weight / self.T_win[entity_idx])
            )

        if np.any(losers == entity_idx):
            denom = self.T_loss[winners]
            valid = denom > 0
            if np.any(valid):
                total_flux += self.alpha * pair_weight * float(
                    np.sum(self.s_loss[winners[valid]] / denom[valid])
                )
        elif np.any(winners == entity_idx) and self.T_loss[entity_idx] > 0:
            total_flux += (
                losers.size
                * self.alpha
                * self.s_loss[entity_idx]
                * (pair_weight / self.T_loss[entity_idx])
            )

        return total_flux

    def _format_impact_result(
        self, match_ref: EntityMatchRef, impact: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not impact["ok"]:
            return None
        return {
            "match_id": match_ref.match_id,
            "is_win": match_ref.is_win,
            "old_score": impact["old"]["score"],
            "new_score": impact["new"]["score"],
            "score_delta": impact["delta"]["score"],
            "abs_delta": abs(impact["delta"]["score"]),
            "win_pr_delta": impact["delta"]["s_win"],
            "loss_pr_delta": impact["delta"]["s_loss"],
        }

    def _impact_of_match_on_entity_variant(
        self,
        match_id: int,
        entity_id: int,
        *,
        include_teleport: bool,
        solve_strategy: str,
        combine_rhs: bool,
        approx_steps: int,
        use_numba: bool,
        tol: float,
        max_iter: int,
    ) -> dict[str, Any]:
        k = self.node_index_for_entity(entity_id)
        if k is None:
            return {
                "ok": False,
                "reason": f"entity_id {entity_id} not found in node mapping",
            }

        match_data = self._match_cache.get(match_id)
        if match_data is None:
            return {
                "ok": False,
                "reason": f"match_id {match_id} not found or has no valid edges",
            }

        rows_w, cols_w, wts_w = match_data.triplets("win")
        rows_l, cols_l, wts_l = match_data.triplets("loss")
        delta_rho = (
            match_data.delta_rho(self.n, use_numba=use_numba)
            if include_teleport
            else None
        )

        flux_estimate = self._estimate_match_flux(
            match_id, entity_id, use_numba=use_numba
        )
        if abs(flux_estimate) < 1e-12:
            old_score = float(self.actual_scores[k])
            return {
                "ok": True,
                "entity_id": entity_id,
                "match_id": match_id,
                "old": {
                    "score": old_score,
                    "s_win": float(self.s_win[k]),
                    "s_loss": float(self.s_loss[k]),
                    "rho": float(self.rho[k]),
                },
                "new": {
                    "score": old_score,
                    "s_win": float(self.s_win[k]),
                    "s_loss": float(self.s_loss[k]),
                    "rho": float(self.rho[k]),
                },
                "delta": {
                    "score": 0.0,
                    "s_win": 0.0,
                    "s_loss": 0.0,
                    "rho": 0.0,
                },
                "internals": {
                    "alpha": self.alpha,
                    "lambda_smooth": self.lam,
                    "k_win_columns": 0,
                    "k_loss_columns": 0,
                    "teleport_applied": False,
                    "flux_estimate": flux_estimate,
                    "early_exit": "negligible_flux",
                    "solve_strategy": solve_strategy,
                    "combine_rhs": combine_rhs,
                    "approx_steps": approx_steps,
                    "use_numba": use_numba,
                },
            }

        s_win_new, aux_win = loo_update_graph_exact(
            self.A_win,
            self.s_win,
            self.rho,
            self.alpha,
            self.T_win,
            rows_w,
            cols_w,
            wts_w,
            delta_rho_vec=delta_rho,
            tol=tol,
            max_iter=max_iter,
            R_solve=self._win_solver.solve,
            solve_strategy=solve_strategy,
            combine_rhs=combine_rhs,
            approx_steps=approx_steps,
            check_conditioning=self._run_validity_checks,
        )
        if self._run_validity_checks:
            self._check_pagerank_validity(
                "s_win_new", s_win_new, aux_win.get("rho_new", self.rho)
            )

        s_loss_new, aux_loss = loo_update_graph_exact(
            self.A_loss,
            self.s_loss,
            self.rho,
            self.alpha,
            self.T_loss,
            rows_l,
            cols_l,
            wts_l,
            delta_rho_vec=delta_rho,
            tol=tol,
            max_iter=max_iter,
            R_solve=self._loss_solver.solve,
            solve_strategy=solve_strategy,
            combine_rhs=combine_rhs,
            approx_steps=approx_steps,
            check_conditioning=self._run_validity_checks,
        )
        if self._run_validity_checks:
            self._check_pagerank_validity(
                "s_loss_new", s_loss_new, aux_loss.get("rho_new", self.rho)
            )

        rho_new = aux_win.get("rho_new", self.rho) if include_teleport else self.rho
        old_score = float(self.actual_scores[k])
        new_score_computed = self._compute_score_at(
            s_win_new, s_loss_new, rho_new, k
        )
        score_delta = new_score_computed - float(self._baseline_scores[k])
        new_score = old_score + score_delta

        return {
            "ok": True,
            "entity_id": entity_id,
            "match_id": match_id,
            "old": {
                "score": old_score,
                "s_win": float(self.s_win[k]),
                "s_loss": float(self.s_loss[k]),
                "rho": float(self.rho[k]),
            },
            "new": {
                "score": new_score,
                "s_win": float(s_win_new[k]),
                "s_loss": float(s_loss_new[k]),
                "rho": float(rho_new[k]),
            },
            "delta": {
                "score": score_delta,
                "s_win": float(s_win_new[k] - self.s_win[k]),
                "s_loss": float(s_loss_new[k] - self.s_loss[k]),
                "rho": float(rho_new[k] - self.rho[k]),
            },
            "internals": {
                "alpha": self.alpha,
                "lambda_smooth": self.lam,
                "k_win_columns": aux_win["k"],
                "k_loss_columns": aux_loss["k"],
                "teleport_applied": aux_win["teleport_applied"]
                or aux_loss["teleport_applied"],
                "solve_strategy": solve_strategy,
                "combine_rhs": combine_rhs,
                "approx_steps": approx_steps,
                "use_numba": use_numba,
            },
        }

    def impact_of_match_on_entity(
        self,
        match_id: int,
        entity_id: int,
        include_teleport: bool = True,
        tol: float = 1e-10,
        max_iter: int = 400,
    ) -> dict[str, Any]:
        """
        Compute exact change in an entity's log-odds score if match is removed.

        Args:
            match_id: Match to remove
            entity_id: Entity to analyze
            include_teleport: Whether to account for teleport vector changes
            tol: Convergence tolerance for solvers
            max_iter: Maximum iterations for solvers

        Returns:
            Dictionary with old/new scores and detailed components
        """
        return self._impact_of_match_on_entity_variant(
            match_id,
            entity_id,
            include_teleport=include_teleport,
            solve_strategy="exact",
            combine_rhs=True,
            approx_steps=3,
            use_numba=False,
            tol=tol,
            max_iter=max_iter,
        )

    def impact_of_match_on_entity_variant(
        self,
        match_id: int,
        entity_id: int,
        *,
        variant: str,
        include_teleport: bool = True,
        tol: float = 1e-10,
        max_iter: int = 400,
    ) -> dict[str, Any]:
        """Internal variant hook for benchmark and profiling experiments."""
        variant_map = {
            "exact_separate": {
                "solve_strategy": "exact",
                "combine_rhs": False,
                "approx_steps": 3,
                "use_numba": False,
            },
            "exact_combined": {
                "solve_strategy": "exact",
                "combine_rhs": True,
                "approx_steps": 3,
                "use_numba": False,
            },
            "perturb_2": {
                "solve_strategy": "neumann",
                "combine_rhs": True,
                "approx_steps": 2,
                "use_numba": False,
            },
            "perturb_4": {
                "solve_strategy": "neumann",
                "combine_rhs": True,
                "approx_steps": 4,
                "use_numba": False,
            },
            "exact_combined_numba": {
                "solve_strategy": "exact",
                "combine_rhs": True,
                "approx_steps": 3,
                "use_numba": True,
            },
        }
        if variant not in variant_map:
            raise ValueError(f"Unsupported LOO variant: {variant}")

        config = variant_map[variant]
        if config["use_numba"] and not _loo_numba.NUMBA_AVAILABLE:
            raise RuntimeError("Requested numba LOO variant but numba is unavailable")

        return self._impact_of_match_on_entity_variant(
            match_id,
            entity_id,
            include_teleport=include_teleport,
            solve_strategy=config["solve_strategy"],
            combine_rhs=config["combine_rhs"],
            approx_steps=config["approx_steps"],
            use_numba=config["use_numba"],
            tol=tol,
            max_iter=max_iter,
        )

    def analyze_entity_matches(
        self,
        entity_id: int,
        limit: Optional[int] = None,
        include_teleport: bool = True,
        use_flux_ranking: bool = True,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> pl.DataFrame:
        """
        Analyze impact of matches involving an entity.

        Args:
            entity_id: Entity to analyze
            limit: Maximum number of matches to analyze
            include_teleport: Whether to account for teleport changes
            use_flux_ranking: Whether to prioritize matches by pre-computed flux
            parallel: Whether to analyze matches in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            DataFrame with match impacts sorted by absolute score change
        """
        return self.analyze_entity_matches_variant(
            entity_id,
            variant="exact_combined",
            limit=limit,
            include_teleport=include_teleport,
            use_flux_ranking=use_flux_ranking,
            parallel=parallel,
            max_workers=max_workers,
        )

    def analyze_entity_matches_variant(
        self,
        entity_id: int,
        *,
        variant: str,
        limit: Optional[int] = None,
        include_teleport: bool = True,
        use_flux_ranking: bool = True,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> pl.DataFrame:
        """Internal variant hook for benchmark and profiling experiments."""
        entity_matches = list(self._entity_match_index.get(entity_id, ()))
        if not entity_matches:
            return pl.DataFrame()

        if variant == "exact_combined_numba" and not _loo_numba.NUMBA_AVAILABLE:
            raise RuntimeError("Requested numba LOO variant but numba is unavailable")

        use_numba = variant == "exact_combined_numba"
        use_flux_prefilter = (
            use_flux_ranking and limit is not None and len(entity_matches) > limit
        )
        if use_flux_prefilter:
            ranked_matches = [
                (
                    ref,
                    self._estimate_match_flux(
                        ref.match_id, entity_id, use_numba=use_numba
                    ),
                )
                for ref in entity_matches
            ]
            ranked_matches.sort(key=lambda item: abs(item[1]), reverse=True)
            entity_matches = [ref for ref, _flux in ranked_matches[:limit]]
        elif limit:
            entity_matches = entity_matches[:limit]

        use_parallel = parallel and max_workers > 1 and len(entity_matches) > 4

        # Analyze each match
        results = []

        if use_parallel:
            # Parallel execution
            def _impact_single(match_ref: EntityMatchRef):
                impact = self.impact_of_match_on_entity_variant(
                    match_ref.match_id,
                    entity_id,
                    variant=variant,
                    include_teleport=include_teleport,
                )
                return match_ref, impact

            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(entity_matches))
            ) as ex:
                futures = [ex.submit(_impact_single, ref) for ref in entity_matches]
                for i, fut in enumerate(as_completed(futures)):
                    match_ref, impact = fut.result()
                    logger.debug(
                        "Completed match %s/%s: %s",
                        i + 1,
                        len(entity_matches),
                        match_ref.match_id,
                    )

                    row = self._format_impact_result(match_ref, impact)
                    if row is not None:
                        results.append(row)
        else:
            # Sequential execution
            for i, match_ref in enumerate(entity_matches):
                logger.debug(
                    "Analyzing match %s/%s: %s",
                    i + 1,
                    len(entity_matches),
                    match_ref.match_id,
                )

                impact = self.impact_of_match_on_entity_variant(
                    match_ref.match_id,
                    entity_id,
                    variant=variant,
                    include_teleport=include_teleport,
                )

                row = self._format_impact_result(match_ref, impact)
                if row is not None:
                    results.append(row)

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results).sort("abs_delta", descending=True)
