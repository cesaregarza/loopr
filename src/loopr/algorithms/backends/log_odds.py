"""Log-odds rating computation backend for tick-tock orchestration."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from loopr.algorithms._log_odds_common import (
    reporting_exposure,
    resolve_lambda,
    teleport_from_share,
)
from loopr.core import (
    Clock,
    ExposureLogOddsConfig,
    PageRankConfig,
    build_exposure_triplets,
    pagerank_from_adjacency,
)
from loopr.core.logging import get_logger
from loopr.core.preparation import prepare_exposure_graph
from loopr.schema import prepare_rank_inputs


class LogOddsBackend:
    """Log-odds rating backend.

    Implements log-odds ratings using PageRank computations with
    exposure-based teleport vectors and lambda smoothing.
    """

    def __init__(
        self,
        config: ExposureLogOddsConfig | None = None,
        *,
        decay_rate: float | None = None,
        beta: float | None = None,
        alpha: float | None = None,
        lambda_mode: str | None = None,
        fixed_lambda: float | None = None,
        pagerank_tol: float | None = None,
        pagerank_max_iter: int | None = None,
        epsilon: float = 1e-9,
    ) -> None:
        """Initialize the log-odds backend.

        Args:
            config: Configuration object. Keyword args override config values.
            decay_rate: Time decay rate for match weights.
            beta: Tournament influence exponent.
            alpha: PageRank damping factor.
            lambda_mode: Lambda computation mode ('auto' or 'fixed').
            fixed_lambda: Fixed lambda value when mode is 'fixed'.
            pagerank_tol: PageRank convergence tolerance.
            pagerank_max_iter: Maximum PageRank iterations.
            epsilon: Small value for numerical stability.
        """
        cfg = config or ExposureLogOddsConfig()
        self.decay_rate = decay_rate if decay_rate is not None else cfg.decay.decay_rate
        self.beta = beta if beta is not None else cfg.engine.beta
        self.alpha = alpha if alpha is not None else cfg.pagerank.alpha
        self.lambda_mode = lambda_mode if lambda_mode is not None else cfg.lambda_mode
        self.fixed_lambda = fixed_lambda if fixed_lambda is not None else cfg.fixed_lambda
        self.pagerank_tol = pagerank_tol if pagerank_tol is not None else cfg.pagerank.tol
        self.pagerank_max_iter = pagerank_max_iter if pagerank_max_iter is not None else cfg.pagerank.max_iter
        self.epsilon = epsilon

        self.logger = get_logger(self.__class__.__name__)
        self.clock = Clock()
        self._last_rho = None

    def compute(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        active_ids: list[int],
        tournament_influence: dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        """Compute log-odds ratings for players.

        Args:
            matches: Match data.
            players: Player/roster data.
            active_ids: Active player IDs (not used in this backend).
            tournament_influence: Tournament influence scores.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame with player ratings and metrics.
        """
        if players is None:
            self.logger.warning(
                "No player data provided, cannot compute ratings"
            )
            return pl.DataFrame()

        inputs = prepare_rank_inputs(matches, players)
        return self.compute_normalized(
            inputs.matches,
            inputs.participants,
            active_ids,
            tournament_influence,
            **kwargs,
        )

    def compute_normalized(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame,
        active_ids: list[int],
        tournament_influence: dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        """Compute ratings from already-normalized match and player frames."""
        graph_inputs = prepare_exposure_graph(
            matches,
            players,
            tournament_influence or {},
            self.clock.now,
            self.decay_rate,
            self.beta,
            active_entities=active_ids or None,
        )
        matches_df = graph_inputs.matches
        if matches_df.is_empty():
            self.logger.warning("No valid matches after conversion")
            return pl.DataFrame()

        entity_metrics = graph_inputs.entity_metrics
        node_ids = graph_inputs.node_ids
        node_to_idx = graph_inputs.node_to_idx
        num_nodes = len(node_ids)
        index_mapping = graph_inputs.index_mapping

        teleport_vector = teleport_from_share(
            matches_df,
            node_to_idx,
            aggregated_metrics=entity_metrics,
            index_mapping=index_mapping,
            epsilon=self.epsilon,
        )
        self._last_rho = teleport_vector

        rows, cols, data = build_exposure_triplets(graph_inputs)
        adjacency_win = sp.csr_matrix(
            (data, (rows, cols)), shape=(num_nodes, num_nodes)
        )

        pagerank_config = PageRankConfig(
            alpha=self.alpha,
            tol=self.pagerank_tol,
            max_iter=self.pagerank_max_iter,
            orientation="col",
            redistribute_dangling=True,
        )

        win_pagerank = pagerank_from_adjacency(
            adjacency_win, teleport_vector, pagerank_config
        )
        loss_pagerank = pagerank_from_adjacency(
            adjacency_win.transpose().tocsr(),
            teleport_vector,
            pagerank_config,
        )

        lambda_smooth = resolve_lambda(
            win_pagerank,
            teleport_vector,
            lambda_mode=self.lambda_mode,
            fixed_lambda=self.fixed_lambda,
            fallback=1e-4,
        )

        smoothed_win_pagerank = win_pagerank + lambda_smooth * teleport_vector
        smoothed_loss_pagerank = loss_pagerank + lambda_smooth * teleport_vector
        scores = np.log(smoothed_win_pagerank / smoothed_loss_pagerank)
        quality_mass = smoothed_win_pagerank / (
            smoothed_win_pagerank + smoothed_loss_pagerank
        )

        exposure = reporting_exposure(
            matches_df,
            node_to_idx,
            aggregated_metrics=entity_metrics,
            index_mapping=index_mapping,
        )

        return pl.DataFrame(
            {
                "id": node_ids,
                "score": scores,
                "quality_mass": quality_mass,
                "win_pr": win_pagerank,
                "loss_pr": loss_pagerank,
                "exposure": exposure,
                "lambda_used": [lambda_smooth] * num_nodes,
            }
        )
