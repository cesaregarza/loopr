"""Classic row-stochastic PageRank backend for tick-tock orchestration."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from loopr.core.constants import (
    LOSER_USER_ID,
    NORMALIZED_WEIGHT,
    WINNER_USER_ID,
)
from loopr.core import (
    Clock,
    PageRankConfig,
    TickTockConfig,
    VolumeInverseTeleport,
    WinsProportional,
    compute_denominators,
    edges_to_triplets,
    normalize_edges,
    pagerank_sparse,
)
from loopr.core.logging import get_logger
from loopr.core.preparation import group_team_members, prepare_row_edge_inputs
from loopr.schema import prepare_rank_inputs


class RowPRBackend:
    """
    Classic row-stochastic PageRank rating backend.

    Implements the original tick-tock rating approach using
    row-stochastic PageRank with teleport-proportional smoothing.
    """

    def __init__(
        self,
        config: TickTockConfig | None = None,
        *,
        decay_rate: float | None = None,
        beta: float | None = None,
        alpha: float | None = None,
        teleport_mode: str | None = None,
        smoothing_gamma: float | None = None,
        smoothing_cap_ratio: float | None = None,
        pagerank_tol: float | None = None,
        pagerank_max_iter: int | None = None,
    ):
        """
        Initialize the row PageRank backend.

        Args:
            config: Configuration object. Keyword args override config values.
            decay_rate: Time decay rate
            beta: Tournament influence exponent
            alpha: PageRank damping factor
            teleport_mode: Teleport vector mode
            smoothing_gamma: Wins-proportional smoothing parameter
            smoothing_cap_ratio: Cap ratio for smoothing
            pagerank_tol: PageRank convergence tolerance
            pagerank_max_iter: Maximum PageRank iterations
        """
        cfg = config or TickTockConfig()
        self.decay_rate = decay_rate if decay_rate is not None else cfg.decay.decay_rate
        self.beta = beta if beta is not None else cfg.engine.beta
        self.alpha = alpha if alpha is not None else cfg.pagerank.alpha
        self.teleport_mode = teleport_mode if teleport_mode is not None else cfg.teleport_mode
        self.smoothing_gamma = smoothing_gamma if smoothing_gamma is not None else cfg.engine.gamma
        self.smoothing_cap_ratio = smoothing_cap_ratio if smoothing_cap_ratio is not None else cfg.engine.cap_ratio
        self.pagerank_tol = pagerank_tol if pagerank_tol is not None else cfg.pagerank.tol
        self.pagerank_max_iter = pagerank_max_iter if pagerank_max_iter is not None else cfg.pagerank.max_iter

        self.logger = get_logger(self.__class__.__name__)
        self.clock = Clock()

        # Initialize teleport strategy
        if self.teleport_mode == "volume_inverse":
            self.teleport_strategy = VolumeInverseTeleport()
        else:
            from loopr.core import UniformTeleport

            self.teleport_strategy = UniformTeleport()

        # Initialize smoothing strategy
        self.smoothing_strategy = WinsProportional(
            gamma=self.smoothing_gamma,
            cap_ratio=self.smoothing_cap_ratio,
        )

    def compute(
        self,
        matches: pl.DataFrame,
        players: pl.DataFrame | None,
        active_ids: list[int],
        tournament_influence: dict[int, float],
        **kwargs,
    ) -> pl.DataFrame:
        """
        Compute classic PageRank ratings.

        Args:
            matches: Match data
            players: Player/roster data
            active_ids: Active player IDs (not used)
            tournament_influence: Tournament influence scores (S)

        Returns:
            DataFrame with id, score, quality_mass
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
        roster_source = group_team_members(players)
        prepared = prepare_row_edge_inputs(
            matches,
            players,
            tournament_influence,
            self.clock.now,
            self.decay_rate,
            self.beta,
            rosters=roster_source,
        )
        edges = prepared.edges

        if edges.is_empty():
            self.logger.warning("No edges built")
            return pl.DataFrame()

        node_ids = prepared.node_ids
        node_to_idx = prepared.node_to_idx
        n = len(node_ids)

        # Compute denominators with smoothing
        denominators = compute_denominators(
            edges,
            self.smoothing_strategy,
            loser_column=LOSER_USER_ID,
            winner_column=WINNER_USER_ID,
        )

        # Normalize edges
        edges = normalize_edges(
            edges,
            denominators,
            loser_column=LOSER_USER_ID,
        )

        # Convert to triplets
        rows, cols, weights = edges_to_triplets(
            edges,
            node_to_idx,
            source_column=LOSER_USER_ID,
            target_column=WINNER_USER_ID,
            weight_column=NORMALIZED_WEIGHT,
        )

        # Compute teleport vector
        teleport = self.teleport_strategy(
            node_ids,
            edges,
            LOSER_USER_ID,
        )

        # Run PageRank
        pr_config = PageRankConfig(
            alpha=self.alpha,
            tol=self.pagerank_tol,
            max_iter=self.pagerank_max_iter,
            orientation="row",
            redistribute_dangling=True,
        )

        pagerank = pagerank_sparse(rows, cols, weights, n, teleport, pr_config)

        # For classic PageRank, quality_mass = pagerank (normalized)
        # This maintains compatibility with existing influence aggregation
        quality_mass = pagerank / pagerank.sum()

        # Create result DataFrame
        result = pl.DataFrame(
            {
                "id": node_ids,
                "score": pagerank,
                "quality_mass": quality_mass,
                "pagerank": pagerank,
            }
        )

        return result
