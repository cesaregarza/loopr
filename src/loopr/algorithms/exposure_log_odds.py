from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from loopr.algorithms._log_odds_common import (
    last_activity_from_metrics,
    reporting_exposure,
    resolve_lambda,
    teleport_from_share,
)
from loopr.core import (
    Clock,
    ExposureLogOddsConfig,
    ExposureLogOddsResult,
    PageRankConfig,
    apply_inactivity_decay,
    build_exposure_triplets,
    pagerank_from_adjacency,
)
from loopr.core.logging import get_logger
from loopr.core.preparation import group_team_members, prepare_exposure_graph
from loopr.schema import prepare_rank_inputs

if TYPE_CHECKING:
    from loopr.analysis.loo_analyzer import LOOAnalyzer


class ExposureLogOddsEngine:
    """Exposure Log-Odds rating engine using modular components.

    Semantics:
      - Teleport ρ is exposure-based (sum of match pair 'share' per entity)
      - Two PageRanks with the SAME ρ (col-stochastic)
      - Lambda auto-tuned to ~2.5% of median PR per node (dividing by median ρ)
      - Reporting 'exposure' uses the summed match-pair weight
      - Optional time-decay after inactivity delay
      - Public output schema uses neutral entity identifiers
    """

    def __init__(
        self,
        config: ExposureLogOddsConfig | None = None,
        *,
        now_ts: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or ExposureLogOddsConfig()
        self.logger = get_logger(self.__class__.__name__)

        if clock is not None:
            self.clock = clock
        else:
            self.clock = Clock(
                now_timestamp=(now_ts if now_ts is not None else time.time())
            )

        self.last_result: ExposureLogOddsResult | None = None
        self._last_rho: np.ndarray | None = None
        self._loo_analyzer: LOOAnalyzer | None = None
        self._loo_matches_df: pl.DataFrame | None = None
        self._loo_players_df: pl.DataFrame | None = None
        self._converted_matches_df: pl.DataFrame | None = None
        self.last_stage_timings: dict[str, float] | None = None

    def _rank_internal(
        self,
        matches: pl.DataFrame,
        participants: pl.DataFrame,
        tournament_influence: dict[int, float] | None = None,
        *,
        appearances: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Rank prepared internal inputs using the Exposure Log-Odds algorithm.

        Args:
            matches: DataFrame containing match data.
            participants: DataFrame containing prepared participant data.
            tournament_influence: Optional mapping of tournament IDs to influence weights.

        Returns:
            DataFrame containing entity rankings.
        """
        start_time = time.perf_counter()
        stage_timings: dict[str, float] = {}
        participants_df = participants
        stage_timings["input_normalization"] = 0.0

        # 1) Get active entities and tournament influences via tick-tock.
        #    Union them with entities who actually appear in matches so subs are kept.
        stage_start = time.perf_counter()
        if self.config.use_tick_tock_active:
            self.logger.info(
                "Running tick-tock to obtain active entities & tournament influences..."
            )
            tick = self._run_tick_tock_for_active_entities(
                matches, participants_df
            )
            active_entities = tick["active_entities"]
            tournament_influence = tick["tournament_influence"]
        else:
            self.logger.info("Skipping tick-tock (use_tick_tock_active=False)")
            active_entities = participants_df["user_id"].to_list()
            tournament_influence = tournament_influence or {}
        stage_timings["active_resolution"] = time.perf_counter() - stage_start

        # 2) Convert matches into compact per-match lists with roster expansion and weights
        self.logger.info("Converting matches...")
        stage_start = time.perf_counter()
        roster_source = group_team_members(participants_df)
        graph_inputs = prepare_exposure_graph(
            matches,
            participants_df,
            tournament_influence or {},
            self.clock.now,
            self.config.decay.decay_rate,
            self.config.engine.beta,
            rosters=roster_source,
            appearances=appearances,
            active_entities=active_entities,
        )
        mdf = graph_inputs.matches
        self._converted_matches_df = mdf
        stage_timings["match_preparation"] = time.perf_counter() - stage_start
        if mdf.is_empty():
            self.logger.warning(
                "No valid matches after conversion; returning empty result."
            )
            stage_timings["total"] = time.perf_counter() - start_time
            self.last_stage_timings = stage_timings
            return pl.DataFrame()

        entity_metrics = graph_inputs.entity_metrics

        # 3) Use the shared graph-prep artifacts for node setup.
        stage_start = time.perf_counter()
        node_ids = graph_inputs.node_ids
        if not node_ids:
            self.logger.warning(
                "No active or appeared entities after union; returning empty result."
            )
            stage_timings["node_setup"] = time.perf_counter() - stage_start
            stage_timings["total"] = time.perf_counter() - start_time
            self.last_stage_timings = stage_timings
            return pl.DataFrame()
        node_to_idx = graph_inputs.node_to_idx
        num_nodes = len(node_ids)
        index_mapping = graph_inputs.index_mapping

        # 4) Build teleport ρ from exposure mass
        rho = teleport_from_share(
            mdf,
            node_to_idx,
            aggregated_metrics=entity_metrics,
            index_mapping=index_mapping,
        )
        self._last_rho = rho
        stage_timings["node_setup"] = time.perf_counter() - stage_start

        # 5) Triplets for A_win (loser -> winner), then mirror for A_loss
        stage_start = time.perf_counter()
        rows, cols, data = build_exposure_triplets(graph_inputs)
        adjacency_win = sp.csr_matrix(
            (data, (rows, cols)), shape=(num_nodes, num_nodes)
        )
        stage_timings["graph_build"] = time.perf_counter() - stage_start

        pr_cfg = PageRankConfig(
            alpha=self.config.pagerank.alpha,
            tol=self.config.pagerank.tol,
            max_iter=self.config.pagerank.max_iter,
            orientation="col",
            redistribute_dangling=True,
        )

        # 6) Win & loss PageRanks with SAME ρ
        stage_start = time.perf_counter()
        self.logger.info("Computing win PageRank...")
        win_pagerank = pagerank_from_adjacency(adjacency_win, rho, pr_cfg)
        self.logger.info("Computing loss PageRank...")
        loss_pagerank = pagerank_from_adjacency(
            adjacency_win.transpose().tocsr(),
            rho,
            pr_cfg,
        )
        stage_timings["pagerank"] = time.perf_counter() - stage_start

        # 7) Lambda smoothing
        stage_start = time.perf_counter()
        lambda_smooth = resolve_lambda(
            win_pagerank,
            rho,
            lambda_mode=self.config.lambda_mode,
            fixed_lambda=self.config.fixed_lambda,
            fallback=self.config.engine.lambda_smooth or 1e-4,
        )
        self.logger.info(f"Lambda used: {lambda_smooth:.6f}")

        # 8) Log-odds scores
        win_smooth = win_pagerank + lambda_smooth * rho
        loss_smooth = loss_pagerank + lambda_smooth * rho
        scores = (
            np.log(win_smooth / loss_smooth)
            if self.config.apply_log_transform
            else (win_smooth / loss_smooth)
        )

        # 9) Inactivity decay
        if self.config.engine.score_decay_rate > 0:
            last_ts = self._last_activity_times(
                entity_metrics,
                index_mapping,
                num_nodes,
            )
            scores = apply_inactivity_decay(
                scores,
                last_ts,
                self.clock.now,
                delay_days=self.config.engine.score_decay_delay_days,
                decay_rate=self.config.engine.score_decay_rate,
            )

        # 10) Calculate reporting exposure
        exposure = reporting_exposure(
            mdf,
            node_to_idx,
            aggregated_metrics=entity_metrics,
            index_mapping=index_mapping,
        )
        stage_timings["score_postprocess"] = time.perf_counter() - stage_start

        # 11) Apply minimum exposure filter if configured
        mask = np.ones(num_nodes, dtype=bool)
        if self.config.engine.min_exposure is not None:
            mask = exposure >= float(self.config.engine.min_exposure)

        # 12) Build result dataframe
        stage_start = time.perf_counter()
        result = (
            pl.DataFrame(
                {
                    "id": node_ids,
                    "score": scores,
                    "win_pr": win_pagerank,
                    "loss_pr": loss_pagerank,
                    "exposure": exposure,
                }
            )
            .filter(pl.Series("active", mask))
            .sort("score", descending=True)
            .with_row_index("player_rank", offset=1)
            .with_columns(pl.col("player_rank").cast(pl.Int64))
        )
        stage_timings["result_assembly"] = time.perf_counter() - stage_start

        # 13) Store diagnostics for downstream analysis
        elapsed = time.perf_counter() - start_time
        stage_timings["total"] = elapsed
        self.last_stage_timings = stage_timings
        self.last_result = ExposureLogOddsResult(
            scores=scores,
            ids=node_ids,
            win_pagerank=win_pagerank,
            loss_pagerank=loss_pagerank,
            teleport=rho,
            exposure=exposure,
            lambda_used=lambda_smooth,
            active_mask=mask,
            raw_scores=scores.copy(),
            computation_time=elapsed,
            stage_timings=stage_timings.copy(),
        )

        # Store tournament influence for analysis tools
        self.tournament_influence = tournament_influence

        self.logger.info(
            "Exposure Log-Odds ranking completed in %.2fs", elapsed
        )
        return result

    def rank_entities(
        self,
        matches: pl.DataFrame,
        participants: pl.DataFrame,
        tournament_influence: dict[int, float] | None = None,
        *,
        appearances: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Validate neutral inputs and return `entity_id` rankings."""
        inputs = prepare_rank_inputs(matches, participants, appearances)
        result = self._rank_internal(
            inputs.matches,
            inputs.participants,
            tournament_influence,
            appearances=inputs.appearances,
        )
        if result.is_empty():
            return result
        return result.rename({"id": "entity_id"})

    def prepare_loo_analyzer(
        self,
        matches_df: pl.DataFrame | None = None,
        players_df: pl.DataFrame | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Prepare LOO analyzer with pre-factorized solvers for fast repeated analysis.

        This should be called after rank_entities() to set up the LOO infrastructure.
        The analyzer is cached and reused unless force_rebuild is True.

        Args:
            matches_df: Optional matches DataFrame (uses internal converted if None)
            players_df: Optional players DataFrame (uses last ranking data if None)
            force_rebuild: Force rebuild even if analyzer exists
        """
        if self.last_result is None:
            raise ValueError(
                "Must call rank_entities() before preparing LOO analyzer"
            )

        # Use internal converted matches if not provided
        if matches_df is None:
            if self._converted_matches_df is None:
                raise ValueError(
                    "No converted matches available. Call rank_entities() first."
                )
            matches_df = self._converted_matches_df
            self.logger.info("Using internally converted matches dataframe")

        # Use last result's entity data if not provided
        if players_df is None:
            players_df = pl.DataFrame(
                {
                    "entity_id": self.last_result.ids,
                    "name": [f"Entity_{entity_id}" for entity_id in self.last_result.ids],
                }
            )
            self.logger.info("Using entity IDs from last ranking result")

        # Check if we need to rebuild
        if not force_rebuild and self._loo_analyzer is not None:
            # Check if data has changed
            if (
                self._loo_matches_df is not None
                and matches_df.equals(self._loo_matches_df)
                and self._loo_players_df is not None
                and players_df.equals(self._loo_players_df)
            ):
                self.logger.info(
                    "LOO analyzer already prepared, reusing existing infrastructure"
                )
                return

        self.logger.info(
            "Preparing LOO analyzer with pre-factorized solvers..."
        )
        from loopr.analysis.loo_analyzer import LOOAnalyzer

        start_time = time.time()
        self._loo_analyzer = LOOAnalyzer(self, matches_df, players_df)
        self._loo_matches_df = matches_df
        self._loo_players_df = players_df

        prep_time = time.time() - start_time
        self.logger.info(f"LOO analyzer prepared in {prep_time:.2f}s")
        self.logger.info(f"  - {len(self._loo_analyzer.node_to_idx)} nodes")
        self.logger.info(f"  - {self._loo_analyzer.A_win.nnz} win edges")
        self.logger.info(f"  - {self._loo_analyzer.A_loss.nnz} loss edges")
        self.logger.info(
            f"  - {len(self._loo_analyzer._match_cache)} matches cached"
        )

    def analyze_match_impact(
        self, match_id: int, entity_id: int, include_teleport: bool = True
    ) -> dict[str, Any]:
        """
        Analyze the impact of removing a match on an entity's score.

        Automatically prepares LOO analyzer if needed.
        Uses pre-factorized solvers for extremely fast analysis.

        Args:
            match_id: Match to analyze
            entity_id: Entity to check impact for
            include_teleport: Whether to include teleport vector changes

        Returns:
            Dictionary with impact analysis
        """
        # Auto-prepare LOO analyzer if not ready
        if self._loo_analyzer is None:
            self.logger.info("LOO analyzer not prepared, initializing now...")
            self.prepare_loo_analyzer()

        return self._loo_analyzer.impact_of_match_on_entity(
            match_id, entity_id, include_teleport
        )

    def analyze_entity_matches(
        self,
        entity_id: int,
        limit: int | None = None,
        include_teleport: bool = True,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> pl.DataFrame:
        """
        Analyze impact of all matches involving an entity.

        Automatically prepares LOO analyzer if needed.
        Uses pre-factorized solvers and parallel processing for speed.

        Args:
            entity_id: Entity to analyze
            limit: Maximum number of matches to analyze
            include_teleport: Whether to include teleport changes
            parallel: Use parallel processing
            max_workers: Number of parallel workers

        Returns:
            DataFrame with match impacts sorted by score change
        """
        # Auto-prepare LOO analyzer if not ready
        if self._loo_analyzer is None:
            self.logger.info("LOO analyzer not prepared, initializing now...")
            self.prepare_loo_analyzer()

        return self._loo_analyzer.analyze_entity_matches(
            entity_id,
            limit,
            include_teleport,
            use_flux_ranking=True,
            parallel=parallel,
            max_workers=max_workers,
        )

    def get_loo_analyzer(self):
        """
        Get the underlying LOO analyzer for advanced usage.

        Returns:
            LOOAnalyzer instance or None if not prepared
        """
        return self._loo_analyzer

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _run_tick_tock_for_active_entities(
        self,
        matches: pl.DataFrame,
        participants: pl.DataFrame,
    ) -> dict[str, Any]:
        from loopr.algorithms.tick_tock import TickTockEngine

        engine = TickTockEngine(self.config.tick_tock)
        # Pass deterministic time if we were given one
        engine.clock = self.clock

        tt_df = engine._rank_internal(matches, participants)
        active_entities = tt_df["player_id"].to_list()
        tournament_influence_data = (
            getattr(engine, "tournament_influence", {}) or {}
        )
        return {
            "active_entities": active_entities,
            "tournament_influence": tournament_influence_data,
        }

    def _last_activity_times(
        self,
        entity_metrics: pl.DataFrame,
        index_mapping: pl.DataFrame,
        num_nodes: int,
    ) -> np.ndarray:
        """Get the last activity timestamp for each player.

        Args:
            entity_metrics: Aggregated per-entity metrics.
            index_mapping: Materialized ID->index lookup.
            num_nodes: Total number of nodes.

        Returns:
            Array of last activity timestamps.
        """
        return last_activity_from_metrics(
            entity_metrics,
            index_mapping,
            num_nodes,
            self.clock.now,
        )
