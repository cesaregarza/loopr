"""Result dataclasses for ranking algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from typing import Any


@dataclass
class RankResult:
    """Results from a ranking algorithm run."""

    scores: np.ndarray
    ids: list[Any]

    win_pagerank: np.ndarray | None = None
    loss_pagerank: np.ndarray | None = None

    teleport: np.ndarray | None = None
    exposure: np.ndarray | None = None

    lambda_used: float | None = None
    iterations: int | None = None
    converged: bool = True

    convergence_history: list[float] | None = None
    computation_time: float | None = None
    stage_timings: dict[str, float] | None = None

    def to_dataframe(
        self,
        id_column: str = "player_id",
        score_column: str = "score",
    ) -> pl.DataFrame:
        """Convert results to a Polars DataFrame.

        Args:
            id_column: Name for ID column. Defaults to "player_id".
            score_column: Name for score column. Defaults to "score".

        Returns:
            DataFrame with results.
        """
        dataframe = pl.DataFrame(
            {id_column: self.ids, score_column: self.scores.tolist()}
        )

        if self.win_pagerank is not None:
            dataframe = dataframe.with_columns(
                pl.Series("win_pr", self.win_pagerank)
            )

        if self.loss_pagerank is not None:
            dataframe = dataframe.with_columns(
                pl.Series("loss_pr", self.loss_pagerank)
            )

        if self.exposure is not None:
            dataframe = dataframe.with_columns(
                pl.Series("exposure", self.exposure)
            )

        if self.teleport is not None:
            dataframe = dataframe.with_columns(
                pl.Series("teleport", self.teleport)
            )

        return dataframe

    def get_top_n(self, count: int = 10) -> pl.DataFrame:
        """Get top N ranked entities.

        Args:
            count: Number of top entities to return. Defaults to 10.

        Returns:
            DataFrame with top N entities sorted by score.
        """
        dataframe = self.to_dataframe()
        return dataframe.sort("score", descending=True).head(count)


@dataclass
class TickTockResult(RankResult):
    """Results specific to Tick-Tock algorithm."""

    tournament_influence: dict[Any, float] | None = None
    retrospective_strength: np.ndarray | None = None
    tick_history: list[np.ndarray] | None = None
    tock_history: list[np.ndarray] | None = None
    denominators: np.ndarray | None = None


@dataclass
class ExposureLogOddsResult(RankResult):
    """Results specific to Exposure Log-Odds algorithm."""

    surprisal_weights: np.ndarray | None = None
    active_mask: np.ndarray | None = None
    raw_scores: np.ndarray | None = None
    decay_factors: np.ndarray | None = None

