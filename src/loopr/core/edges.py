"""Edge building utilities for ranking algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from loopr.core.preparation import (
    PreparedGraphInputs,
    build_team_edge_dataframe,
    prepare_row_edge_inputs,
    prepare_weighted_matches,
)
from loopr.schema import prepare_matches_frame, prepare_rank_inputs

if TYPE_CHECKING:
    from typing import Any


def build_player_edges(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Build entity-level loser->winner edges with tournament weighting."""
    prepared = prepare_rank_inputs(matches, players)
    return _build_player_edges_normalized(
        prepared.matches,
        prepared.participants,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        timestamp_column=timestamp_column,
    )


def _build_player_edges_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Internal player-edge construction for already-normalized inputs."""
    prepared = prepare_row_edge_inputs(
        matches,
        players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        timestamp_column=timestamp_column,
    )
    return prepared.edges


def build_team_edges(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> pl.DataFrame:
    """Build team-level loser->winner edges from matches."""
    matches = prepare_matches_frame(matches)
    return _build_team_edges_normalized(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )


def _build_team_edges_normalized(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> pl.DataFrame:
    """Internal team-edge construction for already-normalized match inputs."""
    weighted = prepare_weighted_matches(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )
    return build_team_edge_dataframe(weighted)


def build_exposure_triplets(
    matches_dataframe: PreparedGraphInputs | pl.DataFrame,
    node_to_index: dict[Any, int] | None = None,
    *,
    index_mapping: pl.DataFrame | None = None,
    pair_edges: pl.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce COO triplets (row=winner_idx, col=loser_idx, data=share)."""
    if isinstance(matches_dataframe, PreparedGraphInputs):
        graph_inputs = matches_dataframe
        matches_df = graph_inputs.matches
        if node_to_index is None:
            node_to_index = graph_inputs.node_to_idx
        if index_mapping is None:
            index_mapping = graph_inputs.index_mapping
        if pair_edges is None:
            pair_edges = graph_inputs.pair_edges
    else:
        matches_df = matches_dataframe

    if node_to_index is None:
        if index_mapping is None:
            raise ValueError("node_to_index or index_mapping is required")
        if index_mapping.is_empty():
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )
        node_to_index = dict(
            zip(index_mapping["id"].to_list(), index_mapping["idx"].to_list())
        )
    else:
        node_to_index = {
            entity_id: idx
            for entity_id, idx in node_to_index.items()
            if entity_id is not None
        }
        if not node_to_index:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

    if pair_edges is None:
        if matches_df.is_empty():
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )
        pair_edges = (
            matches_df.select(["winners", "losers", "share"])
            .explode("winners")
            .drop_nulls("winners")
            .explode("losers")
            .drop_nulls("losers")
            .group_by(["winners", "losers"])
            .agg(pl.col("share").sum().alias("share"))
            .rename({"winners": "winner_id", "losers": "loser_id"})
        )
    elif pair_edges.is_empty():
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    winner_ids = pair_edges["winner_id"].to_list()
    loser_ids = pair_edges["loser_id"].to_list()
    shares = pair_edges["share"].to_numpy()

    count = 0
    n = len(winner_ids)
    row_buf = np.empty(n, dtype=np.int64)
    col_buf = np.empty(n, dtype=np.int64)
    wt_buf = np.empty(n, dtype=np.float64)
    _get = node_to_index.get
    for i in range(n):
        winner_idx = _get(winner_ids[i])
        loser_idx = _get(loser_ids[i])
        if winner_idx is not None and loser_idx is not None:
            row_buf[count] = winner_idx
            col_buf[count] = loser_idx
            wt_buf[count] = shares[i]
            count += 1

    if count == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    return row_buf[:count], col_buf[:count], wt_buf[:count]


def compute_denominators(
    edges: pl.DataFrame,
    smoothing_strategy,
    loser_column: str = "loser_user_id",
    winner_column: str = "winner_user_id",
    weight_column: str = "weight_sum",
) -> pl.DataFrame:
    """Compute denominators with smoothing for edge normalization."""
    loss_sums: dict = {}
    win_sums: dict = {}
    loser_ids = edges[loser_column].to_list()
    winner_ids = edges[winner_column].to_list()
    weights = edges[weight_column].to_numpy()

    for i in range(len(loser_ids)):
        weight = float(weights[i])
        loser_id = loser_ids[i]
        winner_id = winner_ids[i]
        loss_sums[loser_id] = loss_sums.get(loser_id, 0.0) + weight
        win_sums[winner_id] = win_sums.get(winner_id, 0.0) + weight

    node_ids = sorted(loss_sums.keys())
    loss_arr = np.array([loss_sums[node_id] for node_id in node_ids], dtype=float)
    win_arr = np.array([win_sums.get(node_id, 0.0) for node_id in node_ids], dtype=float)

    if smoothing_strategy:
        denom_arr = smoothing_strategy.denom(loss_arr, win_arr)
    else:
        denom_arr = loss_arr.copy()

    lambda_arr = denom_arr - loss_arr

    return pl.DataFrame(
        {
            loser_column: node_ids,
            "loss_weights": loss_arr,
            "win_weights": win_arr,
            "denom": denom_arr,
            "lambda": lambda_arr,
        }
    )


def normalize_edges(
    edges: pl.DataFrame,
    denominators: pl.DataFrame,
    loser_column: str = "loser_user_id",
    weight_column: str = "weight_sum",
) -> pl.DataFrame:
    """Normalize edge weights by denominators."""
    denom_dict = dict(
        zip(
            denominators[loser_column].to_list(),
            denominators["denom"].to_list(),
        )
    )

    loser_ids = edges[loser_column].to_list()
    weights = edges[weight_column].to_numpy()
    normalized = np.empty(len(weights), dtype=float)
    for i, loser_id in enumerate(loser_ids):
        denom = denom_dict.get(loser_id, 1.0)
        normalized[i] = weights[i] / denom if denom != 0.0 else 0.0

    return edges.with_columns(
        pl.Series("normalized_weight", normalized),
        pl.Series("denom", [denom_dict.get(loser_id, 1.0) for loser_id in loser_ids]),
    )


def edges_to_triplets(
    edges: pl.DataFrame,
    node_to_index: dict[Any, int],
    source_column: str,
    target_column: str,
    weight_column: str = "weight_sum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert edge DataFrame to COO triplets."""
    source_ids = edges[source_column].to_list()
    target_ids = edges[target_column].to_list()
    weights = edges[weight_column].to_numpy()

    n = len(source_ids)
    row_buf = np.empty(n, dtype=np.int64)
    col_buf = np.empty(n, dtype=np.int64)
    wt_buf = np.empty(n, dtype=np.float64)
    count = 0
    _get = node_to_index.get
    for i in range(n):
        source_idx = _get(source_ids[i])
        target_idx = _get(target_ids[i])
        if source_idx is not None and target_idx is not None:
            row_buf[count] = source_idx
            col_buf[count] = target_idx
            wt_buf[count] = weights[i]
            count += 1

    return row_buf[:count], col_buf[:count], wt_buf[:count]
