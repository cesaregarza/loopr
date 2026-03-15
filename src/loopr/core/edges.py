"""Edge building utilities for ranking algorithms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from loopr.schema import (
    normalize_matches_schema,
    prepare_rank_inputs,
)

if TYPE_CHECKING:
    from typing import Any


def _prepare_weighted_matches(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Filter byes, resolve timestamps, join influence, compute weight column.

    Args:
        matches: Match data (already normalized).
        tournament_influence: Tournament ID to influence mapping.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.
        timestamp_column: Optional explicit timestamp column name.

    Returns:
        Filtered match DataFrame with ``tournament_strength`` and
        ``match_weight`` columns added.
    """
    if matches.is_empty():
        return matches

    filter_expression = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in matches.columns:
        filter_expression = filter_expression & ~pl.col("is_bye").fill_null(
            False
        )

    match_data = matches.filter(filter_expression)

    # Join tournament influence
    if tournament_influence:
        strength_dataframe = pl.DataFrame(
            {
                "tournament_id": list(tournament_influence.keys()),
                "tournament_strength": list(tournament_influence.values()),
            }
        )
        match_data = match_data.join(
            strength_dataframe,
            on="tournament_id",
            how="left",
            coalesce=True,
        ).with_columns(pl.col("tournament_strength").fill_null(1.0))
    else:
        match_data = match_data.with_columns(
            pl.lit(1.0).alias("tournament_strength")
        )

    # Resolve timestamp
    if timestamp_column and timestamp_column in match_data.columns:
        match_data = match_data.with_columns(
            pl.col(timestamp_column).alias("ts")
        )
    else:
        timestamp_expressions: list = []
        if "last_game_finished_at" in match_data.columns:
            timestamp_expressions.append(pl.col("last_game_finished_at"))
        if "match_created_at" in match_data.columns:
            timestamp_expressions.append(pl.col("match_created_at"))
        timestamp_expressions.append(pl.lit(now_timestamp))
        match_data = match_data.with_columns(
            pl.coalesce(timestamp_expressions).alias("ts")
        )

    # Compute match weight
    time_decay_factor = (
        ((pl.lit(now_timestamp) - pl.col("ts").cast(pl.Float64)) / 86400.0)
        .mul(-decay_rate)
        .exp()
    )

    if beta == 0.0:
        match_weight = time_decay_factor
    else:
        match_weight = time_decay_factor * (
            pl.col("tournament_strength") ** beta
        )

    match_data = match_data.with_columns(match_weight.alias("match_weight"))

    return match_data


def build_player_edges(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Build player-level edges with tournament strength weighting.

    Args:
        matches: Match data.
        players: Player/roster data.
        tournament_influence: Tournament ID to influence mapping.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.
        timestamp_column: Optional timestamp column name. Defaults to None.

    Returns:
        Edge DataFrame with columns: loser_user_id, winner_user_id, weight_sum.
    """
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
    """Internal player-edge construction for already-normalized inputs.

    Uses a dict-based roster lookup to avoid repeated Polars joins.
    """
    if matches.is_empty() or players.is_empty():
        return pl.DataFrame([])

    match_data = _prepare_weighted_matches(
        matches, tournament_influence, now_timestamp, decay_rate, beta,
        timestamp_column=timestamp_column,
    )

    # Build roster lookup: (tournament_id, team_id) → list[user_id]
    roster_lookup: dict[tuple[int, int], list] = {}
    for row in players.select(
        ["tournament_id", "team_id", "user_id"]
    ).iter_rows():
        key = (int(row[0]), int(row[1]))
        roster_lookup.setdefault(key, []).append(row[2])

    # Build loser→winner edges via dict lookup (avoids 3 Polars joins)
    edge_accum: dict[tuple, float] = {}
    for row in match_data.select(
        ["tournament_id", "winner_team_id", "loser_team_id", "match_weight"]
    ).iter_rows():
        tid, wtid, ltid, weight = int(row[0]), int(row[1]), int(row[2]), float(row[3])
        winner_players = roster_lookup.get((tid, wtid), [])
        loser_players = roster_lookup.get((tid, ltid), [])
        for lp in loser_players:
            for wp in winner_players:
                key = (lp, wp)
                edge_accum[key] = edge_accum.get(key, 0.0) + weight

    if not edge_accum:
        return pl.DataFrame([])

    losers, winners, weights = [], [], []
    for (lp, wp), w in edge_accum.items():
        losers.append(lp)
        winners.append(wp)
        weights.append(w)

    return pl.DataFrame(
        {
            "loser_user_id": losers,
            "winner_user_id": winners,
            "weight_sum": weights,
        }
    )


def build_team_edges(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> pl.DataFrame:
    """Build team-level edges from matches.

    Args:
        matches: Match data with team IDs.
        tournament_influence: Tournament ID to influence mapping.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.

    Returns:
        Edge DataFrame with columns: loser_team_id, winner_team_id, weight_sum.
    """
    matches = normalize_matches_schema(matches)
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
    if matches.is_empty():
        return pl.DataFrame([])

    match_data = _prepare_weighted_matches(
        matches, tournament_influence, now_timestamp, decay_rate, beta,
    )

    # Aggregate team-to-team edges
    edges = match_data.group_by(["loser_team_id", "winner_team_id"]).agg(
        pl.col("match_weight").sum().alias("weight_sum")
    )

    return edges


def build_exposure_triplets(
    matches_dataframe: pl.DataFrame,
    node_to_index: dict[Any, int] | None = None,
    *,
    index_mapping: pl.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce COO triplets (row=winner_idx, col=loser_idx, data=share).

    Uses list-explode to form the cartesian product winner×loser per match,
    then maps IDs to indices via dict lookup (avoids Polars join overhead).

    Args:
        matches_dataframe: DataFrame with winners, losers, share columns.
        node_to_index: Mapping from node IDs to indices.
        index_mapping: Polars DataFrame lookup (used only to extract dict if
            node_to_index is not provided).

    Returns:
        Tuple of (row_indices, col_indices, weights).
    """
    # Resolve the dict lookup
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
        node_to_index = {k: v for k, v in node_to_index.items() if k is not None}
        if not node_to_index:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

    # Explode both lists to pairs — single Polars operation
    pairs = (
        matches_dataframe.explode("winners")
        .explode("losers")
        .select(["winners", "losers", "share"])
    )

    # Extract raw arrays and map IDs via dict (avoid two Polars joins)
    winner_ids = pairs["winners"].to_list()
    loser_ids = pairs["losers"].to_list()
    shares = pairs["share"].to_numpy()

    # Vectorized dict lookup with filtering
    n = len(winner_ids)
    row_buf = np.empty(n, dtype=np.int64)
    col_buf = np.empty(n, dtype=np.int64)
    wt_buf = np.empty(n, dtype=np.float64)
    count = 0
    _get = node_to_index.get
    for i in range(n):
        w_idx = _get(winner_ids[i])
        l_idx = _get(loser_ids[i])
        if w_idx is not None and l_idx is not None:
            row_buf[count] = w_idx
            col_buf[count] = l_idx
            wt_buf[count] = shares[i]
            count += 1

    if count == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    rows = row_buf[:count]
    cols = col_buf[:count]
    wts = wt_buf[:count]

    # Aggregate duplicate (row, col) pairs using scipy sparse
    from scipy.sparse import coo_matrix

    n_nodes = max(node_to_index.values()) + 1
    coo = coo_matrix((wts, (rows, cols)), shape=(n_nodes, n_nodes))
    coo.sum_duplicates()

    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data


def compute_denominators(
    edges: pl.DataFrame,
    smoothing_strategy,
    loser_column: str = "loser_user_id",
    winner_column: str = "winner_user_id",
    weight_column: str = "weight_sum",
) -> pl.DataFrame:
    """Compute denominators with smoothing for edge normalization.

    Uses dict-based aggregation to avoid Polars group_by + join overhead.

    Args:
        edges: Edge DataFrame.
        smoothing_strategy: Smoothing strategy object with denom() method.
        loser_column: Column name for loser/source nodes.
        winner_column: Column name for winner/target nodes.
        weight_column: Column name for edge weights.

    Returns:
        DataFrame with columns: node_id, loss_weights, win_weights, denom, lambda.
    """
    # Aggregate loss and win weights via dict (avoids 2 group_bys + 1 join)
    loss_sums: dict = {}
    win_sums: dict = {}
    loser_ids = edges[loser_column].to_list()
    winner_ids = edges[winner_column].to_list()
    wts = edges[weight_column].to_numpy()

    for i in range(len(loser_ids)):
        w = float(wts[i])
        lid = loser_ids[i]
        wid = winner_ids[i]
        loss_sums[lid] = loss_sums.get(lid, 0.0) + w
        win_sums[wid] = win_sums.get(wid, 0.0) + w

    node_ids = sorted(loss_sums.keys())
    loss_arr = np.array([loss_sums[n] for n in node_ids], dtype=float)
    win_arr = np.array([win_sums.get(n, 0.0) for n in node_ids], dtype=float)

    # Apply smoothing strategy
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
    """Normalize edge weights by denominators.

    Uses dict lookup instead of Polars join for speed.

    Args:
        edges: Edge DataFrame.
        denominators: Denominator DataFrame.
        loser_column: Column name for source nodes.
        weight_column: Column name for weights.

    Returns:
        Normalized edge DataFrame.
    """
    # Build denom lookup from small DataFrame
    denom_dict = dict(
        zip(
            denominators[loser_column].to_list(),
            denominators["denom"].to_list(),
        )
    )

    loser_ids = edges[loser_column].to_list()
    wts = edges[weight_column].to_numpy()
    normalized = np.empty(len(wts), dtype=float)
    for i, lid in enumerate(loser_ids):
        d = denom_dict.get(lid, 1.0)
        normalized[i] = wts[i] / d if d != 0.0 else 0.0

    return edges.with_columns(
        pl.Series("normalized_weight", normalized),
        pl.Series("denom", [denom_dict.get(lid, 1.0) for lid in loser_ids]),
    )


def edges_to_triplets(
    edges: pl.DataFrame,
    node_to_index: dict[Any, int],
    source_column: str,
    target_column: str,
    weight_column: str = "weight_sum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert edge DataFrame to COO triplets.

    Uses direct dict lookup instead of Polars joins for speed.

    Args:
        edges: Edge DataFrame.
        node_to_index: Node ID to index mapping.
        source_column: Source node column.
        target_column: Target node column.
        weight_column: Weight column.

    Returns:
        Tuple of (row_indices, col_indices, weights).
    """
    src_ids = edges[source_column].to_list()
    tgt_ids = edges[target_column].to_list()
    wts = edges[weight_column].to_numpy()

    n = len(src_ids)
    row_buf = np.empty(n, dtype=np.int64)
    col_buf = np.empty(n, dtype=np.int64)
    wt_buf = np.empty(n, dtype=np.float64)
    count = 0
    _get = node_to_index.get
    for i in range(n):
        s_idx = _get(src_ids[i])
        t_idx = _get(tgt_ids[i])
        if s_idx is not None and t_idx is not None:
            row_buf[count] = s_idx
            col_buf[count] = t_idx
            wt_buf[count] = wts[i]
            count += 1

    return row_buf[:count], col_buf[:count], wt_buf[:count]
