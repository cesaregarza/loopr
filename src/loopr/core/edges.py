"""Edge building utilities for ranking algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from loopr.core.constants import (
    LOSERS,
    LOSER_USER_ID,
    NORMALIZED_WEIGHT,
    SHARE,
    WEIGHT_SUM,
    WINNERS,
    WINNER_USER_ID,
)
from loopr.core.preparation import (
    PreparedGraphInputs,
    build_team_edge_dataframe,
    group_team_members,
    prepare_row_edge_inputs,
    prepare_weighted_matches,
)
from loopr.schema import (
    prepare_matches_frame,
    prepare_rank_inputs,
)

if TYPE_CHECKING:
    from typing import Any

    from loopr.core.smoothing import SmoothingStrategy


def _empty_triplets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
    )


def _sorted_mapping_arrays(
    *,
    node_to_index: dict[Any, int] | None = None,
    index_mapping: pl.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if node_to_index is not None:
        valid_items = [
            (entity_id, idx)
            for entity_id, idx in node_to_index.items()
            if entity_id is not None
        ]
        if not valid_items:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        ids, indices = zip(*valid_items)
        mapping_ids = np.asarray(ids)
        mapping_idx = np.asarray(indices, dtype=np.int64)
    else:
        if index_mapping is None or index_mapping.is_empty():
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        mapping_ids = np.asarray(index_mapping["id"].to_numpy())
        mapping_idx = index_mapping["idx"].to_numpy().astype(np.int64, copy=False)

    order = np.argsort(mapping_ids)
    return mapping_ids[order], mapping_idx[order]


def _lookup_sorted(
    lookup_ids: np.ndarray,
    mapping_ids: np.ndarray,
    mapping_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if lookup_ids.size == 0 or mapping_ids.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=bool)

    positions = np.searchsorted(mapping_ids, lookup_ids)
    in_bounds = positions < mapping_ids.size
    safe_positions = positions.copy()
    safe_positions[~in_bounds] = 0
    matched = in_bounds & (mapping_ids[safe_positions] == lookup_ids)

    result = np.empty(lookup_ids.size, dtype=mapping_values.dtype)
    if mapping_values.size:
        result[:] = mapping_values[0]
        result[matched] = mapping_values[safe_positions[matched]]
    return result, matched


def _triplets_from_joined_edges(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    weights: np.ndarray,
    *,
    mapping_ids: np.ndarray,
    mapping_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_ids.size == 0 or target_ids.size == 0 or mapping_ids.size == 0:
        return _empty_triplets()

    rows, row_valid = _lookup_sorted(source_ids, mapping_ids, mapping_idx)
    cols, col_valid = _lookup_sorted(target_ids, mapping_ids, mapping_idx)
    valid = row_valid & col_valid
    if not np.any(valid):
        return _empty_triplets()

    return (
        rows[valid].astype(np.int64, copy=False),
        cols[valid].astype(np.int64, copy=False),
        weights[valid].astype(np.float64, copy=False),
    )


def build_player_edges(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Build entity-level loser->winner edges with tournament weighting."""
    prepared = prepare_rank_inputs(matches, players)
    prepared_matches = prepared.matches
    prepared_players = prepared.participants
    return _build_player_edges_normalized(
        prepared_matches,
        prepared_players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        timestamp_column=timestamp_column,
    )


def _build_player_edges_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    timestamp_column: str | None = None,
) -> pl.DataFrame:
    """Internal player-edge construction for already-normalized inputs."""
    roster_source = None
    if (
        players is not None
        and {"tournament_id", "team_id", "user_id"}.issubset(players.columns)
    ):
        roster_source = group_team_members(players)
    prepared = prepare_row_edge_inputs(
        matches,
        players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        rosters=roster_source,
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
    mapping_ids, mapping_idx = _sorted_mapping_arrays(
        node_to_index=node_to_index,
        index_mapping=index_mapping,
    )
    if mapping_ids.size == 0:
        return _empty_triplets()

    if pair_edges is None:
        if matches_df.is_empty():
            return _empty_triplets()
        pair_edges = (
            matches_df.select([WINNERS, LOSERS, SHARE])
            .explode(WINNERS)
            .drop_nulls(WINNERS)
            .explode(LOSERS)
            .drop_nulls(LOSERS)
            .group_by([WINNERS, LOSERS])
            .agg(pl.col(SHARE).sum().alias(SHARE))
            .rename({WINNERS: "winner_id", LOSERS: "loser_id"})
        )
    elif pair_edges.is_empty():
        return _empty_triplets()

    return _triplets_from_joined_edges(
        np.asarray(pair_edges["winner_id"].to_numpy()),
        np.asarray(pair_edges["loser_id"].to_numpy()),
        pair_edges["share"].to_numpy().astype(np.float64, copy=False),
        mapping_ids=mapping_ids,
        mapping_idx=mapping_idx,
    )


def compute_denominators(
    edges: pl.DataFrame,
    smoothing_strategy: SmoothingStrategy | None,
    loser_column: str = LOSER_USER_ID,
    winner_column: str = WINNER_USER_ID,
    weight_column: str = WEIGHT_SUM,
) -> pl.DataFrame:
    """Compute denominators with smoothing for edge normalization."""
    if edges.is_empty():
        return pl.DataFrame(
            schema={
                loser_column: edges.schema.get(loser_column, pl.Int64),
                "loss_weights": pl.Float64,
                "win_weights": pl.Float64,
                "denom": pl.Float64,
                "lambda": pl.Float64,
            }
        )

    denominators = (
        edges.group_by(loser_column)
        .agg(pl.col(weight_column).sum().alias("loss_weights"))
        .sort(loser_column)
    )
    win_sums = (
        edges.group_by(winner_column)
        .agg(pl.col(weight_column).sum().alias("win_weights"))
        .sort(winner_column)
    )

    loss_arr = denominators["loss_weights"].to_numpy().astype(
        np.float64, copy=False
    )
    denom_ids = np.asarray(denominators[loser_column].to_numpy())
    win_ids = np.asarray(win_sums[winner_column].to_numpy())
    win_values = win_sums["win_weights"].to_numpy().astype(np.float64, copy=False)
    win_arr = np.zeros(denom_ids.size, dtype=np.float64)
    if win_ids.size:
        aligned_win, matched = _lookup_sorted(denom_ids, win_ids, win_values)
        win_arr[matched] = aligned_win[matched]

    if smoothing_strategy:
        denom_arr = smoothing_strategy.denom(loss_arr, win_arr)
    else:
        denom_arr = loss_arr.copy()

    lambda_arr = denom_arr - loss_arr

    return pl.DataFrame(
        {
            loser_column: denom_ids.tolist(),
            "loss_weights": loss_arr,
            "win_weights": win_arr,
            "denom": denom_arr,
            "lambda": lambda_arr,
        }
    )


def normalize_edges(
    edges: pl.DataFrame,
    denominators: pl.DataFrame,
    loser_column: str = LOSER_USER_ID,
    weight_column: str = WEIGHT_SUM,
) -> pl.DataFrame:
    """Normalize edge weights by denominators."""
    if edges.is_empty():
        return edges.with_columns(
            pl.Series(name=NORMALIZED_WEIGHT, values=[], dtype=pl.Float64),
            pl.Series(name="denom", values=[], dtype=pl.Float64),
        )

    edge_ids = np.asarray(edges[loser_column].to_numpy())
    weights = edges[weight_column].to_numpy().astype(np.float64, copy=False)
    denom_ids = np.asarray(denominators[loser_column].to_numpy())
    denom_values = denominators["denom"].to_numpy().astype(np.float64, copy=False)

    expanded_denoms = np.ones(edge_ids.size, dtype=np.float64)
    if denom_ids.size:
        mapped_denoms, matched = _lookup_sorted(edge_ids, denom_ids, denom_values)
        expanded_denoms[matched] = mapped_denoms[matched]

    normalized = np.zeros(edge_ids.size, dtype=np.float64)
    np.divide(
        weights,
        expanded_denoms,
        out=normalized,
        where=expanded_denoms != 0.0,
    )

    return edges.with_columns(
        pl.Series("denom", expanded_denoms),
        pl.Series(NORMALIZED_WEIGHT, normalized),
    )


def edges_to_triplets(
    edges: pl.DataFrame,
    node_to_index: dict[Any, int],
    source_column: str,
    target_column: str,
    weight_column: str = WEIGHT_SUM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert edge DataFrame to COO triplets."""
    mapping_ids, mapping_idx = _sorted_mapping_arrays(node_to_index=node_to_index)
    if mapping_ids.size == 0:
        return _empty_triplets()

    return _triplets_from_joined_edges(
        np.asarray(edges[source_column].to_numpy()),
        np.asarray(edges[target_column].to_numpy()),
        edges[weight_column].to_numpy().astype(np.float64, copy=False),
        mapping_ids=mapping_ids,
        mapping_idx=mapping_idx,
    )
