"""Shared helpers for LOOPR log-odds style engines."""

from __future__ import annotations

import numpy as np
import polars as pl

from loopr.core.constants import (
    LAMBDA_TARGET_FRACTION,
    LOSERS,
    SHARE,
    WEIGHT,
    WINNERS,
)
from loopr.core.preparation import (
    aggregate_entity_metrics,
    appeared_entity_ids,
    build_index_mapping,
    merged_node_ids,
)


def _metric_vector_from_aggregated(
    aggregated_metrics: pl.DataFrame,
    index_mapping: pl.DataFrame | dict[int, int],
    value_column: str,
    num_nodes: int,
) -> np.ndarray:
    """Project an aggregated metric column onto node index order.

    Accepts index_mapping as either a Polars DataFrame (legacy) or a plain
    dict for faster lookups.
    """
    vector = np.zeros(num_nodes, dtype=float)
    if (
        aggregated_metrics.is_empty()
        or value_column not in aggregated_metrics.columns
    ):
        return vector

    # Build dict lookup if given a DataFrame
    if isinstance(index_mapping, pl.DataFrame):
        if index_mapping.is_empty():
            return vector
        idx_dict = dict(
            zip(
                index_mapping["id"].to_list(),
                index_mapping["idx"].to_list(),
            )
        )
    else:
        idx_dict = index_mapping

    ids = aggregated_metrics["id"].to_list()
    vals = aggregated_metrics[value_column].to_numpy()
    _get = idx_dict.get
    for i, entity_id in enumerate(ids):
        idx = _get(entity_id)
        if idx is not None:
            vector[idx] = vals[i]
    return vector

def _metric_vector(
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    value_column: str,
    output_column: str,
    *,
    aggregated_metrics: pl.DataFrame | None = None,
    index_mapping: pl.DataFrame | None = None,
) -> np.ndarray:
    if aggregated_metrics is not None:
        if index_mapping is None:
            index_mapping = build_index_mapping(node_to_idx)
        return _metric_vector_from_aggregated(
            aggregated_metrics,
            index_mapping,
            value_column,
            len(node_to_idx),
        )

    if matches_df.is_empty():
        return np.zeros(len(node_to_idx), dtype=float)

    pieces = []
    for entity_column in (WINNERS, LOSERS):
        if entity_column not in matches_df.columns:
            continue
        pieces.append(
            matches_df.select([entity_column, value_column])
            .explode(entity_column)
            .rename({entity_column: "id"})
        )

    if not pieces:
        return np.zeros(len(node_to_idx), dtype=float)

    aggregated = (
        pl.concat(pieces)
        .drop_nulls("id")
        .group_by("id")
        .agg(pl.col(value_column).sum().alias(output_column))
    )
    return _metric_vector_from_aggregated(
        aggregated,
        node_to_idx,
        output_column,
        len(node_to_idx),
    )


def teleport_from_share(
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    *,
    aggregated_metrics: pl.DataFrame | None = None,
    index_mapping: pl.DataFrame | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Build a normalized teleport vector from per-match exposure share."""
    rho = _metric_vector(
        matches_df,
        node_to_idx,
        SHARE,
        "e_share",
        aggregated_metrics=aggregated_metrics,
        index_mapping=index_mapping,
    ) + epsilon
    total = float(rho.sum())
    if total == 0.0 or not np.isfinite(total):
        rho[:] = 1.0
        total = float(rho.sum())
    return rho / total


def reporting_exposure(
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    *,
    aggregated_metrics: pl.DataFrame | None = None,
    index_mapping: pl.DataFrame | None = None,
) -> np.ndarray:
    """Build the user-facing exposure vector from per-match weights."""
    return _metric_vector(
        matches_df,
        node_to_idx,
        WEIGHT,
        "exposure",
        aggregated_metrics=aggregated_metrics,
        index_mapping=index_mapping,
    )


def last_activity_from_metrics(
    aggregated_metrics: pl.DataFrame,
    index_mapping: pl.DataFrame,
    num_nodes: int,
    default_timestamp: float,
) -> np.ndarray:
    """Project the max entity timestamp onto node index order."""
    last_ts = _metric_vector_from_aggregated(
        aggregated_metrics,
        index_mapping,
        "ts",
        num_nodes,
    )
    last_ts[last_ts == 0.0] = float(default_timestamp)
    return last_ts


def resolve_lambda(
    win_pagerank: np.ndarray,
    rho: np.ndarray,
    *,
    lambda_mode: str,
    fixed_lambda: float | None,
    fallback: float,
) -> float:
    """Resolve the smoothing lambda used by log-odds ranking."""
    if fixed_lambda is not None:
        return float(fixed_lambda)
    if lambda_mode == "auto":
        target = LAMBDA_TARGET_FRACTION * float(np.median(win_pagerank))
        median_rho = float(np.median(rho))
        return 0.0 if median_rho == 0.0 else max(target / median_rho, 0.0)
    return fallback
