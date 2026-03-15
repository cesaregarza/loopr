"""Shared helpers for LOOPR log-odds style engines."""

from __future__ import annotations

import numpy as np
import polars as pl


def build_index_mapping(node_to_idx: dict[int, int]) -> pl.DataFrame:
    """Materialize a reusable ID->index lookup frame."""
    valid_items = [
        (entity_id, idx)
        for entity_id, idx in node_to_idx.items()
        if entity_id is not None
    ]
    if not valid_items:
        return pl.DataFrame(schema={"id": pl.Int64, "idx": pl.Int64})

    entity_ids, indices = zip(*valid_items)
    return pl.DataFrame({"id": list(entity_ids), "idx": list(indices)})


def aggregate_entity_metrics(
    matches_df: pl.DataFrame,
    *,
    precomputed: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Aggregate per-entity share, weight, and last activity once."""
    if precomputed is not None:
        return precomputed

    if matches_df.is_empty():
        return pl.DataFrame(
            schema={
                "id": pl.Int64,
                "share": pl.Float64,
                "weight": pl.Float64,
                "ts": pl.Float64,
            }
        )

    pieces = []
    value_columns = [
        column
        for column in ("share", "weight", "ts")
        if column in matches_df.columns
    ]

    for entity_column in ("winners", "losers"):
        if entity_column not in matches_df.columns:
            continue
        pieces.append(
            matches_df.select(
                [pl.col(entity_column).alias("id"), *value_columns]
            ).explode("id")
        )

    if not pieces:
        return pl.DataFrame(
            schema={
                "id": pl.Int64,
                "share": pl.Float64,
                "weight": pl.Float64,
                "ts": pl.Float64,
            }
        )

    aggregations = []
    if "share" in value_columns:
        aggregations.append(pl.col("share").sum().alias("share"))
    if "weight" in value_columns:
        aggregations.append(pl.col("weight").sum().alias("weight"))
    if "ts" in value_columns:
        aggregations.append(pl.col("ts").max().alias("ts"))

    return (
        pl.concat(pieces)
        .drop_nulls("id")
        .group_by("id")
        .agg(aggregations)
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


def appeared_entity_ids(matches_df: pl.DataFrame) -> set[int]:
    """Return all entities that appear in the normalized winners/losers lists."""
    entity_ids: set[int] = set()
    for column in ("winners", "losers"):
        if column not in matches_df.columns or matches_df.is_empty():
            continue
        entity_ids.update(
            matches_df.select(column)
            .explode(column)[column]
            .drop_nulls()
            .unique()
            .to_list()
        )
    return entity_ids


def merged_node_ids(
    matches_df: pl.DataFrame,
    active_ids: list[int] | None = None,
    *,
    aggregated_metrics: pl.DataFrame | None = None,
) -> list[int]:
    """Combine active IDs with actually-appeared IDs in deterministic order."""
    if aggregated_metrics is not None:
        merged = set(aggregated_metrics["id"].to_list())
    else:
        merged = appeared_entity_ids(matches_df)
    if active_ids:
        merged.update(active_ids)
    return sorted(merged)


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
    for entity_column in ("winners", "losers"):
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
        "share",
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
        "weight",
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
        target = 0.025 * float(np.median(win_pagerank))
        median_rho = float(np.median(rho))
        return 0.0 if median_rho == 0.0 else max(target / median_rho, 0.0)
    return fallback
