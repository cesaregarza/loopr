"""Shared helpers for LOOPR log-odds style engines."""

from __future__ import annotations

import numpy as np
import polars as pl


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
) -> list[int]:
    """Combine active IDs with actually-appeared IDs in deterministic order."""
    merged = appeared_entity_ids(matches_df)
    if active_ids:
        merged.update(active_ids)
    return sorted(merged)


def _metric_vector(
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    value_column: str,
    output_column: str,
) -> np.ndarray:
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

    vector = np.zeros(len(node_to_idx), dtype=float)
    for row in aggregated.iter_rows(named=True):
        idx = node_to_idx.get(row["id"])
        if idx is not None:
            vector[idx] = float(row[output_column])
    return vector


def teleport_from_share(
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Build a normalized teleport vector from per-match exposure share."""
    rho = _metric_vector(matches_df, node_to_idx, "share", "e_share") + epsilon
    total = float(rho.sum())
    if total == 0.0 or not np.isfinite(total):
        rho[:] = 1.0
        total = float(rho.sum())
    return rho / total


def reporting_exposure(
    matches_df: pl.DataFrame,
    node_to_idx: dict[int, int],
) -> np.ndarray:
    """Build the user-facing exposure vector from per-match weights."""
    return _metric_vector(matches_df, node_to_idx, "weight", "exposure")


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
