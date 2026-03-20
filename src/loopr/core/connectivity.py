"""Connectivity helpers for resolved LOOPR comparison graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from loopr.core.preparation import PreparedGraphInputs


@dataclass(frozen=True)
class ConnectivityReport:
    """Summary of connected-component structure for a resolved graph."""

    component_count: int
    total_entity_count: int
    largest_component_entity_count: int
    largest_component_entity_ids: tuple[int, ...]
    largest_component_node_fraction: float | None
    largest_component_share_fraction: float | None
    largest_component_weight_fraction: float | None
    disconnected_share_fraction: float | None
    disconnected_weight_fraction: float | None

    @property
    def is_disconnected(self) -> bool:
        return self.component_count > 1


def _undirected_adjacency(graph_inputs: PreparedGraphInputs) -> csr_matrix:
    entity_metrics = graph_inputs.entity_metrics
    if entity_metrics.is_empty():
        return csr_matrix((0, 0), dtype=float)

    node_ids = entity_metrics["id"].to_list()
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    pair_edges = graph_inputs.pair_edges
    if pair_edges.is_empty():
        return csr_matrix((len(node_ids), len(node_ids)), dtype=float)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for winner_id, loser_id, share in pair_edges.iter_rows():
        winner_idx = node_to_idx[int(winner_id)]
        loser_idx = node_to_idx[int(loser_id)]
        share_value = float(share)
        rows.extend([winner_idx, loser_idx])
        cols.extend([loser_idx, winner_idx])
        data.extend([share_value, share_value])

    adjacency = coo_matrix(
        (data, (rows, cols)),
        shape=(len(node_ids), len(node_ids)),
        dtype=float,
    ).tocsr()
    adjacency.sum_duplicates()
    return adjacency


def analyze_graph_connectivity(
    graph_inputs: PreparedGraphInputs,
) -> ConnectivityReport:
    """Analyze component structure on the undirected pair-share graph."""
    entity_metrics = graph_inputs.entity_metrics.sort("id")
    if entity_metrics.is_empty():
        return ConnectivityReport(
            component_count=0,
            total_entity_count=0,
            largest_component_entity_count=0,
            largest_component_entity_ids=(),
            largest_component_node_fraction=None,
            largest_component_share_fraction=None,
            largest_component_weight_fraction=None,
            disconnected_share_fraction=None,
            disconnected_weight_fraction=None,
        )

    node_ids = entity_metrics["id"].to_list()
    share = (
        np.array(entity_metrics["share"].to_list(), dtype=float)
        if "share" in entity_metrics.columns
        else np.ones(len(node_ids), dtype=float)
    )
    weight = (
        np.array(entity_metrics["weight"].to_list(), dtype=float)
        if "weight" in entity_metrics.columns
        else np.ones(len(node_ids), dtype=float)
    )
    adjacency = _undirected_adjacency(
        PreparedGraphInputs(
            matches=graph_inputs.matches,
            entity_metrics=entity_metrics,
            pair_edges=graph_inputs.pair_edges,
            node_ids=graph_inputs.node_ids,
            node_to_idx=graph_inputs.node_to_idx,
            index_mapping=graph_inputs.index_mapping,
        )
    )

    if adjacency.shape[0] == 0:
        component_count = entity_metrics.height
        largest_idx = int(np.argmax(share))
        largest_mask = np.zeros(entity_metrics.height, dtype=bool)
        largest_mask[largest_idx] = True
        labels = np.arange(entity_metrics.height, dtype=int)
    elif adjacency.nnz == 0:
        component_count = entity_metrics.height
        largest_idx = int(np.argmax(share))
        largest_mask = np.zeros(entity_metrics.height, dtype=bool)
        largest_mask[largest_idx] = True
        labels = np.arange(entity_metrics.height, dtype=int)
    else:
        component_count, labels = connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
        component_share = np.bincount(labels, weights=share, minlength=component_count)
        largest_component = int(np.argmax(component_share))
        largest_mask = labels == largest_component

    if adjacency.nnz == 0:
        component_share = share
        component_weight = weight
    else:
        component_share = np.bincount(labels, weights=share, minlength=component_count)
        component_weight = np.bincount(
            labels,
            weights=weight,
            minlength=component_count,
        )

    largest_component_entity_ids = tuple(
        int(entity_id)
        for entity_id, keep in zip(node_ids, largest_mask, strict=True)
        if keep
    )
    largest_share_fraction = (
        float(component_share.max() / component_share.sum())
        if component_share.sum() > 0
        else None
    )
    largest_weight_fraction = (
        float(component_weight.max() / component_weight.sum())
        if component_weight.sum() > 0
        else None
    )
    largest_node_fraction = (
        float(largest_mask.sum() / len(node_ids)) if node_ids else None
    )

    disconnected_share_fraction = (
        None
        if largest_share_fraction is None
        else max(0.0, 1.0 - largest_share_fraction)
    )
    disconnected_weight_fraction = (
        None
        if largest_weight_fraction is None
        else max(0.0, 1.0 - largest_weight_fraction)
    )

    return ConnectivityReport(
        component_count=int(component_count),
        total_entity_count=len(node_ids),
        largest_component_entity_count=int(largest_mask.sum()),
        largest_component_entity_ids=largest_component_entity_ids,
        largest_component_node_fraction=largest_node_fraction,
        largest_component_share_fraction=largest_share_fraction,
        largest_component_weight_fraction=largest_weight_fraction,
        disconnected_share_fraction=disconnected_share_fraction,
        disconnected_weight_fraction=disconnected_weight_fraction,
    )


def filter_resolved_matches_to_entities(
    resolved_matches: pl.DataFrame,
    entity_ids: tuple[int, ...] | list[int] | set[int],
) -> pl.DataFrame:
    """Keep only resolved matches whose winners/losers stay in the given set."""
    if resolved_matches.is_empty():
        return resolved_matches

    allowed_ids = list(entity_ids)
    return resolved_matches.filter(
        pl.col("winners")
        .list.eval(pl.element().is_in(allowed_ids))
        .list.all()
        & pl.col("losers")
        .list.eval(pl.element().is_in(allowed_ids))
        .list.all()
    )
