"""Data conversion utilities for ranking algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from loopr.core.edges import _prepare_weighted_matches
from loopr.schema import (
    prepare_rank_inputs,
    prepare_matches_frame,
    prepare_participants_frame,
)

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True)
class PreparedExposureMatches:
    """Reusable intermediate tables for exposure-based ranking."""

    matches: pl.DataFrame
    pair_edges: pl.DataFrame
    entity_metrics: pl.DataFrame


def _empty_pair_edges() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "winner_id": pl.Int64,
            "loser_id": pl.Int64,
            "share": pl.Float64,
        }
    )


def _empty_entity_metrics() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "id": pl.Int64,
            "share": pl.Float64,
            "weight": pl.Float64,
            "ts": pl.Float64,
        }
    )


def _prepare_weighted_match_frame(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float,
) -> pl.DataFrame:
    needed_columns = [
        "match_id",
        "tournament_id",
        "winner_team_id",
        "loser_team_id",
    ]

    for column_name in ["last_game_finished_at", "match_created_at"]:
        if column_name in matches.columns:
            needed_columns.append(column_name)

    if "is_bye" in matches.columns:
        needed_columns.append("is_bye")

    match_data = matches.select(
        [column_name for column_name in needed_columns if column_name in matches.columns]
    )
    match_data = _prepare_weighted_matches(
        match_data,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )
    return match_data.rename({"match_weight": "weight"})


def _group_team_members(
    players: pl.DataFrame,
) -> pl.DataFrame:
    if players.is_empty():
        return pl.DataFrame(
            schema={
                "tournament_id": pl.Int64,
                "team_id": pl.Int64,
                "user_id": pl.List(pl.Int64),
            }
        )
    return (
        players.select(["tournament_id", "team_id", "user_id"])
        .group_by(["tournament_id", "team_id"])
        .agg(pl.col("user_id"))
    )


def _group_match_appearances(
    appearances: pl.DataFrame | None,
    players: pl.DataFrame,
) -> pl.DataFrame | None:
    if appearances is None or appearances.is_empty():
        return None

    appearance_rows = appearances
    if "team_id" not in appearance_rows.columns:
        team_lookup = players.select(
            ["tournament_id", "user_id", "team_id"]
        ).unique(subset=["tournament_id", "user_id"], keep="any")
        appearance_rows = appearance_rows.join(
            team_lookup,
            on=["tournament_id", "user_id"],
            how="left",
        )

    appearance_rows = appearance_rows.drop_nulls("team_id")
    if appearance_rows.is_empty():
        return None

    return (
        appearance_rows.select(
            ["tournament_id", "match_id", "team_id", "user_id"]
        )
        .group_by(["tournament_id", "match_id", "team_id"])
        .agg(pl.col("user_id"))
    )


def _assign_match_rosters(
    match_data: pl.DataFrame,
    roster_source: pl.DataFrame,
    appearance_source: pl.DataFrame | None,
) -> pl.DataFrame:
    winner_rosters = roster_source.rename(
        {"team_id": "winner_team_id", "user_id": "winner_roster"}
    )
    loser_rosters = roster_source.rename(
        {"team_id": "loser_team_id", "user_id": "loser_roster"}
    )

    match_data = match_data.join(
        winner_rosters,
        on=["tournament_id", "winner_team_id"],
        how="left",
    ).join(
        loser_rosters,
        on=["tournament_id", "loser_team_id"],
        how="left",
    )

    if appearance_source is not None:
        winner_appearances = appearance_source.rename(
            {"team_id": "winner_team_id", "user_id": "winner_appearance"}
        )
        loser_appearances = appearance_source.rename(
            {"team_id": "loser_team_id", "user_id": "loser_appearance"}
        )
        match_data = match_data.join(
            winner_appearances,
            on=["tournament_id", "match_id", "winner_team_id"],
            how="left",
        ).join(
            loser_appearances,
            on=["tournament_id", "match_id", "loser_team_id"],
            how="left",
        )

        winners_expr = (
            pl.when(pl.col("winner_appearance").is_not_null())
            .then(pl.col("winner_appearance"))
            .otherwise(pl.col("winner_roster"))
            .alias("winners")
        )
        losers_expr = (
            pl.when(pl.col("loser_appearance").is_not_null())
            .then(pl.col("loser_appearance"))
            .otherwise(pl.col("loser_roster"))
            .alias("losers")
        )
    else:
        winners_expr = pl.col("winner_roster").alias("winners")
        losers_expr = pl.col("loser_roster").alias("losers")

    return (
        match_data.with_columns([winners_expr, losers_expr])
        .filter(
            pl.col("winners").is_not_null()
            & pl.col("losers").is_not_null()
            & (pl.col("winners").list.len() > 0)
            & (pl.col("losers").list.len() > 0)
        )
    )


def _aggregate_entity_metrics_from_matches(
    match_data: pl.DataFrame,
) -> pl.DataFrame:
    if match_data.is_empty():
        return _empty_entity_metrics()

    pieces = []
    value_columns = [
        column for column in ("share", "weight", "ts") if column in match_data.columns
    ]

    for entity_column in ("winners", "losers"):
        pieces.append(
            match_data.select([pl.col(entity_column).alias("id"), *value_columns])
            .explode("id")
            .drop_nulls("id")
        )

    aggregations = []
    if "share" in value_columns:
        aggregations.append(pl.col("share").sum().alias("share"))
    if "weight" in value_columns:
        aggregations.append(pl.col("weight").sum().alias("weight"))
    if "ts" in value_columns:
        aggregations.append(pl.col("ts").max().alias("ts"))

    return pl.concat(pieces).group_by("id").agg(aggregations)


def _build_exposure_pair_edges(
    match_data: pl.DataFrame,
) -> pl.DataFrame:
    if match_data.is_empty() or "share" not in match_data.columns:
        return _empty_pair_edges()

    return (
        match_data.select(["winners", "losers", "share"])
        .explode("winners")
        .drop_nulls("winners")
        .explode("losers")
        .drop_nulls("losers")
        .group_by(["winners", "losers"])
        .agg(pl.col("share").sum().alias("share"))
        .rename({"winners": "winner_id", "losers": "loser_id"})
    )


def _prepare_exposure_matches_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    include_share: bool = True,
    streaming: bool = False,
) -> PreparedExposureMatches:
    """Build compact match rows plus reusable exposure intermediates."""
    del streaming  # retained for API compatibility

    match_data = _prepare_weighted_match_frame(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )

    roster_source = _group_team_members(rosters if rosters is not None else players)
    appearance_source = _group_match_appearances(appearances, players)
    match_data = _assign_match_rosters(
        match_data,
        roster_source,
        appearance_source,
    )

    if include_share:
        match_data = match_data.with_columns(
            [
                pl.col("winners").list.len().alias("winner_count"),
                pl.col("losers").list.len().alias("loser_count"),
            ]
        ).with_columns(
            (
                pl.col("weight")
                / (pl.col("winner_count") * pl.col("loser_count"))
            ).alias("share")
        )

    final_columns = [
        "match_id",
        "tournament_id",
        "winners",
        "losers",
        "weight",
        "ts",
    ]
    if include_share:
        final_columns.extend(["winner_count", "loser_count", "share"])

    compact_matches = match_data.select(final_columns)

    if not include_share:
        return PreparedExposureMatches(
            matches=compact_matches,
            pair_edges=_empty_pair_edges(),
            entity_metrics=_aggregate_entity_metrics_from_matches(compact_matches),
        )

    return PreparedExposureMatches(
        matches=compact_matches,
        pair_edges=_build_exposure_pair_edges(compact_matches),
        entity_metrics=_aggregate_entity_metrics_from_matches(compact_matches),
    )


def convert_matches_dataframe(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    include_share: bool = True,
    streaming: bool = False,
) -> pl.DataFrame:
    """Build a compact matches table with winners/losers roster lists and weights.

    This is the optimized Polars-based conversion path.

    Args:
        matches: Match data with tournament_id, winner_team_id, loser_team_id.
        players: Player/roster data mapping users to teams.
        tournament_influence: Tournament ID to influence score mapping.
        now_timestamp: Current timestamp for decay calculations.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.
        rosters: Optional pre-filtered roster data. Defaults to None.
        appearances: Optional per-match player appearances. Supports two schemas:
            - With team context: columns [tournament_id, match_id, team_id, user_id]
            - Player-only: columns [tournament_id, match_id, user_id] (team inferred from rosters)
        include_share: Whether to include share calculations. Defaults to True.
        streaming: Whether to use streaming mode. Defaults to False.

    Returns:
        DataFrame with columns: match_id, tournament_id, winners (list of user_ids),
        losers (list of user_ids), weight (float), ts (timestamp). If include_share=True,
        also includes winner_count, loser_count, share.
    """
    prepared = prepare_rank_inputs(matches, players, appearances)
    rosters = (
        prepare_participants_frame(rosters) if rosters is not None else None
    )
    return _convert_matches_dataframe_normalized(
        prepared.matches,
        prepared.participants,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        rosters=rosters,
        appearances=prepared.appearances,
        include_share=include_share,
        streaming=streaming,
    )


def _convert_matches_dataframe_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    include_share: bool = True,
    streaming: bool = False,
) -> pl.DataFrame:
    """Internal conversion path that assumes all inputs are already normalized."""
    prepared = _prepare_exposure_matches_normalized(
        matches,
        players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        rosters=rosters,
        appearances=appearances,
        include_share=include_share,
        streaming=streaming,
    )
    return prepared.matches


def convert_matches_format(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert polars DataFrame matches to list of dicts with winners/losers lists.

    This is the fallback/legacy conversion path.

    Args:
        matches: Match data.
        players: Player/roster data.
        tournament_influence: Tournament influence scores.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.

    Returns:
        List of match dictionaries.
    """
    prepared = prepare_rank_inputs(matches, players)
    return _convert_matches_format_normalized(
        prepared.matches,
        prepared.participants,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )


def _convert_matches_format_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Internal fallback conversion path for already-normalized inputs."""
    converted = []

    for row in matches.iter_rows(named=True):
        if row.get("is_bye", False):
            continue

        winner_team_id = row.get("winner_team_id")
        loser_team_id = row.get("loser_team_id")

        if not winner_team_id or not loser_team_id:
            continue

        tournament_id = row["tournament_id"]

        winner_player_ids = players.filter(
            (pl.col("tournament_id") == tournament_id)
            & (pl.col("team_id") == winner_team_id)
        )["user_id"].to_list()

        loser_player_ids = players.filter(
            (pl.col("tournament_id") == tournament_id)
            & (pl.col("team_id") == loser_team_id)
        )["user_id"].to_list()

        if not winner_player_ids or not loser_player_ids:
            continue

        tournament_strength = tournament_influence.get(tournament_id, 1.0)

        if "last_game_finished_at" in row and row["last_game_finished_at"]:
            timestamp = row["last_game_finished_at"]
        elif "match_created_at" in row and row["match_created_at"]:
            timestamp = row["match_created_at"]
        else:
            timestamp = now_timestamp

        days_ago = (now_timestamp - timestamp) / 86400.0
        time_decay_factor = math.exp(-decay_rate * days_ago)

        weight = time_decay_factor * (tournament_strength**beta)

        converted.append(
            {
                "winners": winner_player_ids,
                "losers": loser_player_ids,
                "weight": weight,
                "tournament_id": tournament_id,
                "match_id": row.get("match_id"),
                "timestamp": timestamp,
            }
        )

    return converted


def convert_team_matches(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert team matches to required format.

    Args:
        matches: Match data with team IDs.
        tournament_influence: Tournament influence scores.
        now_timestamp: Current timestamp.
        decay_rate: Time decay rate.
        beta: Tournament influence exponent.

    Returns:
        List of match dictionaries with team IDs as single-element lists.
    """
    matches = prepare_matches_frame(matches)
    return _convert_team_matches_normalized(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )


def _convert_team_matches_normalized(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Internal team conversion path for already-normalized match inputs."""
    converted = []

    for row in matches.iter_rows(named=True):
        if row.get("is_bye", False):
            continue

        winner_team_id = row.get("winner_team_id")
        loser_team_id = row.get("loser_team_id")

        if not winner_team_id or not loser_team_id:
            continue

        tournament_id = row["tournament_id"]
        tournament_strength = tournament_influence.get(tournament_id, 1.0)

        if "last_game_finished_at" in row and row["last_game_finished_at"]:
            timestamp = row["last_game_finished_at"]
        elif "match_created_at" in row and row["match_created_at"]:
            timestamp = row["match_created_at"]
        else:
            timestamp = now_timestamp

        days_ago = (now_timestamp - timestamp) / 86400.0
        time_decay_factor = math.exp(-decay_rate * days_ago)

        weight = time_decay_factor * (tournament_strength**beta)

        converted.append(
            {
                "winners": [winner_team_id],
                "losers": [loser_team_id],
                "weight": weight,
                "tournament_id": tournament_id,
                "match_id": row.get("match_id"),
                "timestamp": timestamp,
            }
        )

    return converted


def factorize_ids(
    node_ids: list[Any],
) -> tuple[list[Any], dict[Any, int]]:
    """Convert list of IDs to indices.

    Args:
        node_ids: List of unique IDs.

    Returns:
        Tuple of (unique_ids, id_to_index_mapping).
    """
    unique_ids = list(dict.fromkeys(node_ids))
    id_to_index = {node_id: index for index, node_id in enumerate(unique_ids)}
    return unique_ids, id_to_index


def build_node_mapping(
    matches_dataframe: pl.DataFrame,
    winner_column: str = "winners",
    loser_column: str = "losers",
) -> tuple[list[Any], dict[Any, int]]:
    """Build node ID to index mapping from matches DataFrame.

    Args:
        matches_dataframe: DataFrame with winner/loser columns.
        winner_column: Name of winner column. Defaults to "winners".
        loser_column: Name of loser column. Defaults to "losers".

    Returns:
        Tuple of (node_list, node_to_index_map).
    """
    if (
        winner_column in matches_dataframe.columns
        and loser_column in matches_dataframe.columns
    ):
        if matches_dataframe[winner_column].dtype == pl.List:
            winner_ids = (
                matches_dataframe.select(pl.col(winner_column).list.explode())[
                    winner_column
                ]
                .unique()
                .to_list()
            )
            loser_ids = (
                matches_dataframe.select(pl.col(loser_column).list.explode())[
                    loser_column
                ]
                .unique()
                .to_list()
            )
        else:
            winner_ids = matches_dataframe[winner_column].unique().to_list()
            loser_ids = matches_dataframe[loser_column].unique().to_list()

        all_node_ids = list(set(winner_ids) | set(loser_ids))
    else:
        raise ValueError(f"Columns {winner_column} or {loser_column} not found")

    return factorize_ids(all_node_ids)
