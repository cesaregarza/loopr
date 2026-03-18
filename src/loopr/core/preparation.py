"""Shared internal preparation pipeline for LOOPR ranking engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from loopr.core.constants import (
    LOSERS,
    LOSER_USER_ID,
    MATCH_ID,
    NORMALIZED_WEIGHT,
    SECONDS_PER_DAY,
    SHARE,
    TOURNAMENT_ID,
    WEIGHT,
    WEIGHT_SUM,
    WINNERS,
    WINNER_USER_ID,
)


def _sample_rows(
    frame: pl.DataFrame,
    columns: list[str],
    *,
    limit: int = 5,
) -> str:
    sample = (
        frame.select(columns)
        .unique()
        .head(limit)
        .iter_rows(named=True)
    )
    return ", ".join(
        "/".join(str(row[column]) for column in columns)
        for row in sample
    )


@dataclass(frozen=True)
class WeightedMatches:
    """Normalized matches with timestamps and weights attached."""

    matches: pl.DataFrame


@dataclass(frozen=True)
class ResolvedMatches:
    """Weighted matches with winner/loser entity lists resolved."""

    weighted_matches: pl.DataFrame
    matches: pl.DataFrame


@dataclass(frozen=True)
class PreparedGraphInputs:
    """Resolved matches plus reusable exposure graph artifacts."""

    matches: pl.DataFrame
    entity_metrics: pl.DataFrame
    pair_edges: pl.DataFrame
    node_ids: list[Any]
    node_to_idx: dict[Any, int]
    index_mapping: pl.DataFrame


@dataclass(frozen=True)
class PreparedRowEdges:
    """Resolved matches plus row-oriented loser->winner edges."""

    matches: pl.DataFrame
    edges: pl.DataFrame
    node_ids: list[Any]
    node_to_idx: dict[Any, int]


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


def _empty_resolved_matches(include_share: bool) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "match_id": pl.Int64,
        "tournament_id": pl.Int64,
        "winners": pl.List(pl.Int64),
        "losers": pl.List(pl.Int64),
        "weight": pl.Float64,
        "ts": pl.Float64,
    }
    if include_share:
        schema["winner_count"] = pl.Int64
        schema["loser_count"] = pl.Int64
        schema["share"] = pl.Float64
    return pl.DataFrame(schema=schema)


def _empty_row_edges() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "loser_user_id": pl.Int64,
            "winner_user_id": pl.Int64,
            "weight_sum": pl.Float64,
        }
    )


def build_index_mapping(node_to_idx: dict[Any, int]) -> pl.DataFrame:
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
        return _empty_entity_metrics()

    pieces = []
    value_columns = [
        column
        for column in (SHARE, WEIGHT, "ts")
        if column in matches_df.columns
    ]

    for entity_column in (WINNERS, LOSERS):
        if entity_column not in matches_df.columns:
            continue
        pieces.append(
            matches_df.select(
                [pl.col(entity_column).alias("id"), *value_columns]
            )
            .explode("id")
            .drop_nulls("id")
        )

    if not pieces:
        return _empty_entity_metrics()

    aggregations = []
    if SHARE in value_columns:
        aggregations.append(pl.col(SHARE).sum().alias(SHARE))
    if WEIGHT in value_columns:
        aggregations.append(pl.col(WEIGHT).sum().alias(WEIGHT))
    if "ts" in value_columns:
        aggregations.append(pl.col("ts").max().alias("ts"))

    return pl.concat(pieces).group_by("id").agg(aggregations)


def appeared_entity_ids(matches_df: pl.DataFrame) -> set[int]:
    """Return all entities that appear in the normalized winners/losers lists."""
    entity_ids: set[int] = set()
    for column in (WINNERS, LOSERS):
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


def _attach_weight_columns(
    match_data: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float,
    *,
    timestamp_column: str | None = None,
    legacy_timestamp_fill_after_influence: bool = False,
) -> pl.DataFrame:
    if tournament_influence:
        strength_dataframe = pl.DataFrame(
            {
                TOURNAMENT_ID: list(tournament_influence.keys()),
                "tournament_strength": list(tournament_influence.values()),
            }
        )
        match_data = match_data.join(
            strength_dataframe,
            on=TOURNAMENT_ID,
            how="left",
            coalesce=True,
        ).with_columns(pl.col("tournament_strength").fill_null(1.0))

        if legacy_timestamp_fill_after_influence:
            # Legacy Sendou edge-building filled nulls after the influence
            # join, which turns missing timestamps into 1.0 before the later
            # coalesce. Exposure conversion does not do this, so keep it
            # opt-in for the row/team edge parity paths only.
            fill_columns = {
                column_name
                for column_name in (
                    "last_game_finished_at",
                    "match_created_at",
                    timestamp_column,
                )
                if column_name is not None and column_name in match_data.columns
            }
            if fill_columns:
                match_data = match_data.with_columns(
                    [
                        pl.col(column_name).fill_null(1.0).alias(column_name)
                        for column_name in sorted(fill_columns)
                    ]
                )
    else:
        match_data = match_data.with_columns(
            pl.lit(1.0).alias("tournament_strength")
        )

    if timestamp_column and timestamp_column in match_data.columns:
        match_data = match_data.with_columns(
            pl.col(timestamp_column).cast(pl.Float64).alias("ts")
        )
    else:
        timestamp_expressions: list[pl.Expr] = []
        if "last_game_finished_at" in match_data.columns:
            timestamp_expressions.append(
                pl.col("last_game_finished_at").cast(pl.Float64)
            )
        if "match_created_at" in match_data.columns:
            timestamp_expressions.append(
                pl.col("match_created_at").cast(pl.Float64)
            )
        timestamp_expressions.append(pl.lit(float(now_timestamp)))
        match_data = match_data.with_columns(
            pl.coalesce(timestamp_expressions).alias("ts")
        )

    time_decay_factor = (
        ((pl.lit(float(now_timestamp)) - pl.col("ts")) / SECONDS_PER_DAY)
        .mul(-decay_rate)
        .exp()
    )
    if beta == 0.0:
        weight_expression = time_decay_factor
    else:
        weight_expression = time_decay_factor * (
            pl.col("tournament_strength") ** beta
        )

    return match_data.with_columns(weight_expression.alias(WEIGHT))


def prepare_weighted_matches(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float,
    *,
    timestamp_column: str | None = None,
    legacy_timestamp_fill_after_influence: bool = False,
) -> WeightedMatches:
    """Filter byes, resolve timestamps, join influence, and compute weights."""
    filter_expression = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )

    if "is_bye" in matches.columns:
        filter_expression = filter_expression & ~pl.col("is_bye").fill_null(
            False
        )

    match_data = matches.filter(filter_expression)

    match_data = _attach_weight_columns(
        match_data,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        timestamp_column=timestamp_column,
        legacy_timestamp_fill_after_influence=legacy_timestamp_fill_after_influence,
    )
    return WeightedMatches(match_data)


def _group_team_members(participants: pl.DataFrame) -> pl.DataFrame:
    if participants.is_empty():
        return pl.DataFrame(
            schema={
                "tournament_id": pl.Int64,
                "team_id": pl.Int64,
                "user_id": pl.List(pl.Int64),
            }
        )
    return (
        participants.select(["tournament_id", "team_id", "user_id"])
        .group_by(["tournament_id", "team_id"])
        .agg(pl.col("user_id"))
    )


def group_team_members(participants: pl.DataFrame) -> pl.DataFrame:
    """Group per-team participant rows into roster lists."""
    return _group_team_members(participants)


def _group_match_appearances(
    appearances: pl.DataFrame | None,
    participants: pl.DataFrame,
) -> pl.DataFrame | None:
    if appearances is None or appearances.is_empty():
        return None

    appearance_rows = appearances
    if "team_id" not in appearance_rows.columns:
        ambiguous_participants = (
            participants.select(["tournament_id", "user_id", "team_id"])
            .group_by(["tournament_id", "user_id"])
            .agg(pl.col("team_id").n_unique().alias("team_count"))
            .filter(pl.col("team_count") > 1)
        )
        if ambiguous_participants.height > 0:
            sample = _sample_rows(
                ambiguous_participants,
                ["tournament_id", "user_id"],
            )
            raise ValueError(
                "appearances is missing group_id for entities assigned to "
                "multiple groups within the same event. Add group_id to "
                f"disambiguate. Sample event_id/entity_id pairs: {sample}"
            )

        team_lookup = participants.select(
            ["tournament_id", "user_id", "team_id"]
        ).unique(subset=["tournament_id", "user_id"], keep="any")
        appearance_rows = appearance_rows.join(
            team_lookup,
            on=["tournament_id", "user_id"],
            how="left",
        )

    unresolved_rows = appearance_rows.filter(pl.col("team_id").is_null())
    if unresolved_rows.height > 0:
        sample = _sample_rows(
            unresolved_rows,
            ["tournament_id", "match_id", "user_id"],
        )
        raise ValueError(
            "Could not infer group_id for some appearances. Ensure each "
            "appearance entity exists in participants for the same event. "
            f"Sample event_id/match_id/entity_id rows: {sample}"
        )

    if appearance_rows.is_empty():
        return None

    return (
        appearance_rows.select(
            ["tournament_id", "match_id", "team_id", "user_id"]
        )
        .group_by(["tournament_id", "match_id", "team_id"])
        .agg(pl.col("user_id"))
    )


def _is_grouped_lookup(
    frame: pl.DataFrame | None,
    *,
    required_columns: set[str],
) -> bool:
    if frame is None or not required_columns.issubset(frame.columns):
        return False
    return isinstance(frame.schema.get("user_id"), pl.List)


def _resolve_roster_source(
    participants: pl.DataFrame,
    rosters: pl.DataFrame | None,
) -> pl.DataFrame:
    if _is_grouped_lookup(
        rosters,
        required_columns={TOURNAMENT_ID, "team_id", "user_id"},
    ):
        return rosters
    return _group_team_members(rosters if rosters is not None else participants)


def _resolve_appearance_source(
    appearances: pl.DataFrame | None,
    participants: pl.DataFrame,
) -> pl.DataFrame | None:
    if _is_grouped_lookup(
        appearances,
        required_columns={TOURNAMENT_ID, MATCH_ID, "team_id", "user_id"},
    ):
        return appearances
    return _group_match_appearances(appearances, participants)


def _assign_match_rosters(
    weighted_matches: pl.DataFrame,
    roster_source: pl.DataFrame,
    appearance_source: pl.DataFrame | None,
) -> pl.DataFrame:
    winner_rosters = roster_source.rename(
        {"team_id": "winner_team_id", "user_id": "winner_roster"}
    )
    loser_rosters = roster_source.rename(
        {"team_id": "loser_team_id", "user_id": "loser_roster"}
    )

    match_data = weighted_matches.join(
        winner_rosters,
        on=[TOURNAMENT_ID, "winner_team_id"],
        how="left",
    ).join(
        loser_rosters,
        on=[TOURNAMENT_ID, "loser_team_id"],
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
            on=[TOURNAMENT_ID, MATCH_ID, "winner_team_id"],
            how="left",
        ).join(
            loser_appearances,
            on=[TOURNAMENT_ID, MATCH_ID, "loser_team_id"],
            how="left",
        )

        winners_expr = (
            pl.when(pl.col("winner_appearance").is_not_null())
            .then(pl.col("winner_appearance"))
            .otherwise(pl.col("winner_roster"))
            .alias(WINNERS)
        )
        losers_expr = (
            pl.when(pl.col("loser_appearance").is_not_null())
            .then(pl.col("loser_appearance"))
            .otherwise(pl.col("loser_roster"))
            .alias(LOSERS)
        )
    else:
        winners_expr = pl.col("winner_roster").alias(WINNERS)
        losers_expr = pl.col("loser_roster").alias(LOSERS)

    resolved_match_data = match_data.with_columns([winners_expr, losers_expr])
    invalid_rows = resolved_match_data.filter(
        pl.col(WINNERS).is_null()
        | pl.col(LOSERS).is_null()
        | (pl.col(WINNERS).list.len() == 0)
        | (pl.col(LOSERS).list.len() == 0)
    )
    if invalid_rows.height > 0:
        sample = _sample_rows(
            invalid_rows,
            [MATCH_ID, TOURNAMENT_ID],
        )
        raise ValueError(
            "Could not resolve winner/loser participants for some matches. "
            "Ensure participants include both teams for every match and that "
            "appearances align with those teams. "
            f"Sample match_id/event_id pairs: {sample}"
        )

    return resolved_match_data


def _finalize_resolved_match_rows(
    match_data: pl.DataFrame,
    *,
    include_share: bool,
) -> pl.DataFrame:
    if include_share:
        match_data = match_data.with_columns(
            [
                pl.col(WINNERS).list.len().alias("winner_count"),
                pl.col(LOSERS).list.len().alias("loser_count"),
            ]
        ).with_columns(
            (
                pl.col(WEIGHT)
                / (pl.col("winner_count") * pl.col("loser_count"))
            ).alias(SHARE)
        )

    final_columns = [
        MATCH_ID,
        TOURNAMENT_ID,
        WINNERS,
        LOSERS,
        WEIGHT,
        "ts",
    ]
    if include_share:
        final_columns.extend(["winner_count", "loser_count", SHARE])
    return match_data.select(final_columns)


def resolve_match_participants(
    weighted_matches: WeightedMatches | pl.DataFrame,
    participants: pl.DataFrame,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    include_share: bool = True,
) -> ResolvedMatches:
    """Resolve winner and loser entity lists for each weighted match."""
    weighted_df = (
        weighted_matches.matches
        if isinstance(weighted_matches, WeightedMatches)
        else weighted_matches
    )

    if weighted_df.is_empty():
        return ResolvedMatches(
            weighted_matches=weighted_df,
            matches=_empty_resolved_matches(include_share),
        )

    roster_source = _resolve_roster_source(participants, rosters)
    appearance_source = _resolve_appearance_source(appearances, participants)
    match_data = _assign_match_rosters(
        weighted_df,
        roster_source,
        appearance_source,
    )

    if match_data.is_empty():
        return ResolvedMatches(
            weighted_matches=weighted_df,
            matches=_empty_resolved_matches(include_share),
        )

    return ResolvedMatches(
        weighted_matches=weighted_df,
        matches=_finalize_resolved_match_rows(
            match_data,
            include_share=include_share,
        ),
    )


def _build_exposure_pair_edges(matches_df: pl.DataFrame) -> pl.DataFrame:
    if matches_df.is_empty() or SHARE not in matches_df.columns:
        return _empty_pair_edges()

    return (
        matches_df.select([WINNERS, LOSERS, SHARE])
        .explode(WINNERS)
        .drop_nulls(WINNERS)
        .explode(LOSERS)
        .drop_nulls(LOSERS)
        .group_by([WINNERS, LOSERS])
        .agg(pl.col(SHARE).sum().alias(SHARE))
        .rename({WINNERS: "winner_id", LOSERS: "loser_id"})
    )


def prepare_graph_inputs(
    resolved_matches: ResolvedMatches | pl.DataFrame,
    *,
    active_entities: list[int] | None = None,
) -> PreparedGraphInputs:
    """Build reusable exposure graph artifacts from resolved matches."""
    matches_df = (
        resolved_matches.matches
        if isinstance(resolved_matches, ResolvedMatches)
        else resolved_matches
    )

    entity_metrics = aggregate_entity_metrics(matches_df)
    pair_edges = _build_exposure_pair_edges(matches_df)
    node_ids = merged_node_ids(
        matches_df,
        active_entities,
        aggregated_metrics=entity_metrics,
    )
    node_to_idx = {entity_id: idx for idx, entity_id in enumerate(node_ids)}
    index_mapping = build_index_mapping(node_to_idx)
    return PreparedGraphInputs(
        matches=matches_df,
        entity_metrics=entity_metrics,
        pair_edges=pair_edges,
        node_ids=node_ids,
        node_to_idx=node_to_idx,
        index_mapping=index_mapping,
    )


def _build_row_edge_dataframe(matches_df: pl.DataFrame) -> pl.DataFrame:
    if matches_df.is_empty():
        return _empty_row_edges()

    return (
        matches_df.select([WINNERS, LOSERS, WEIGHT])
        .explode(WINNERS)
        .drop_nulls(WINNERS)
        .explode(LOSERS)
        .drop_nulls(LOSERS)
        .group_by([LOSERS, WINNERS])
        .agg(pl.col(WEIGHT).sum().alias(WEIGHT_SUM))
        .rename({LOSERS: LOSER_USER_ID, WINNERS: WINNER_USER_ID})
    )


def prepare_row_edges(
    resolved_matches: ResolvedMatches | pl.DataFrame,
) -> PreparedRowEdges:
    """Build row-oriented loser->winner edges from resolved matches."""
    matches_df = (
        resolved_matches.matches
        if isinstance(resolved_matches, ResolvedMatches)
        else resolved_matches
    )
    edges = _build_row_edge_dataframe(matches_df)
    if edges.is_empty():
        return PreparedRowEdges(
            matches=matches_df,
            edges=edges,
            node_ids=[],
            node_to_idx={},
        )

    node_ids = sorted(
        set(edges[LOSER_USER_ID].unique().to_list())
        | set(edges[WINNER_USER_ID].unique().to_list())
    )
    node_to_idx = {entity_id: idx for idx, entity_id in enumerate(node_ids)}
    return PreparedRowEdges(
        matches=matches_df,
        edges=edges,
        node_ids=node_ids,
        node_to_idx=node_to_idx,
    )


def build_team_edge_dataframe(
    weighted_matches: WeightedMatches | pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate team-to-team edges from weighted matches."""
    weighted_df = (
        weighted_matches.matches
        if isinstance(weighted_matches, WeightedMatches)
        else weighted_matches
    )
    if weighted_df.is_empty():
        return pl.DataFrame([])

    return weighted_df.group_by(["loser_team_id", "winner_team_id"]).agg(
        pl.col(WEIGHT).sum().alias(WEIGHT_SUM)
    )


def participants_by_tournament(matches_df: pl.DataFrame) -> dict[int, list[int]]:
    """Return unique participating entities per tournament from resolved matches."""
    if matches_df.is_empty():
        return {}

    pieces = []
    for entity_column in (WINNERS, LOSERS):
        if entity_column not in matches_df.columns:
            continue
        pieces.append(
            matches_df.select(
                [TOURNAMENT_ID, pl.col(entity_column).alias("entity_id")]
            )
            .explode("entity_id")
            .drop_nulls("entity_id")
        )

    if not pieces:
        return {}

    participants = (
        pl.concat(pieces)
        .group_by(TOURNAMENT_ID)
        .agg(pl.col("entity_id").unique().sort())
    )
    return dict(
        zip(
            participants[TOURNAMENT_ID].to_list(),
            participants["entity_id"].to_list(),
        )
    )


def prepare_exposure_graph(
    matches: pl.DataFrame,
    participants: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    active_entities: list[int] | None = None,
) -> PreparedGraphInputs:
    """Convenience wrapper for the full exposure-graph prep pipeline."""
    weighted = prepare_weighted_matches(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )
    resolved = resolve_match_participants(
        weighted,
        participants,
        rosters=rosters,
        appearances=appearances,
        include_share=True,
    )
    return prepare_graph_inputs(
        resolved,
        active_entities=active_entities,
    )


def prepare_row_edge_inputs(
    matches: pl.DataFrame,
    participants: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    timestamp_column: str | None = None,
) -> PreparedRowEdges:
    """Convenience wrapper for the full row-edge preparation pipeline."""
    weighted = prepare_weighted_matches(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        timestamp_column=timestamp_column,
        legacy_timestamp_fill_after_influence=True,
    )
    resolved = resolve_match_participants(
        weighted,
        participants,
        rosters=rosters,
        appearances=appearances,
        include_share=False,
    )
    return prepare_row_edges(resolved)
