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
    PLACEMENT,
    SECONDS_PER_DAY,
    SHARE,
    TOURNAMENT_ID,
    WEIGHT,
    WEIGHT_SUM,
    WINNERS,
    WINNER_USER_ID,
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


def _validate_result_mode(result_mode: str) -> str:
    if result_mode not in {"teams", "positional"}:
        raise ValueError(
            "result_mode must be one of: teams, positional"
        )
    return result_mode


def _validate_positional_weight_mode(positional_weight_mode: str) -> str:
    if positional_weight_mode not in {
        "pairwise_full",
        "pairwise_average",
    }:
        raise ValueError(
            "positional_weight_mode must be one of: "
            "pairwise_full, pairwise_average"
        )
    return positional_weight_mode


def _attach_weight_columns(
    match_data: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float,
    *,
    timestamp_column: str | None = None,
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
    )
    return WeightedMatches(match_data)


def prepare_weighted_positional_results(
    results: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float,
    *,
    timestamp_column: str | None = None,
) -> WeightedMatches:
    """Filter positional result rows and attach timestamps and weights."""
    identity_columns = [
        column
        for column in ("user_id", "team_id")
        if column in results.columns
    ]
    if not identity_columns:
        raise ValueError(
            "positional results must include either user_id or team_id"
        )

    filter_expression = pl.any_horizontal(
        [pl.col(column).is_not_null() for column in identity_columns]
    )
    if "is_bye" in results.columns:
        filter_expression = filter_expression & ~pl.col("is_bye").fill_null(
            False
        )

    result_rows = results.filter(filter_expression)
    if result_rows.is_empty():
        return WeightedMatches(result_rows)

    counts = (
        result_rows.group_by([TOURNAMENT_ID, MATCH_ID])
        .len()
        .rename({"len": "finisher_count"})
    )
    invalid = counts.filter(pl.col("finisher_count") < 2)
    if invalid.height:
        raise ValueError(
            "positional results must contain at least two finishers per result set "
            "after filtering"
        )

    timestamp_counts = (
        result_rows.group_by([TOURNAMENT_ID, MATCH_ID])
        .agg(pl.col("placement").count().alias("row_count"))
        .join(
            result_rows.group_by([TOURNAMENT_ID, MATCH_ID]).agg(
                pl.col("last_game_finished_at")
                .cast(pl.Float64, strict=False)
                .drop_nulls()
                .n_unique()
                .alias("finished_at_unique")
                if "last_game_finished_at" in result_rows.columns
                else pl.lit(0).alias("finished_at_unique"),
                pl.col("match_created_at")
                .cast(pl.Float64, strict=False)
                .drop_nulls()
                .n_unique()
                .alias("created_at_unique")
                if "match_created_at" in result_rows.columns
                else pl.lit(0).alias("created_at_unique"),
            ),
            on=[TOURNAMENT_ID, MATCH_ID],
            how="left",
        )
    )
    inconsistent = timestamp_counts.filter(
        (pl.col("finished_at_unique") > 1)
        | (pl.col("created_at_unique") > 1)
    )
    if inconsistent.height:
        raise ValueError(
            "positional results must use a consistent timestamp per result set"
        )

    result_rows = _attach_weight_columns(
        result_rows,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        timestamp_column=timestamp_column,
    )
    return WeightedMatches(result_rows)


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
        team_lookup = participants.select(
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

    return (
        match_data.with_columns([winners_expr, losers_expr])
        .filter(
            pl.col(WINNERS).is_not_null()
            & pl.col(LOSERS).is_not_null()
            & (pl.col(WINNERS).list.len() > 0)
            & (pl.col(LOSERS).list.len() > 0)
        )
    )


def _resolve_positional_identity_column(results_df: pl.DataFrame) -> str:
    identity_columns = [
        column
        for column in ("user_id", "team_id")
        if column in results_df.columns
        and results_df.select(pl.col(column).is_not_null().sum()).item() > 0
    ]
    if len(identity_columns) != 1:
        raise ValueError(
            "positional results must use exactly one of user_id or team_id"
        )
    return identity_columns[0]


def _assign_positional_members(
    weighted_results: pl.DataFrame,
    participants: pl.DataFrame | None,
    rosters: pl.DataFrame | None,
    appearances: pl.DataFrame | None,
) -> pl.DataFrame:
    identity_column = _resolve_positional_identity_column(weighted_results)
    if identity_column == "user_id":
        return weighted_results.select(
            [
                MATCH_ID,
                TOURNAMENT_ID,
                PLACEMENT,
                pl.col("user_id").alias("finisher_id"),
                pl.concat_list(pl.col("user_id")).alias("members"),
                WEIGHT,
                "ts",
            ]
        )

    if participants is None:
        raise ValueError(
            "participants are required for positional results using group_id"
        )

    roster_source = _resolve_roster_source(participants, rosters)
    appearance_source = _resolve_appearance_source(appearances, participants)
    grouped = weighted_results.join(
        roster_source.rename({"user_id": "roster"}),
        on=[TOURNAMENT_ID, "team_id"],
        how="left",
    )

    if appearance_source is not None:
        grouped = grouped.join(
            appearance_source.rename({"user_id": "appearance"}),
            on=[TOURNAMENT_ID, MATCH_ID, "team_id"],
            how="left",
        )
        members_expr = (
            pl.when(pl.col("appearance").is_not_null())
            .then(pl.col("appearance"))
            .otherwise(pl.col("roster"))
            .alias("members")
        )
    else:
        members_expr = pl.col("roster").alias("members")

    return (
        grouped.with_columns(
            [pl.col("team_id").alias("finisher_id"), members_expr]
        )
        .select(
            [
                MATCH_ID,
                TOURNAMENT_ID,
                PLACEMENT,
                "finisher_id",
                "members",
                WEIGHT,
                "ts",
            ]
        )
        .filter(
            pl.col("members").is_not_null()
            & (pl.col("members").list.len() > 0)
        )
    )


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


def _expand_positional_match_rows(
    positional_rows: pl.DataFrame,
    *,
    positional_weight_mode: str,
) -> pl.DataFrame:
    expanded_rows: list[dict[str, Any]] = []
    partitions = positional_rows.sort(
        [TOURNAMENT_ID, MATCH_ID, PLACEMENT, "finisher_id"]
    ).partition_by([TOURNAMENT_ID, MATCH_ID], maintain_order=True)

    for result_set in partitions:
        if result_set.height < 2:
            raise ValueError(
                "positional results must contain at least two resolved finishers per result set"
            )

        placements = result_set[PLACEMENT].to_list()
        finisher_ids = result_set["finisher_id"].to_list()
        members = result_set["members"].to_list()
        tournament_id = int(result_set[TOURNAMENT_ID][0])
        match_id = int(result_set[MATCH_ID][0])
        ts = float(result_set["ts"][0])
        base_weight = float(result_set[WEIGHT][0])

        comparisons: list[tuple[int, int]] = []
        for i in range(result_set.height):
            for j in range(i + 1, result_set.height):
                if placements[i] < placements[j]:
                    comparisons.append((i, j))
                elif placements[i] == placements[j]:
                    comparisons.append((i, j))
                    comparisons.append((j, i))

        if not comparisons:
            raise ValueError(
                "positional results must imply at least one comparison per result set"
            )

        if positional_weight_mode == "pairwise_average":
            resolved_weight = base_weight / len(comparisons)
        else:
            resolved_weight = base_weight

        for winner_idx, loser_idx in comparisons:
            if finisher_ids[winner_idx] == finisher_ids[loser_idx]:
                continue
            expanded_rows.append(
                {
                    MATCH_ID: match_id,
                    TOURNAMENT_ID: tournament_id,
                    WINNERS: members[winner_idx],
                    LOSERS: members[loser_idx],
                    WEIGHT: resolved_weight,
                    "ts": ts,
                }
            )

    if not expanded_rows:
        return _empty_resolved_matches(include_share=False)

    return pl.DataFrame(expanded_rows)


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


def resolve_positional_results(
    weighted_results: WeightedMatches | pl.DataFrame,
    participants: pl.DataFrame | None = None,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    include_share: bool = True,
    positional_weight_mode: str = "pairwise_full",
) -> ResolvedMatches:
    """Resolve ordered finishes into implied loser->winner rows."""
    _validate_positional_weight_mode(positional_weight_mode)

    weighted_df = (
        weighted_results.matches
        if isinstance(weighted_results, WeightedMatches)
        else weighted_results
    )
    if weighted_df.is_empty():
        return ResolvedMatches(
            weighted_matches=weighted_df,
            matches=_empty_resolved_matches(include_share),
        )

    positional_rows = _assign_positional_members(
        weighted_df,
        participants,
        rosters,
        appearances,
    )
    if positional_rows.is_empty():
        return ResolvedMatches(
            weighted_matches=weighted_df,
            matches=_empty_resolved_matches(include_share),
        )

    counts = (
        positional_rows.group_by([TOURNAMENT_ID, MATCH_ID])
        .len()
        .rename({"len": "finisher_count"})
    )
    invalid = counts.filter(pl.col("finisher_count") < 2)
    if invalid.height:
        raise ValueError(
            "positional results must contain at least two resolved finishers per result set"
        )

    expanded = _expand_positional_match_rows(
        positional_rows,
        positional_weight_mode=positional_weight_mode,
    )
    if expanded.is_empty():
        return ResolvedMatches(
            weighted_matches=weighted_df,
            matches=_empty_resolved_matches(include_share),
        )

    return ResolvedMatches(
        weighted_matches=weighted_df,
        matches=_finalize_resolved_match_rows(
            expanded,
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
    participants: pl.DataFrame | None,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    active_entities: list[int] | None = None,
    result_mode: str = "teams",
    positional_weight_mode: str = "pairwise_full",
) -> PreparedGraphInputs:
    """Convenience wrapper for the full exposure-graph prep pipeline."""
    result_mode = _validate_result_mode(result_mode)
    if result_mode == "positional":
        weighted = prepare_weighted_positional_results(
            matches,
            tournament_influence,
            now_timestamp,
            decay_rate,
            beta,
        )
        resolved = resolve_positional_results(
            weighted,
            participants,
            rosters=rosters,
            appearances=appearances,
            include_share=True,
            positional_weight_mode=positional_weight_mode,
        )
    else:
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
    participants: pl.DataFrame | None,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
    *,
    rosters: pl.DataFrame | None = None,
    appearances: pl.DataFrame | None = None,
    timestamp_column: str | None = None,
    result_mode: str = "teams",
    positional_weight_mode: str = "pairwise_full",
) -> PreparedRowEdges:
    """Convenience wrapper for the full row-edge preparation pipeline."""
    result_mode = _validate_result_mode(result_mode)
    if result_mode == "positional":
        weighted = prepare_weighted_positional_results(
            matches,
            tournament_influence,
            now_timestamp,
            decay_rate,
            beta,
            timestamp_column=timestamp_column,
        )
        resolved = resolve_positional_results(
            weighted,
            participants,
            rosters=rosters,
            appearances=appearances,
            include_share=False,
            positional_weight_mode=positional_weight_mode,
        )
    else:
        weighted = prepare_weighted_matches(
            matches,
            tournament_influence,
            now_timestamp,
            decay_rate,
            beta,
            timestamp_column=timestamp_column,
        )
        resolved = resolve_match_participants(
            weighted,
            participants,
            rosters=rosters,
            appearances=appearances,
            include_share=False,
        )
    return prepare_row_edges(resolved)
