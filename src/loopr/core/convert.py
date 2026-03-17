"""Data conversion utilities for ranking algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from loopr.core.preparation import (
    PreparedGraphInputs,
    prepare_exposure_graph,
    resolve_match_participants,
    prepare_weighted_matches,
)
from loopr.schema import (
    prepare_matches_frame,
    prepare_participants_frame,
    prepare_rank_inputs,
)

if TYPE_CHECKING:
    from typing import Any


PreparedExposureMatches = PreparedGraphInputs


def _prepare_exposure_matches_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
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
    """Build reusable exposure intermediates from already-normalized inputs."""
    del streaming  # retained for API compatibility

    if not include_share:
        raise ValueError(
            "_prepare_exposure_matches_normalized requires include_share=True"
        )

    return prepare_exposure_graph(
        matches,
        players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        rosters=rosters,
        appearances=appearances,
    )


def convert_matches_dataframe(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
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
    """Build a compact matches table with winners/losers lists and weights."""
    prepared = prepare_rank_inputs(matches, players, appearances)
    prepared_matches = prepared.matches
    prepared_players = prepared.participants
    prepared_appearances = prepared.appearances

    rosters = (
        prepare_participants_frame(rosters) if rosters is not None else None
    )
    return _convert_matches_dataframe_normalized(
        prepared_matches,
        prepared_players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        rosters=rosters,
        appearances=prepared_appearances,
        include_share=include_share,
        streaming=streaming,
    )


def _convert_matches_dataframe_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
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
    del streaming  # retained for API compatibility

    if include_share:
        prepared = prepare_exposure_graph(
            matches,
            players,
            tournament_influence,
            now_timestamp,
            decay_rate,
            beta,
            rosters=rosters,
            appearances=appearances,
        )
        return prepared.matches

    weighted = prepare_weighted_matches(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )
    resolved = resolve_match_participants(
        weighted,
        players,
        rosters=rosters,
        appearances=appearances,
        include_share=False,
    )
    return resolved.matches


def convert_matches_format(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert matches to the historical list-of-dicts format."""
    prepared = prepare_rank_inputs(matches, players)
    prepared_matches = prepared.matches
    prepared_players = prepared.participants
    return _convert_matches_format_normalized(
        prepared_matches,
        prepared_players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )


def _convert_matches_format_normalized(
    matches: pl.DataFrame,
    players: pl.DataFrame | None,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Internal fallback conversion path for already-normalized inputs."""
    converted = _convert_matches_dataframe_normalized(
        matches,
        players,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
        include_share=False,
    )
    return [
        {
            "winners": row["winners"],
            "losers": row["losers"],
            "weight": row["weight"],
            "tournament_id": row["tournament_id"],
            "match_id": row["match_id"],
            "timestamp": row["ts"],
        }
        for row in converted.iter_rows(named=True)
    ]


def convert_team_matches(
    matches: pl.DataFrame,
    tournament_influence: dict[int, float],
    now_timestamp: float,
    decay_rate: float,
    beta: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert team matches to the historical list-of-dicts format."""
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
    weighted = prepare_weighted_matches(
        matches,
        tournament_influence,
        now_timestamp,
        decay_rate,
        beta,
    )
    match_data = weighted.matches
    return [
        {
            "winners": [row["winner_team_id"]],
            "losers": [row["loser_team_id"]],
            "weight": row["weight"],
            "tournament_id": row["tournament_id"],
            "match_id": row["match_id"],
            "timestamp": row["ts"],
        }
        for row in match_data.iter_rows(named=True)
    ]


def factorize_ids(
    node_ids: list[Any],
) -> tuple[list[Any], dict[Any, int]]:
    """Convert a list of IDs into a stable unique list plus index map."""
    unique_ids = list(dict.fromkeys(node_ids))
    id_to_index = {node_id: index for index, node_id in enumerate(unique_ids)}
    return unique_ids, id_to_index


def build_node_mapping(
    matches_dataframe: pl.DataFrame,
    winner_column: str = "winners",
    loser_column: str = "losers",
) -> tuple[list[Any], dict[Any, int]]:
    """Build node ID to index mapping from a matches DataFrame."""
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
