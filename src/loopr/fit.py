"""Dataset-fit diagnostics for LOOPR inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from loopr.core.connectivity import analyze_graph_connectivity
from loopr.core.preparation import (
    prepare_graph_inputs,
    prepare_weighted_matches,
    resolve_match_participants,
)
from loopr.schema import prepare_rank_inputs

PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class DatasetFitCheck:
    """One dataset-fit diagnostic result."""

    name: str
    status: str
    detail: str
    value: float | int | None = None


@dataclass(frozen=True)
class DatasetFitReport:
    """Summary of how well a dataset matches LOOPR's assumptions."""

    overall_status: str
    checks: tuple[DatasetFitCheck, ...]
    metrics: dict[str, Any]

    def to_dataframe(self) -> pl.DataFrame:
        """Render checks into a compact tabular report."""
        return pl.DataFrame(
            {
                "check": [check.name for check in self.checks],
                "status": [check.status for check in self.checks],
                "value": pl.Series(
                    "value",
                    [
                        None
                        if check.value is None
                        else float(check.value)
                        for check in self.checks
                    ],
                    dtype=pl.Float64,
                    strict=False,
                ),
                "detail": [check.detail for check in self.checks],
            }
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


def _competitive_matches(matches: pl.DataFrame) -> pl.DataFrame:
    filter_expression = (
        pl.col("winner_team_id").is_not_null()
        & pl.col("loser_team_id").is_not_null()
    )
    if "is_bye" in matches.columns:
        filter_expression = filter_expression & ~pl.col("is_bye").fill_null(False)
    return matches.filter(filter_expression)


def _roster_sizes(participants: pl.DataFrame) -> pl.DataFrame:
    if participants.is_empty():
        return pl.DataFrame(
            schema={
                "tournament_id": pl.Int64,
                "team_id": pl.Int64,
                "roster_size": pl.UInt32,
            }
        )
    return participants.group_by(["tournament_id", "team_id"]).agg(
        pl.len().alias("roster_size")
    )


def _multi_team_entities(participants: pl.DataFrame) -> pl.DataFrame:
    if participants.is_empty():
        return pl.DataFrame(
            schema={
                "tournament_id": pl.Int64,
                "user_id": pl.Int64,
                "team_count": pl.UInt32,
            }
        )
    return (
        participants.group_by(["tournament_id", "user_id"])
        .agg(pl.col("team_id").n_unique().alias("team_count"))
        .filter(pl.col("team_count") > 1)
    )


def _match_team_coverage(
    matches: pl.DataFrame,
    grouped_appearances: pl.DataFrame,
) -> float | None:
    match_teams = pl.concat(
        [
            matches.select(
                [
                    "tournament_id",
                    "match_id",
                    pl.col("winner_team_id").alias("team_id"),
                ]
            ),
            matches.select(
                [
                    "tournament_id",
                    "match_id",
                    pl.col("loser_team_id").alias("team_id"),
                ]
            ),
        ]
    ).unique()
    if match_teams.is_empty():
        return None

    covered_match_teams = match_teams.join(
        grouped_appearances.select(["tournament_id", "match_id", "team_id"]).unique(),
        on=["tournament_id", "match_id", "team_id"],
        how="inner",
    )
    return covered_match_teams.height / match_teams.height


def assess_dataset_fit(
    matches: pl.DataFrame,
    participants: pl.DataFrame,
    appearances: pl.DataFrame | None = None,
) -> DatasetFitReport:
    """Assess whether a dataset matches LOOPR's main modeling assumptions."""
    prepared = prepare_rank_inputs(matches, participants, appearances)
    competitive_matches = _competitive_matches(prepared.matches)
    roster_sizes = _roster_sizes(prepared.participants)
    multi_team_entities = _multi_team_entities(prepared.participants)

    metrics: dict[str, Any] = {
        "total_matches": prepared.matches.height,
        "competitive_matches": competitive_matches.height,
        "participant_rows": prepared.participants.height,
        "appearance_rows": (
            None if prepared.appearances is None else prepared.appearances.height
        ),
        "average_roster_size": (
            None
            if roster_sizes.is_empty()
            else float(roster_sizes["roster_size"].mean())
        ),
        "max_roster_size": (
            None
            if roster_sizes.is_empty()
            else int(roster_sizes["roster_size"].max())
        ),
        "multi_team_entity_pairs": multi_team_entities.height,
        "resolvable_match_count": None,
        "resolvable_match_fraction": None,
        "appearance_match_coverage": None,
        "entity_graph_component_count": None,
        "entity_graph_largest_component_size": None,
        "entity_graph_largest_component_fraction": None,
        "entity_graph_largest_component_share_fraction": None,
        "entity_graph_largest_component_weight_fraction": None,
        "entity_graph_disconnected_share_fraction": None,
    }

    checks: list[DatasetFitCheck] = []

    if competitive_matches.is_empty():
        checks.append(
            DatasetFitCheck(
                "competitive_matches",
                FAIL,
                "No competitive winner/loser matches remain after excluding byes and null team IDs.",
                0,
            )
        )
    else:
        checks.append(
            DatasetFitCheck(
                "competitive_matches",
                PASS,
                "Competitive winner/loser matches are present.",
                competitive_matches.height,
            )
        )

    winner_rosters = roster_sizes.rename(
        {"team_id": "winner_team_id", "roster_size": "winner_roster_size"}
    )
    loser_rosters = roster_sizes.rename(
        {"team_id": "loser_team_id", "roster_size": "loser_roster_size"}
    )
    roster_coverage = competitive_matches.join(
        winner_rosters,
        on=["tournament_id", "winner_team_id"],
        how="left",
    ).join(
        loser_rosters,
        on=["tournament_id", "loser_team_id"],
        how="left",
    )
    missing_rosters = roster_coverage.filter(
        pl.col("winner_roster_size").is_null()
        | pl.col("loser_roster_size").is_null()
    )
    resolvable_matches = roster_coverage.filter(
        pl.col("winner_roster_size").is_not_null()
        & pl.col("loser_roster_size").is_not_null()
    ).select(competitive_matches.columns)
    resolvable_fraction = (
        None
        if competitive_matches.is_empty()
        else resolvable_matches.height / competitive_matches.height
    )
    metrics["resolvable_match_count"] = resolvable_matches.height
    metrics["resolvable_match_fraction"] = resolvable_fraction

    if competitive_matches.is_empty():
        checks.append(
            DatasetFitCheck(
                "team_rosters",
                PASS,
                "No competitive matches were available for roster coverage checks.",
                None,
            )
        )
    elif missing_rosters.is_empty():
        checks.append(
            DatasetFitCheck(
                "team_rosters",
                PASS,
                "Every competitive match team has a participant roster.",
                resolvable_fraction,
            )
        )
    else:
        sample = _sample_rows(missing_rosters, ["match_id", "tournament_id"])
        checks.append(
            DatasetFitCheck(
                "team_rosters",
                FAIL,
                "Some competitive matches cannot be attributed back to complete participant rosters, "
                "so LOOPR cannot rank the dataset as-is without dropping those matches. "
                f"Sample match_id/event_id pairs: {sample}",
                resolvable_fraction,
            )
        )

    grouped_appearances_for_connectivity: pl.DataFrame | None = None

    if prepared.appearances is None:
        if multi_team_entities.height > 0:
            sample = _sample_rows(
                multi_team_entities,
                ["tournament_id", "user_id"],
            )
            checks.append(
                DatasetFitCheck(
                    "multi_team_membership_without_appearances",
                    FAIL,
                    "Some entities belong to multiple groups within the same event "
                    "but no appearances table was provided. "
                    f"Sample event_id/entity_id pairs: {sample}",
                    multi_team_entities.height,
                )
            )
        else:
            checks.append(
                DatasetFitCheck(
                    "multi_team_membership_without_appearances",
                    PASS,
                    "No same-event multi-team entity memberships were detected without appearances.",
                    0,
                )
            )

        max_roster_size = metrics["max_roster_size"]
        if max_roster_size is not None and max_roster_size > 2:
            checks.append(
                DatasetFitCheck(
                    "roster_fallback_risk",
                    WARN,
                    "No appearances table was provided and team rosters are larger than two entities, "
                    "so roster fallback may over-attribute match effects.",
                    max_roster_size,
                )
            )
        else:
            checks.append(
                DatasetFitCheck(
                    "roster_fallback_risk",
                    PASS,
                    "Roster fallback risk looks low for the observed team sizes.",
                    max_roster_size,
                )
            )
    else:
        appearance_rows = prepared.appearances
        grouped_appearances = appearance_rows

        if "team_id" not in grouped_appearances.columns:
            if multi_team_entities.height > 0:
                sample = _sample_rows(
                    multi_team_entities,
                    ["tournament_id", "user_id"],
                )
                checks.append(
                    DatasetFitCheck(
                        "appearance_team_inference",
                        FAIL,
                        "Appearances is missing group_id for entities assigned to multiple groups "
                        f"within the same event. Sample event_id/entity_id pairs: {sample}",
                        multi_team_entities.height,
                    )
                )
            else:
                checks.append(
                    DatasetFitCheck(
                        "appearance_team_inference",
                        PASS,
                        "Appearances can infer group_id from participants without same-event ambiguity.",
                        0,
                    )
                )

            team_lookup = prepared.participants.select(
                ["tournament_id", "user_id", "team_id"]
            ).unique(subset=["tournament_id", "user_id"], keep="any")
            grouped_appearances = grouped_appearances.join(
                team_lookup,
                on=["tournament_id", "user_id"],
                how="left",
            )
        else:
            checks.append(
                DatasetFitCheck(
                    "appearance_team_inference",
                    PASS,
                    "Appearances already includes group_id for each row.",
                    appearance_rows.height,
                )
            )

        unresolved_appearances = grouped_appearances.filter(pl.col("team_id").is_null())
        if unresolved_appearances.height > 0:
            sample = _sample_rows(
                unresolved_appearances,
                ["tournament_id", "match_id", "user_id"],
            )
            checks.append(
                DatasetFitCheck(
                    "appearance_participant_membership",
                    FAIL,
                    "Some appearance rows could not be matched back to participants. "
                    f"Sample event_id/match_id/entity_id rows: {sample}",
                    unresolved_appearances.height,
                )
            )
            grouped_appearances = grouped_appearances.drop_nulls("team_id")
        else:
            invalid_membership = grouped_appearances.join(
                prepared.participants.select(
                    ["tournament_id", "team_id", "user_id"]
                ).unique(),
                on=["tournament_id", "team_id", "user_id"],
                how="anti",
            )
            if invalid_membership.height > 0:
                sample = _sample_rows(
                    invalid_membership,
                    ["tournament_id", "match_id", "user_id"],
                )
                checks.append(
                    DatasetFitCheck(
                        "appearance_participant_membership",
                        FAIL,
                        "Some appearance rows reference entity/team pairs not present in participants. "
                        f"Sample event_id/match_id/entity_id rows: {sample}",
                        invalid_membership.height,
                    )
                )
                grouped_appearances = grouped_appearances.join(
                    prepared.participants.select(
                        ["tournament_id", "team_id", "user_id"]
                    ).unique(),
                    on=["tournament_id", "team_id", "user_id"],
                    how="semi",
                )
            else:
                checks.append(
                    DatasetFitCheck(
                        "appearance_participant_membership",
                        PASS,
                        "Appearance rows align with participant rosters.",
                        appearance_rows.height,
                    )
                )

        if grouped_appearances.is_empty():
            coverage = 0.0 if not competitive_matches.is_empty() else None
        else:
            grouped_appearances = grouped_appearances.group_by(
                ["tournament_id", "match_id", "team_id"]
            ).agg(pl.col("user_id"))
            grouped_appearances_for_connectivity = grouped_appearances
            coverage = _match_team_coverage(competitive_matches, grouped_appearances)
        metrics["appearance_match_coverage"] = coverage

        if coverage is None:
            checks.append(
                DatasetFitCheck(
                    "appearance_match_coverage",
                    PASS,
                    "No competitive matches were available for appearance coverage checks.",
                    None,
                )
            )
        elif coverage >= 0.8:
            checks.append(
                DatasetFitCheck(
                    "appearance_match_coverage",
                    PASS,
                    "Appearance coverage is high enough to trust match-level participation attribution.",
                    coverage,
                )
            )
        elif coverage >= 0.5:
            checks.append(
                DatasetFitCheck(
                    "appearance_match_coverage",
                    WARN,
                    "Appearance coverage is partial; LOOPR can use it, but many match sides still rely on roster fallback.",
                    coverage,
                )
            )
        else:
            checks.append(
                DatasetFitCheck(
                    "appearance_match_coverage",
                    WARN,
                    "Appearance coverage is sparse. LOOPR can still fall back to event rosters, "
                    "but the dataset likely has missing or incomplete match-level participation data.",
                    coverage,
                )
            )

    failed_checks = {check.name for check in checks if check.status == FAIL}
    if "competitive_matches" not in failed_checks:
        weighted_matches = prepare_weighted_matches(
            resolvable_matches,
            tournament_influence={},
            now_timestamp=0.0,
            decay_rate=0.0,
            beta=0.0,
        )
        resolved_matches = resolve_match_participants(
            weighted_matches,
            prepared.participants,
            appearances=grouped_appearances_for_connectivity,
            include_share=True,
        )
        connectivity_graph = prepare_graph_inputs(resolved_matches)
        connectivity_report = analyze_graph_connectivity(connectivity_graph)
        graph_metrics = {
            "entity_graph_component_count": connectivity_report.component_count,
            "entity_graph_largest_component_size": (
                connectivity_report.largest_component_entity_count
            ),
            "entity_graph_largest_component_fraction": (
                connectivity_report.largest_component_node_fraction
            ),
            "entity_graph_largest_component_share_fraction": (
                connectivity_report.largest_component_share_fraction
            ),
            "entity_graph_largest_component_weight_fraction": (
                connectivity_report.largest_component_weight_fraction
            ),
            "entity_graph_disconnected_share_fraction": (
                connectivity_report.disconnected_share_fraction
            ),
        }
        metrics.update(graph_metrics)

        component_count = connectivity_report.component_count
        disconnected_share_fraction = (
            connectivity_report.disconnected_share_fraction
        )
        if component_count == 0:
            checks.append(
                DatasetFitCheck(
                    "entity_graph_connectivity",
                    FAIL,
                    "No entity comparison graph could be formed from the resolved matches.",
                    0,
                )
            )
        elif component_count == 1:
            checks.append(
                DatasetFitCheck(
                    "entity_graph_connectivity",
                    PASS,
                    "Resolved entity comparison graph is fully connected.",
                    1,
                )
            )
        elif (
            disconnected_share_fraction is not None
            and disconnected_share_fraction <= 0.005
        ):
            checks.append(
                DatasetFitCheck(
                    "entity_graph_connectivity",
                    WARN,
                    "Disconnected comparison islands remain, but they sit outside a tiny share tail. "
                    "Keeping the largest component would retain "
                    f"{(connectivity_report.largest_component_share_fraction or 0.0):.1%} of share mass.",
                    connectivity_report.largest_component_share_fraction,
                )
            )
        else:
            checks.append(
                DatasetFitCheck(
                    "entity_graph_connectivity",
                    FAIL,
                    "The resolved entity comparison graph is disconnected in a material way. "
                    "Keeping the largest component would retain "
                    f"{(connectivity_report.largest_component_share_fraction or 0.0):.1%} of share mass.",
                    connectivity_report.largest_component_share_fraction,
                )
            )

    statuses = {check.status for check in checks}
    if FAIL in statuses:
        overall_status = "poor_fit"
    elif WARN in statuses:
        overall_status = "usable_with_caution"
    else:
        overall_status = "good_fit"

    return DatasetFitReport(
        overall_status=overall_status,
        checks=tuple(checks),
        metrics=metrics,
    )
