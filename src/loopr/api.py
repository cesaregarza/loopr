"""Small stable public helpers for one-shot ranking workflows."""

from __future__ import annotations

import warnings

import polars as pl

from loopr.algorithms import LOOPREngine
from loopr.core import Clock, ExposureLogOddsConfig


def rank_entities(
    matches: pl.DataFrame,
    participants: pl.DataFrame,
    *,
    appearances: pl.DataFrame | None = None,
    config: ExposureLogOddsConfig | None = None,
    now_ts: float | None = None,
    clock: Clock | None = None,
    tournament_influence: dict[int, float] | None = None,
    component_policy: str = "keep_largest",
) -> pl.DataFrame:
    """Rank entities from neutral-schema input tables.

    This is the recommended one-shot public entrypoint when you already have
    `matches`, `participants`, and optional `appearances` tables. It constructs
    a default `LOOPREngine`, runs the ranking workflow, and returns a rankings
    dataframe keyed by `entity_id`.

    Use `LOOPREngine` directly when you need engine state, diagnostics, or
    leave-one-match-out analysis after ranking.
    """

    engine = LOOPREngine(config=config, now_ts=now_ts, clock=clock)
    rankings = engine.rank_entities(
        matches,
        participants,
        tournament_influence,
        appearances=appearances,
        component_policy=component_policy,
    )
    report = engine.last_connectivity_report or {}
    warning_message = report.get("warning_message")
    if warning_message:
        warnings.warn(warning_message, stacklevel=2)
    return rankings
