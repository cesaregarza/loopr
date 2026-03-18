#!/usr/bin/env python3
"""Compare `loopr` outputs against `sendouq_analysis` on shared tournament data.

This harness loads the legacy Sendou ranking implementation directly from a
local checkout, parses the same tournament JSON once, then compares:

- converted `winners` / `losers` match tables
- final ranking outputs
- tournament influence values when the selected engine computes them

Examples:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_sendouq_fidelity.py \
  --input-json /root/dev/sendouq_analysis/sample_tournament.json \
  --profile simple

PYTHONPATH=src .venv/bin/python scripts/compare_sendouq_fidelity.py \
  --input-json /root/dev/sendouq_analysis/tournament_data.json \
  --engine both \
  --profile default

PYTHONPATH=src .venv/bin/python scripts/compare_sendouq_fidelity.py \
  --dataset-source window \
  --window-dir /root/dev/sendouq_analysis/data/embeddings_window_540d_all \
  --window-filter ranked_quality \
  --engine both \
  --profile default
```
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import resource
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import polars as pl
from scipy.stats import spearmanr

from loopr.algorithms import (
    ExposureLogOddsEngine as LooprExposureEngine,
    TickTockEngine as LooprTickTockEngine,
)
from loopr.core import (
    DecayConfig as LooprDecayConfig,
    EngineConfig as LooprEngineConfig,
    ExposureLogOddsConfig as LooprExposureConfig,
    PageRankConfig as LooprPageRankConfig,
    TickTockConfig as LooprTickTockConfig,
    convert_matches_dataframe as loopr_convert_matches_dataframe,
)
from loopr.core import build_exposure_triplets as loopr_build_exposure_triplets


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak) / (1024.0 * 1024.0)
    return float(peak) / 1024.0


def _install_sendou_namespace(sendou_root: Path) -> None:
    rankings_root = sendou_root / "src" / "rankings"
    if not rankings_root.exists():
        raise FileNotFoundError(
            f"Could not find sendou rankings sources at {rankings_root}"
        )

    package = types.ModuleType("rankings")
    package.__path__ = [str(rankings_root)]
    sys.modules["rankings"] = package


def _load_sendou_modules(sendou_root: Path) -> dict[str, Any]:
    _install_sendou_namespace(sendou_root)

    return {
        "config": importlib.import_module("rankings.core.config"),
        "parser": importlib.import_module("rankings.core.parser"),
        "convert": importlib.import_module("rankings.core.convert"),
        "edges": importlib.import_module("rankings.core.edges"),
        "exposure": importlib.import_module(
            "rankings.algorithms.exposure_log_odds"
        ),
        "tick_tock": importlib.import_module("rankings.algorithms.tick_tock"),
    }


def _load_tournament_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported payload shape in {path}")


def _window_tournament_filter(
    tournaments: pl.DataFrame,
    *,
    filter_mode: str,
) -> pl.DataFrame:
    if filter_mode == "none":
        return tournaments
    if filter_mode == "ranked_only":
        return tournaments.filter(pl.col("is_ranked") == True)
    if filter_mode == "ranked_quality":
        return tournaments.filter(
            (pl.col("is_ranked") == True)
            & (pl.col("team_count") >= 4)
            & (pl.col("participated_users_count") >= 16)
            & (pl.col("match_count") >= 10)
            & (~pl.col("name").str.contains("(?i)test|debug|demo|practice"))
        )
    raise ValueError(f"Unsupported window filter mode: {filter_mode}")


def _load_window_dataset(
    *,
    window_dir: Path,
    filter_mode: str,
    limit_tournaments: int | None,
) -> dict[str, Any]:
    matches_path = window_dir / "matches.parquet"
    players_path = window_dir / "players.parquet"
    appearances_path = window_dir / "appearance_teams.parquet"
    tournaments_path = window_dir / "tournaments.parquet"

    for path in (
        matches_path,
        players_path,
        appearances_path,
        tournaments_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required window artifact: {path}")

    tournaments = pl.read_parquet(tournaments_path)
    tournaments = _window_tournament_filter(tournaments, filter_mode=filter_mode)
    tournaments = tournaments.sort("start_time_ms")
    if limit_tournaments is not None:
        tournaments = tournaments.head(limit_tournaments)

    tournament_ids = tournaments.select("tournament_id")
    matches = pl.read_parquet(matches_path).join(
        tournament_ids,
        on="tournament_id",
        how="inner",
    )
    appearances = pl.read_parquet(appearances_path).join(
        tournament_ids,
        on="tournament_id",
        how="inner",
    )

    player_names = (
        pl.read_parquet(players_path)
        .select(
            [
                pl.col("player_id").cast(pl.Int64).alias("user_id"),
                pl.col("display_name").alias("username"),
            ]
        )
        .unique(subset=["user_id"])
    )

    players = (
        appearances.select(["tournament_id", "team_id", "user_id"])
        .unique(subset=["tournament_id", "team_id", "user_id"])
        .join(player_names, on="user_id", how="left")
    )

    return {
        "tournaments_df": tournaments,
        "matches_sendou": matches,
        "players_sendou": players,
        "appearances_sendou": appearances,
    }


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_appearances_from_players_payload(
    tournament_id: int,
    payload: dict[str, Any] | list[Any],
) -> list[dict[str, int | None]]:
    """Best-effort extraction of per-match appearances from Sendou payloads."""
    rows: list[dict[str, int | None]] = []

    def _emit(match_id: Any, team_id: Any, players_node: Any) -> None:
        if players_node is None:
            return
        if isinstance(players_node, list):
            for player in players_node:
                if isinstance(player, dict):
                    user_id = _coerce_int(
                        player.get("userId") or player.get("id")
                    )
                else:
                    user_id = _coerce_int(player)
                if user_id is not None and match_id is not None:
                    rows.append(
                        {
                            "tournament_id": int(tournament_id),
                            "match_id": _coerce_int(match_id),
                            "team_id": _coerce_int(team_id),
                            "user_id": user_id,
                        }
                    )
        elif isinstance(players_node, dict):
            nested = players_node.get("userIds") or players_node.get("players")
            if isinstance(nested, list):
                _emit(match_id, team_id, nested)

    def _parse_match_obj(match_obj: dict[str, Any]) -> None:
        match_id = match_obj.get("matchId") or match_obj.get("id")
        teams = match_obj.get("teams") or []
        if isinstance(teams, list):
            for team in teams:
                if not isinstance(team, dict):
                    continue
                team_id = team.get("teamId") or team.get("id")
                players_node = (
                    team.get("players")
                    or team.get("userIds")
                    or team.get("users")
                    or team.get("roster")
                )
                _emit(match_id, team_id, players_node)

    node: Any = payload
    for key in ("data", "payload", "result"):
        if isinstance(node, dict) and key in node:
            node = node[key]

    if isinstance(node, dict) and isinstance(node.get("matches"), list):
        for match_obj in node["matches"]:
            if isinstance(match_obj, dict):
                _parse_match_obj(match_obj)
    elif isinstance(node, list):
        if all(isinstance(item, dict) for item in node) and any(
            ("matchId" in item or "teams" in item) for item in node
        ):
            for match_obj in node:
                _parse_match_obj(match_obj)
        else:
            for item in node:
                if not isinstance(item, dict):
                    continue
                user_id = _coerce_int(
                    item.get("userId") or item.get("user_id") or item.get("id")
                )
                match_ids = (
                    item.get("matchIds")
                    or item.get("match_ids")
                    or item.get("matches")
                )
                if user_id is None or not isinstance(match_ids, list):
                    continue
                for match_id in match_ids:
                    match_id_int = _coerce_int(match_id)
                    if match_id_int is None:
                        continue
                    rows.append(
                        {
                            "tournament_id": int(tournament_id),
                            "match_id": match_id_int,
                            "team_id": None,
                            "user_id": user_id,
                        }
                    )

    return [
        row
        for row in rows
        if row.get("match_id") is not None and row.get("user_id") is not None
    ]


def _extract_appearances(
    tournaments: list[dict[str, Any]],
) -> pl.DataFrame | None:
    rows: list[dict[str, int | None]] = []

    for entry in tournaments:
        tournament = entry.get("tournament", {})
        ctx = tournament.get("ctx", {}) if isinstance(tournament, dict) else {}
        tournament_id = ctx.get("id")
        player_matches = entry.get("player_matches")
        if tournament_id is None or not player_matches:
            continue
        rows.extend(
            _extract_appearances_from_players_payload(
                int(tournament_id), player_matches
            )
        )

    if not rows:
        return None

    return (
        pl.DataFrame(rows)
        .with_columns(
            [
                pl.col("tournament_id").cast(pl.Int64, strict=False),
                pl.col("match_id").cast(pl.Int64, strict=False),
                pl.col("team_id").cast(pl.Int64, strict=False),
                pl.col("user_id").cast(pl.Int64, strict=False),
            ]
        )
        .unique(subset=["tournament_id", "match_id", "user_id"])
    )


def _rename_present(
    dataframe: pl.DataFrame | None,
    rename_map: dict[str, str],
) -> pl.DataFrame | None:
    if dataframe is None:
        return None
    present = {
        source: target
        for source, target in rename_map.items()
        if source in dataframe.columns
    }
    return dataframe.rename(present) if present else dataframe


def _list_key(values: list[int] | None) -> str:
    if values is None:
        return ""
    return ",".join(str(value) for value in sorted(values))


def _canonicalize_converted_matches(dataframe: pl.DataFrame) -> pl.DataFrame:
    if dataframe.is_empty():
        return dataframe

    winners_key = [_list_key(values) for values in dataframe["winners"].to_list()]
    losers_key = [_list_key(values) for values in dataframe["losers"].to_list()]

    return (
        dataframe.with_columns(
            [
                pl.Series("__w_key", winners_key),
                pl.Series("__l_key", losers_key),
            ]
        )
        .sort(["tournament_id", "match_id", "__w_key", "__l_key"])
        .select(
            [
                "tournament_id",
                "match_id",
                "__w_key",
                "__l_key",
                "weight",
                "ts",
                *[
                    column
                    for column in ("winner_count", "loser_count", "share")
                    if column in dataframe.columns
                ],
            ]
        )
    )


def _top_k_overlap(
    left_ids: list[int],
    right_ids: list[int],
    k: int,
) -> float | None:
    if k <= 0:
        return None
    if not left_ids and not right_ids:
        return 1.0
    left_top = set(left_ids[:k])
    right_top = set(right_ids[:k])
    denom = max(min(k, len(left_ids), len(right_ids)), 1)
    return len(left_top & right_top) / denom


def _safe_spearman(left: pl.Series, right: pl.Series) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    statistic = spearmanr(left.to_numpy(), right.to_numpy()).statistic
    if statistic is None or math.isnan(statistic):
        return None
    return float(statistic)


def _compare_converted_matches(
    sendou_df: pl.DataFrame,
    loopr_df: pl.DataFrame,
    *,
    tolerance: float,
) -> dict[str, Any]:
    left = _canonicalize_converted_matches(sendou_df)
    right = _canonicalize_converted_matches(loopr_df)

    joined = left.join(
        right,
        on=["tournament_id", "match_id", "__w_key", "__l_key"],
        how="full",
        suffix="_loopr",
    )
    common = joined.filter(
        pl.col("weight").is_not_null() & pl.col("weight_loopr").is_not_null()
    )

    summary: dict[str, Any] = {
        "sendou_rows": int(left.height),
        "loopr_rows": int(right.height),
        "common_rows": int(common.height),
        "sendou_only_rows": int(
            joined.filter(pl.col("weight_loopr").is_null()).height
        ),
        "loopr_only_rows": int(
            joined.filter(pl.col("weight").is_null()).height
        ),
    }

    for column in ("weight", "ts", "winner_count", "loser_count", "share"):
        if column not in common.columns or f"{column}_loopr" not in common.columns:
            continue
        diff = common.select(
            (pl.col(column) - pl.col(f"{column}_loopr"))
            .abs()
            .max()
            .alias("diff")
        ).item()
        summary[f"max_abs_{column}_diff"] = float(diff or 0.0)

    summary["exact_match"] = bool(
        summary["sendou_only_rows"] == 0
        and summary["loopr_only_rows"] == 0
        and all(
            summary.get(metric, 0.0) <= tolerance
            for metric in (
                "max_abs_weight_diff",
                "max_abs_ts_diff",
                "max_abs_share_diff",
            )
            if metric in summary
        )
    )
    return summary


def _build_shared_node_mapping(
    left: pl.DataFrame,
    right: pl.DataFrame,
) -> tuple[list[int], dict[int, int]]:
    ids: set[int] = set()
    for dataframe in (left, right):
        if dataframe.is_empty():
            continue
        for column in ("winners", "losers"):
            ids.update(
                dataframe.select(column)
                .explode(column)[column]
                .drop_nulls()
                .cast(pl.Int64, strict=False)
                .to_list()
            )
    node_ids = sorted(ids)
    return node_ids, {entity_id: idx for idx, entity_id in enumerate(node_ids)}


def _triplet_frame(
    rows: Any,
    cols: Any,
    weights: Any,
    node_ids: list[int],
) -> pl.DataFrame:
    if len(rows) == 0:
        return pl.DataFrame(
            schema={
                "winner_id": pl.Int64,
                "loser_id": pl.Int64,
                "weight_sum": pl.Float64,
            }
        )

    winner_ids = [node_ids[int(idx)] for idx in rows]
    loser_ids = [node_ids[int(idx)] for idx in cols]
    return (
        pl.DataFrame(
            {
                "winner_id": winner_ids,
                "loser_id": loser_ids,
                "weight_sum": [float(weight) for weight in weights],
            }
        )
        .sort(["winner_id", "loser_id"])
    )


def _compare_individual_pair_graph(
    sendou_converted: pl.DataFrame,
    loopr_converted: pl.DataFrame,
    *,
    sendou_edges_module: Any,
    tolerance: float,
) -> dict[str, Any]:
    node_ids, node_to_idx = _build_shared_node_mapping(
        sendou_converted,
        loopr_converted,
    )

    sendou_rows, sendou_cols, sendou_weights = sendou_edges_module.build_exposure_triplets(
        sendou_converted,
        node_to_idx,
    )
    loopr_rows, loopr_cols, loopr_weights = loopr_build_exposure_triplets(
        loopr_converted,
        node_to_idx,
    )

    left = _triplet_frame(sendou_rows, sendou_cols, sendou_weights, node_ids)
    right = _triplet_frame(loopr_rows, loopr_cols, loopr_weights, node_ids)

    joined = left.join(
        right,
        on=["winner_id", "loser_id"],
        how="full",
        suffix="_loopr",
    )
    common = joined.filter(
        pl.col("weight_sum").is_not_null() & pl.col("weight_sum_loopr").is_not_null()
    )

    max_abs_diff = 0.0
    if not common.is_empty():
        max_abs_diff = float(
            common.select(
                (pl.col("weight_sum") - pl.col("weight_sum_loopr"))
                .abs()
                .max()
                .alias("diff")
            ).item()
            or 0.0
        )

    return {
        "nodes": int(len(node_ids)),
        "sendou_pairs": int(left.height),
        "loopr_pairs": int(right.height),
        "common_pairs": int(common.height),
        "sendou_only_pairs": int(
            joined.filter(pl.col("weight_sum_loopr").is_null()).height
        ),
        "loopr_only_pairs": int(
            joined.filter(pl.col("weight_sum").is_null()).height
        ),
        "max_abs_weight_diff": max_abs_diff,
        "exact_match": bool(
            joined.filter(pl.col("weight_sum_loopr").is_null()).is_empty()
            and joined.filter(pl.col("weight_sum").is_null()).is_empty()
            and max_abs_diff <= tolerance
        ),
    }


def _normalize_rankings(
    dataframe: pl.DataFrame,
    *,
    id_column: str,
    score_column: str,
) -> pl.DataFrame:
    rename_map = {id_column: "entity_id", score_column: "score"}
    normalized = dataframe.rename(rename_map)
    keep = ["entity_id", "score"]
    for column in ("win_pr", "loss_pr", "exposure"):
        if column in normalized.columns:
            keep.append(column)
    return normalized.select(keep).sort("score", descending=True)


def _compare_rankings(
    sendou_df: pl.DataFrame,
    loopr_df: pl.DataFrame,
    *,
    tolerance: float,
) -> dict[str, Any]:
    joined = sendou_df.join(
        loopr_df,
        on="entity_id",
        how="full",
        suffix="_loopr",
    )
    common = joined.filter(
        pl.col("score").is_not_null() & pl.col("score_loopr").is_not_null()
    )

    summary: dict[str, Any] = {
        "sendou_rows": int(sendou_df.height),
        "loopr_rows": int(loopr_df.height),
        "common_entities": int(common.height),
        "sendou_only_entities": int(
            joined.filter(pl.col("score_loopr").is_null()).height
        ),
        "loopr_only_entities": int(
            joined.filter(pl.col("score").is_null()).height
        ),
        "spearman_score": _safe_spearman(
            common["score"], common["score_loopr"]
        ),
        "top_25_overlap": _top_k_overlap(
            sendou_df["entity_id"].to_list(),
            loopr_df["entity_id"].to_list(),
            25,
        ),
        "top_100_overlap": _top_k_overlap(
            sendou_df["entity_id"].to_list(),
            loopr_df["entity_id"].to_list(),
            100,
        ),
    }

    for column in ("score", "win_pr", "loss_pr", "exposure"):
        if column not in common.columns or f"{column}_loopr" not in common.columns:
            continue
        diff_columns = common.select(
            [
                (pl.col(column) - pl.col(f"{column}_loopr"))
                .abs()
                .max()
                .alias("max_diff"),
                (pl.col(column) - pl.col(f"{column}_loopr"))
                .abs()
                .mean()
                .alias("mean_diff"),
            ]
        ).row(0)
        summary[f"max_abs_{column}_diff"] = float(diff_columns[0] or 0.0)
        summary[f"mean_abs_{column}_diff"] = float(diff_columns[1] or 0.0)

    summary["all_close"] = bool(
        summary["sendou_only_entities"] == 0
        and summary["loopr_only_entities"] == 0
        and all(
            summary.get(metric, 0.0) <= tolerance
            for metric in (
                "max_abs_score_diff",
                "max_abs_win_pr_diff",
                "max_abs_loss_pr_diff",
                "max_abs_exposure_diff",
            )
            if metric in summary
        )
    )
    return summary


def _compare_influence(
    sendou_influence: dict[int, float] | None,
    loopr_influence: dict[int, float] | None,
) -> dict[str, Any]:
    left = sendou_influence or {}
    right = loopr_influence or {}
    keys = sorted(set(left) | set(right))
    if not keys:
        return {
            "common_tournaments": 0,
            "sendou_only_tournaments": 0,
            "loopr_only_tournaments": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
        }

    diffs = [abs(left.get(key, 1.0) - right.get(key, 1.0)) for key in keys]
    return {
        "common_tournaments": int(len(set(left) & set(right))),
        "sendou_only_tournaments": int(len(set(left) - set(right))),
        "loopr_only_tournaments": int(len(set(right) - set(left))),
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(sum(diffs) / len(diffs)),
    }


def _build_exposure_configs(
    config_mod: Any,
    *,
    profile: str,
) -> tuple[Any, LooprExposureConfig]:
    if profile == "default":
        sendou_cfg = config_mod.ExposureLogOddsConfig()
        loopr_cfg = LooprExposureConfig()
    elif profile == "simple":
        sendou_cfg = config_mod.ExposureLogOddsConfig()
        sendou_cfg.use_tick_tock_active = False
        sendou_cfg.engine.beta = 0.0
        sendou_cfg.engine.score_decay_rate = 0.0
        sendou_cfg.decay = config_mod.DecayConfig(half_life_days=0.0)

        loopr_cfg = LooprExposureConfig(
            use_tick_tock_active=False,
            decay=LooprDecayConfig(half_life_days=0.0),
            engine=LooprEngineConfig(beta=0.0, score_decay_rate=0.0),
        )
    else:
        raise ValueError(f"Unsupported exposure profile: {profile}")

    return sendou_cfg, loopr_cfg


def _build_tick_tock_configs(
    config_mod: Any,
    *,
    profile: str,
) -> tuple[Any, LooprTickTockConfig]:
    if profile == "default":
        return config_mod.TickTockConfig(), LooprTickTockConfig()
    if profile == "simple":
        sendou_cfg = config_mod.TickTockConfig()
        sendou_cfg.engine.beta = 0.0
        sendou_cfg.decay = config_mod.DecayConfig(half_life_days=0.0)

        loopr_cfg = LooprTickTockConfig(
            decay=LooprDecayConfig(half_life_days=0.0),
            engine=LooprEngineConfig(beta=0.0),
        )
        return sendou_cfg, loopr_cfg
    raise ValueError(f"Unsupported tick-tock profile: {profile}")


def _prepare_dataset(
    *,
    sendou_root: Path,
    dataset_source: str,
    input_json: Path,
    window_dir: Path,
    window_filter: str,
    limit_tournaments: int | None,
) -> dict[str, Any]:
    modules = _load_sendou_modules(sendou_root)
    tournaments_payload: list[dict[str, Any]] | None = None
    tournaments_df: pl.DataFrame | None = None

    if dataset_source == "json":
        tournaments_payload = _load_tournament_payload(input_json)
        if limit_tournaments is not None:
            tournaments_payload = tournaments_payload[:limit_tournaments]

        tables = modules["parser"].parse_tournaments_data(tournaments_payload)
        matches_sendou = tables["matches"]
        players_sendou = tables["players"]
        if matches_sendou is None or players_sendou is None:
            raise ValueError(
                "Parsed tournaments did not produce matches and players"
            )
        appearances_sendou = _extract_appearances(tournaments_payload)
    elif dataset_source == "window":
        window_tables = _load_window_dataset(
            window_dir=window_dir,
            filter_mode=window_filter,
            limit_tournaments=limit_tournaments,
        )
        tournaments_df = window_tables["tournaments_df"]
        matches_sendou = window_tables["matches_sendou"]
        players_sendou = window_tables["players_sendou"]
        appearances_sendou = window_tables["appearances_sendou"]
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

    if "player_id" not in players_sendou.columns and "user_id" in players_sendou.columns:
        players_sendou = players_sendou.with_columns(
            pl.col("user_id").alias("player_id")
        )

    appearances_loopr = _rename_present(
        appearances_sendou,
        {
            "tournament_id": "event_id",
            "team_id": "group_id",
            "user_id": "entity_id",
        },
    )

    matches_loopr = _rename_present(
        matches_sendou,
        {
            "tournament_id": "event_id",
            "winner_team_id": "winner_id",
            "loser_team_id": "loser_id",
            "last_game_finished_at": "completed_at",
            "match_created_at": "created_at",
            "is_bye": "walkover",
        },
    )
    participants_loopr = _rename_present(
        players_sendou,
        {
            "tournament_id": "event_id",
            "team_id": "group_id",
            "user_id": "entity_id",
        },
    )

    now_ts = float(
        matches_sendou.select(pl.max("match_created_at")).item() or time.time()
    )

    return {
        "modules": modules,
        "dataset_source": dataset_source,
        "tournaments_payload": tournaments_payload,
        "tournaments_df": tournaments_df,
        "matches_sendou": matches_sendou,
        "players_sendou": players_sendou,
        "appearances_sendou": appearances_sendou,
        "matches_loopr": matches_loopr,
        "participants_loopr": participants_loopr,
        "appearances_loopr": appearances_loopr,
        "now_ts": now_ts,
    }


def _run_exposure_comparison(
    *,
    modules: dict[str, Any],
    matches_sendou: pl.DataFrame,
    players_sendou: pl.DataFrame,
    appearances_sendou: pl.DataFrame | None,
    matches_loopr: pl.DataFrame,
    participants_loopr: pl.DataFrame,
    appearances_loopr: pl.DataFrame | None,
    now_ts: float,
    profile: str,
    tolerance: float,
) -> dict[str, Any]:
    sendou_cfg, loopr_cfg = _build_exposure_configs(
        modules["config"], profile=profile
    )
    sendou_engine = modules["exposure"].ExposureLogOddsEngine(
        sendou_cfg,
        now_ts=now_ts,
    )
    loopr_engine = LooprExposureEngine(loopr_cfg, now_ts=now_ts)

    sendou_start = time.perf_counter()
    sendou_rankings = sendou_engine.rank_players(
        matches_sendou,
        players_sendou,
        appearances=appearances_sendou,
    )
    sendou_elapsed = time.perf_counter() - sendou_start

    loopr_start = time.perf_counter()
    loopr_rankings = loopr_engine.rank_entities(
        matches_loopr,
        participants_loopr,
        appearances=appearances_loopr,
    )
    loopr_elapsed = time.perf_counter() - loopr_start

    return {
        "runtime_seconds": {
            "sendouq_analysis": round(sendou_elapsed, 6),
            "loopr": round(loopr_elapsed, 6),
        },
        "ranking_comparison": _compare_rankings(
            _normalize_rankings(sendou_rankings, id_column="id", score_column="score"),
            _normalize_rankings(
                loopr_rankings, id_column="entity_id", score_column="score"
            ),
            tolerance=tolerance,
        ),
        "tournament_influence_comparison": _compare_influence(
            getattr(sendou_engine, "tournament_influence", None),
            getattr(loopr_engine, "tournament_influence", None),
        ),
    }


def _run_tick_tock_comparison(
    *,
    modules: dict[str, Any],
    matches_sendou: pl.DataFrame,
    players_sendou: pl.DataFrame,
    matches_loopr: pl.DataFrame,
    participants_loopr: pl.DataFrame,
    now_ts: float,
    profile: str,
    tolerance: float,
) -> dict[str, Any]:
    sendou_cfg, loopr_cfg = _build_tick_tock_configs(
        modules["config"], profile=profile
    )
    sendou_engine = modules["tick_tock"].TickTockEngine(
        sendou_cfg,
        now_ts=now_ts,
    )
    loopr_engine = LooprTickTockEngine(loopr_cfg, now_ts=now_ts)

    sendou_start = time.perf_counter()
    sendou_rankings = sendou_engine.rank_players(matches_sendou, players_sendou)
    sendou_elapsed = time.perf_counter() - sendou_start

    loopr_start = time.perf_counter()
    loopr_rankings = loopr_engine.rank_entities(
        matches_loopr,
        participants_loopr,
    )
    loopr_elapsed = time.perf_counter() - loopr_start

    return {
        "runtime_seconds": {
            "sendouq_analysis": round(sendou_elapsed, 6),
            "loopr": round(loopr_elapsed, 6),
        },
        "ranking_comparison": _compare_rankings(
            _normalize_rankings(
                sendou_rankings, id_column="player_id", score_column="rating"
            ),
            _normalize_rankings(
                loopr_rankings, id_column="entity_id", score_column="score"
            ),
            tolerance=tolerance,
        ),
        "tournament_influence_comparison": _compare_influence(
            getattr(sendou_engine, "tournament_influence", None),
            getattr(loopr_engine, "tournament_influence", None),
        ),
    }


def _run_isolated_benchmark(
    *,
    engine: str,
    implementation: str,
    dataset_source: str,
    profile: str,
    sendou_root: Path,
    input_json: Path,
    window_dir: Path,
    window_filter: str,
    limit_tournaments: int | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-benchmark",
        "--dataset-source",
        dataset_source,
        "--engine",
        engine,
        "--implementation",
        implementation,
        "--profile",
        profile,
        "--sendouq-root",
        str(sendou_root),
        "--input-json",
        str(input_json),
        "--window-dir",
        str(window_dir),
        "--window-filter",
        window_filter,
    ]
    if limit_tournaments is not None:
        cmd.extend(["--limit-tournaments", str(limit_tournaments)])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str((Path(__file__).resolve().parent.parent / "src").resolve())
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(completed.stdout)


def _collect_process_metrics(args: argparse.Namespace) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    engines = (
        ["exposure", "tick_tock"] if args.engine == "both" else [args.engine]
    )
    for engine in engines:
        metrics[engine] = {}
        for implementation in ("sendouq_analysis", "loopr"):
            metrics[engine][implementation] = _run_isolated_benchmark(
                engine=engine,
                implementation=implementation,
                dataset_source=args.dataset_source,
                profile=args.profile,
                sendou_root=args.sendouq_root,
                input_json=args.input_json,
                window_dir=args.window_dir,
                window_filter=args.window_filter,
                limit_tournaments=args.limit_tournaments,
            )
    return metrics


def _run_internal_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _prepare_dataset(
        sendou_root=args.sendouq_root,
        dataset_source=args.dataset_source,
        input_json=args.input_json,
        window_dir=args.window_dir,
        window_filter=args.window_filter,
        limit_tournaments=args.limit_tournaments,
    )

    start = time.perf_counter()
    if args.engine == "exposure":
        if args.implementation == "sendouq_analysis":
            sendou_cfg, _ = _build_exposure_configs(
                dataset["modules"]["config"],
                profile=args.profile,
            )
            engine = dataset["modules"]["exposure"].ExposureLogOddsEngine(
                sendou_cfg,
                now_ts=dataset["now_ts"],
            )
            rankings = engine.rank_players(
                dataset["matches_sendou"],
                dataset["players_sendou"],
                appearances=dataset["appearances_sendou"],
            )
        else:
            _, loopr_cfg = _build_exposure_configs(
                dataset["modules"]["config"],
                profile=args.profile,
            )
            engine = LooprExposureEngine(loopr_cfg, now_ts=dataset["now_ts"])
            rankings = engine.rank_entities(
                dataset["matches_loopr"],
                dataset["participants_loopr"],
                appearances=dataset["appearances_loopr"],
            )
    elif args.engine == "tick_tock":
        if args.implementation == "sendouq_analysis":
            sendou_cfg, _ = _build_tick_tock_configs(
                dataset["modules"]["config"],
                profile=args.profile,
            )
            engine = dataset["modules"]["tick_tock"].TickTockEngine(
                sendou_cfg,
                now_ts=dataset["now_ts"],
            )
            rankings = engine.rank_players(
                dataset["matches_sendou"],
                dataset["players_sendou"],
            )
        else:
            _, loopr_cfg = _build_tick_tock_configs(
                dataset["modules"]["config"],
                profile=args.profile,
            )
            engine = LooprTickTockEngine(loopr_cfg, now_ts=dataset["now_ts"])
            rankings = engine.rank_entities(
                dataset["matches_loopr"],
                dataset["participants_loopr"],
            )
    else:
        raise ValueError(f"Unsupported engine for benchmark: {args.engine}")

    elapsed = time.perf_counter() - start
    return {
        "engine": args.engine,
        "dataset_source": args.dataset_source,
        "implementation": args.implementation,
        "profile": args.profile,
        "rows": int(rankings.height),
        "wall_seconds": round(elapsed, 6),
        "peak_rss_mb": round(_peak_rss_mb(), 3),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _prepare_dataset(
        sendou_root=args.sendouq_root,
        dataset_source=args.dataset_source,
        input_json=args.input_json,
        window_dir=args.window_dir,
        window_filter=args.window_filter,
        limit_tournaments=args.limit_tournaments,
    )

    tournaments_loaded = (
        len(dataset["tournaments_payload"])
        if dataset["tournaments_payload"] is not None
        else int(dataset["tournaments_df"].height)
        if dataset["tournaments_df"] is not None
        else 0
    )

    sendou_converted = dataset["modules"]["convert"].convert_matches_dataframe(
        dataset["matches_sendou"],
        dataset["players_sendou"],
        {},
        dataset["now_ts"],
        0.0,
        0.0,
        appearances=dataset["appearances_sendou"],
        include_share=True,
    )
    loopr_converted = loopr_convert_matches_dataframe(
        dataset["matches_loopr"],
        dataset["participants_loopr"],
        {},
        dataset["now_ts"],
        0.0,
        0.0,
        appearances=dataset["appearances_loopr"],
        include_share=True,
    )

    report: dict[str, Any] = {
        "input": {
            "sendouq_root": str(args.sendouq_root),
            "dataset_source": args.dataset_source,
            "input_json": str(args.input_json),
            "window_dir": str(args.window_dir),
            "window_filter": args.window_filter,
            "profile": args.profile,
            "engine": args.engine,
            "limit_tournaments": args.limit_tournaments,
            "tolerance": args.tolerance,
        },
        "dataset": {
            "tournaments_loaded": tournaments_loaded,
            "matches_rows": int(dataset["matches_sendou"].height),
            "players_rows": int(dataset["players_sendou"].height),
            "appearance_rows": int(dataset["appearances_sendou"].height)
            if dataset["appearances_sendou"] is not None
            else 0,
            "max_match_created_at": dataset["now_ts"],
        },
        "resolved_individual_match_comparison": _compare_converted_matches(
            sendou_converted,
            loopr_converted,
            tolerance=args.tolerance,
        ),
        "resolved_individual_pair_graph_comparison": _compare_individual_pair_graph(
            sendou_converted,
            loopr_converted,
            sendou_edges_module=dataset["modules"]["edges"],
            tolerance=args.tolerance,
        ),
        "process_metrics": _collect_process_metrics(args),
    }

    if args.engine in {"exposure", "both"}:
        report["exposure_log_odds"] = _run_exposure_comparison(
            modules=dataset["modules"],
            matches_sendou=dataset["matches_sendou"],
            players_sendou=dataset["players_sendou"],
            appearances_sendou=dataset["appearances_sendou"],
            matches_loopr=dataset["matches_loopr"],
            participants_loopr=dataset["participants_loopr"],
            appearances_loopr=dataset["appearances_loopr"],
            now_ts=dataset["now_ts"],
            profile=args.profile,
            tolerance=args.tolerance,
        )

    if args.engine in {"tick_tock", "both"}:
        report["tick_tock"] = _run_tick_tock_comparison(
            modules=dataset["modules"],
            matches_sendou=dataset["matches_sendou"],
            players_sendou=dataset["players_sendou"],
            matches_loopr=dataset["matches_loopr"],
            participants_loopr=dataset["participants_loopr"],
            now_ts=dataset["now_ts"],
            profile=args.profile,
            tolerance=args.tolerance,
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare loopr outputs against the legacy sendouq_analysis "
            "implementation on the same tournament payload."
        )
    )
    parser.add_argument(
        "--sendouq-root",
        type=Path,
        default=Path("/root/dev/sendouq_analysis"),
        help="Path to the local sendouq_analysis checkout.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/root/dev/sendouq_analysis/sample_tournament.json"),
        help="Tournament JSON payload to parse and compare.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["json", "window"],
        default="json",
        help=(
            "Dataset source. 'json' parses a tournament export payload. "
            "'window' loads the larger compiled parquet dataset."
        ),
    )
    parser.add_argument(
        "--window-dir",
        type=Path,
        default=Path("/root/dev/sendouq_analysis/data/embeddings_window_540d_all"),
        help="Path to the compiled Sendou parquet window dataset.",
    )
    parser.add_argument(
        "--window-filter",
        choices=["none", "ranked_only", "ranked_quality"],
        default="ranked_quality",
        help=(
            "Tournament filter used for --dataset-source window. "
            "'ranked_quality' matches the stricter Plus-validation style filter."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["exposure", "tick_tock", "both"],
        default="both",
        help="Which engine family to compare.",
    )
    parser.add_argument(
        "--profile",
        choices=["default", "simple"],
        default="default",
        help=(
            "Comparison profile. 'default' uses package defaults. "
            "'simple' disables tick-tock-active for exposure and turns off "
            "decay/beta-heavy weighting."
        ),
    )
    parser.add_argument(
        "--limit-tournaments",
        type=int,
        default=None,
        help="Optional cap on tournaments loaded from the payload.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance used for all-close style checks.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the report JSON.",
    )
    parser.add_argument(
        "--internal-benchmark",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--implementation",
        choices=["sendouq_analysis", "loopr"],
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.internal_benchmark:
        print(json.dumps(_run_internal_benchmark(args), sort_keys=True))
        return
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
