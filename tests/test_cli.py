from __future__ import annotations

from pathlib import Path

import polars as pl

from loopr.cli import main


def test_cli_rank_writes_csv_output(
    single_match_neutral_tables,
    tmp_path: Path,
):
    matches_path = tmp_path / "matches.csv"
    participants_path = tmp_path / "participants.csv"
    appearances_path = tmp_path / "appearances.csv"
    output_path = tmp_path / "rankings.csv"

    single_match_neutral_tables["matches"].write_csv(matches_path)
    single_match_neutral_tables["participants"].write_csv(participants_path)
    single_match_neutral_tables["appearances"].write_csv(appearances_path)

    exit_code = main(
        [
            "rank",
            "--matches",
            str(matches_path),
            "--participants",
            str(participants_path),
            "--appearances",
            str(appearances_path),
            "--output",
            str(output_path),
            "--now-ts",
            "1700000000",
        ]
    )

    assert exit_code == 0
    rankings = pl.read_csv(output_path)
    assert "entity_id" in rankings.columns
    assert "score" in rankings.columns
    assert rankings.height > 0


def test_cli_rank_prints_csv_to_stdout(
    single_match_neutral_tables,
    tmp_path: Path,
    capsys,
):
    matches_path = tmp_path / "matches.csv"
    participants_path = tmp_path / "participants.csv"

    single_match_neutral_tables["matches"].write_csv(matches_path)
    single_match_neutral_tables["participants"].write_csv(participants_path)

    exit_code = main(
        [
            "rank",
            "--matches",
            str(matches_path),
            "--participants",
            str(participants_path),
            "--now-ts",
            "1700000000",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "entity_id" in stdout
    assert "score" in stdout
