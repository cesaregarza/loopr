from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from loopr.cli import main
from loopr._version import __version__


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


def test_cli_rank_writes_parquet_output(
    single_match_neutral_tables,
    tmp_path: Path,
):
    matches_path = tmp_path / "matches.parquet"
    participants_path = tmp_path / "participants.parquet"
    appearances_path = tmp_path / "appearances.parquet"
    output_path = tmp_path / "rankings.parquet"

    single_match_neutral_tables["matches"].write_parquet(matches_path)
    single_match_neutral_tables["participants"].write_parquet(participants_path)
    single_match_neutral_tables["appearances"].write_parquet(appearances_path)

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
    rankings = pl.read_parquet(output_path)
    assert "entity_id" in rankings.columns
    assert "score" in rankings.columns


def test_cli_rank_reports_bad_output_suffix(
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
            "--output",
            str(tmp_path / "rankings.txt"),
        ]
    )

    assert exit_code == 2
    assert "Unsupported output format" in capsys.readouterr().err


def test_cli_rank_reports_missing_input_file(tmp_path: Path, capsys):
    participants_path = tmp_path / "participants.csv"
    pl.DataFrame(
        {
            "event_id": [1],
            "group_id": [10],
            "entity_id": [1],
        }
    ).write_csv(participants_path)

    exit_code = main(
        [
            "rank",
            "--matches",
            str(tmp_path / "missing.csv"),
            "--participants",
            str(participants_path),
        ]
    )

    assert exit_code == 2
    assert "No such file or directory" in capsys.readouterr().err


def test_cli_version_reports_package_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert f"loopr {__version__}" in capsys.readouterr().out


def test_cli_rank_warns_and_keeps_largest_component(
    tmp_path: Path,
    capsys,
):
    matches = pl.DataFrame(
        {
            "event_id": [1, 2],
            "match_id": [10, 20],
            "winner_id": [100, 300],
            "loser_id": [200, 400],
        }
    )
    participants = pl.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "group_id": [100, 200, 300, 400],
            "entity_id": [1, 2, 3, 4],
        }
    )
    matches_path = tmp_path / "matches.csv"
    participants_path = tmp_path / "participants.csv"
    output_path = tmp_path / "rankings.csv"
    matches.write_csv(matches_path)
    participants.write_csv(participants_path)

    with pytest.warns(UserWarning, match="disconnected components"):
        exit_code = main(
            [
                "rank",
                "--matches",
                str(matches_path),
                "--participants",
                str(participants_path),
                "--output",
                str(output_path),
                "--now-ts",
                "1700000000",
            ]
        )

    assert exit_code == 0
    rankings = pl.read_csv(output_path)
    assert set(rankings["entity_id"].to_list()) == {1, 2}
