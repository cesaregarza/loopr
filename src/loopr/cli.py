"""Small CSV/parquet-first CLI for LOOPR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from loopr._version import __version__
from loopr.api import rank_entities


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(
        f"Unsupported input format for {path}. Use .csv or .parquet."
    )


def _write_table(frame: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.write_csv(path)
        return
    if suffix == ".parquet":
        frame.write_parquet(path)
        return
    raise ValueError(
        f"Unsupported output format for {path}. Use .csv or .parquet."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loopr",
        description="Rank individuals from team-shaped competition results.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank_parser = subparsers.add_parser(
        "rank",
        help="Rank entities from neutral-schema CSV or parquet tables.",
    )
    rank_parser.add_argument("--matches", required=True, type=Path)
    rank_parser.add_argument("--participants", required=True, type=Path)
    rank_parser.add_argument("--appearances", type=Path)
    rank_parser.add_argument(
        "--component-policy",
        choices=["keep_largest", "allow", "error"],
        default="keep_largest",
        help="How to handle disconnected comparison graphs.",
    )
    rank_parser.add_argument(
        "--output",
        type=Path,
        help="Output .csv or .parquet path. Defaults to CSV on stdout.",
    )
    rank_parser.add_argument(
        "--now-ts",
        type=float,
        help="Optional fixed timestamp for reproducible weighting.",
    )
    return parser


def _run_rank_command(args: argparse.Namespace) -> int:
    matches = _read_table(args.matches)
    participants = _read_table(args.participants)
    appearances = _read_table(args.appearances) if args.appearances else None

    rankings = rank_entities(
        matches,
        participants,
        appearances=appearances,
        now_ts=args.now_ts,
        component_policy=args.component_policy,
    )

    if args.output is not None:
        _write_table(rankings, args.output)
        return 0

    sys.stdout.write(rankings.write_csv())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "rank":
            return _run_rank_command(args)
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        sys.stderr.write(f"{parser.prog}: error: {exc}\n")
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
