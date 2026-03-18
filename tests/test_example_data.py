from __future__ import annotations

from pathlib import Path

import polars as pl

from loopr import rank_entities
from loopr.example_data import write_quickstart_dataset


def test_quickstart_generator_reproduces_tracked_example(tmp_path: Path):
    written = write_quickstart_dataset(tmp_path)
    example_dir = Path("examples/quickstart")

    for name in ("entities", "groups", "matches", "participants", "appearances"):
        generated = pl.read_csv(written[name])
        tracked = pl.read_csv(example_dir / f"{name}.csv")
        assert generated.equals(tracked)


def test_tracked_quickstart_dataset_ranks_cleanly():
    example_dir = Path("examples/quickstart")
    matches = pl.read_csv(example_dir / "matches.csv")
    participants = pl.read_csv(example_dir / "participants.csv")
    appearances = pl.read_csv(example_dir / "appearances.csv")

    rankings = rank_entities(
        matches,
        participants,
        appearances=appearances,
        now_ts=float(matches["completed_at"].max()),
    )

    assert rankings.height > 0
    assert "entity_id" in rankings.columns
    assert "score" in rankings.columns
    assert "exposure" in rankings.columns
