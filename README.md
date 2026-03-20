# LOOPR

`loopr` is a domain-agnostic ranking library for event-based competition data.
It is built for the common real-world case where outcomes are recorded for
groups or teams, but the thing you actually want to rank is individuals. Give
it event results plus who belonged to or appeared for each group, and it
decomposes that team-shaped evidence into entity-level rankings.

The stable public API is:

- `rank_entities(...)` for the one-shot ranking path
- `assess_dataset_fit(...)` for checking whether a dataset matches LOOPR's main input assumptions
- `prepare_rank_inputs(...)` for validating neutral input tables
- `LOOPREngine` when you want engine state, diagnostics, or LOO analysis
- `ExposureLogOddsConfig` when you need to override defaults

`LOOPREngine` is currently an alias for `ExposureLogOddsEngine`.

## Installation

PyPI distribution name:

```bash
pip install loopr-ranking
```

Import path remains:

```python
import loopr
```

For local development:

```bash
pip install -e .
pip install -e .[dev]
```

If you use `uv`:

```bash
uv sync --extra dev
```

Requires Python 3.10+.

## Docker Build

The wheel and sdist release path is also available through the multi-stage
`Dockerfile`.

Run tests and quickstart smoke inside Docker:

```bash
docker build --target test .
```

Build distributable artifacts into a local directory:

```bash
docker buildx build --target artifacts --output type=local,dest=dist .
```

## Quick Start

Command line:

```bash
loopr rank \
  --matches examples/quickstart/matches.csv \
  --participants examples/quickstart/participants.csv \
  --appearances examples/quickstart/appearances.csv \
  --output rankings.csv
```

If the package is installed and you prefer the module entrypoint:

```bash
python -m loopr rank \
  --matches examples/quickstart/matches.csv \
  --participants examples/quickstart/participants.csv \
  --appearances examples/quickstart/appearances.csv \
  --output rankings.csv
```

Python:

```python
import polars as pl

from loopr import rank_entities

matches = pl.read_csv("examples/quickstart/matches.csv")
participants = pl.read_csv("examples/quickstart/participants.csv")
appearances = pl.read_csv("examples/quickstart/appearances.csv")
entities = pl.read_csv("examples/quickstart/entities.csv")

rankings = rank_entities(
    matches,
    participants,
    appearances=appearances,
)

print(
    rankings.join(entities, on="entity_id")
    .select(["entity_id", "entity_name", "score", "exposure"])
    .head(10)
)
```

If your spreadsheets look like this, you are on the happy path.

`matches.csv`

| event_id | match_id | winner_id | loser_id | completed_at |
| --- | --- | --- | --- | --- |
| 1 | 10 | 100 | 200 | 1700000000 |
| 1 | 11 | 100 | 300 | 1700003600 |
| 1 | 12 | 300 | 200 | 1700007200 |

`participants.csv`

| event_id | group_id | entity_id |
| --- | --- | --- |
| 1 | 100 | 1 |
| 1 | 100 | 2 |
| 1 | 200 | 3 |
| 1 | 200 | 4 |
| 1 | 300 | 5 |
| 1 | 300 | 6 |

`appearances.csv` (optional)

| event_id | match_id | entity_id | group_id |
| --- | --- | --- | --- |
| 1 | 10 | 1 | 100 |
| 1 | 10 | 2 | 100 |
| 1 | 10 | 3 | 200 |
| 1 | 10 | 4 | 200 |

If you omit `appearances`, `loopr` assumes the full event-level group roster
participated in each result.

The example files shown above live in the repository under
`examples/quickstart/`. They are not installed with the wheel. After
installing the package, you can regenerate them with:

```bash
python -m loopr.example_data --output-dir examples/quickstart
```

## Recommended Defaults

For most users, the right starting point is:

- `rank_entities(...)`
- default configuration
- `appearances` included whenever available

By default, `rank_entities(...)` keeps only the largest connected component of
the resolved comparison graph. If the graph is disconnected, LOOPR warns and
drops the smaller disconnected pieces before ranking. Use
`component_policy="allow"` to keep all components anyway or
`component_policy="error"` to fail fast instead.

Treat the output as a ranking signal, not a calibrated probability-like score.
Tune only after you have an external validation target or a clear operational
reason. The deeper guidance lives in the repository docs under
`docs/defaults-and-recipes.md` and `docs/validation-harness.md`.

## Public Input Shape

The main public path expects binary group-result inputs and returns
individual/entity rankings.

### Matches

Required columns:

- `event_id`
- `match_id`
- `winner_id`
- `loser_id`

Optional columns:

- `completed_at`
- `created_at`
- `walkover`

### Participants

Required columns:

- `event_id`
- `group_id`
- `entity_id`

### Appearances

Optional per-match participation data:

- `event_id`
- `match_id`
- `entity_id`
- optional `group_id`

If `appearances` are present, `loopr` uses them to determine who actually
played in a result instead of assuming the full roster participated.

## Stable API

For published usage, prefer these as the supported surface:

- `assess_dataset_fit(...)`
- `rank_entities(...)`
- `loopr rank`
- `prepare_rank_inputs(...)`
- `LOOPREngine`
- `ExposureLogOddsConfig`

The public API is neutral-schema-only. Inputs using the older
`tournament` / `team` / `user` naming are not part of the supported public
surface.

## Engine Summary

### `LOOPREngine` / `ExposureLogOddsEngine`

The recommended default. It is volume-neutral, exposure-based, supports time
decay and inactivity decay, and includes exact leave-one-match-out analysis.

### `TickTockEngine`

An iterative tournament-influence engine that alternates between entity rating
updates and tournament-strength updates.

### `TTLEngine`

A hybrid flow that uses tick-tock tournament influence updates with a rating
backend such as the log-odds backend.

## Documentation

README is the general entrypoint. The advanced docs live in the repository
under `docs/`, including:

- `docs/README.md`: advanced docs index
- `docs/input-patterns.md`: schema guidance and normalization
- `docs/how-loopr-works.md`: technical walkthrough
- `docs/mathematical-machinery.md`: equations and intuition
- `docs/engines-and-configuration.md`: engine comparison and configuration
- `docs/defaults-and-recipes.md`: starting setup and baseline recipes
- `docs/validation-and-benchmarks.md`: evaluation guidance
- `docs/validation-harness.md`: reusable validation contract
- `docs/ablations.md`: ingredient-level comparison design
- `docs/case-studies/README.md`: applied examples
- `docs/analysis-and-diagnostics.md`: diagnostics and LOO analysis

## Common Next Steps

- Use `assess_dataset_fit(...)` before serious tuning when you want a quick read on appearance coverage, roster fallback risk, and participant/appearance alignment.
  It is a heuristic suitability check for LOOPR's global-ranking assumptions,
  not a guarantee of ranking quality and not merely a schema or runtime-validity check.
  It also reports whether the resolved entity comparison graph is disconnected,
  how much share mass sits outside the largest component, and whether ranking
  the dataset as-is is a bad fit for LOOPR.
- Use `prepare_rank_inputs(...)` when you want schema validation before ranking
  or when debugging normalized inputs.
- Pass `appearances` when match-level participation differs from the stored
  roster.
- Start with `rank_entities(...)` and the default config, then validate before
  tuning; see `docs/defaults-and-recipes.md` in the repository.

## Development

Run the test suite with:

```bash
pytest -q
```

The codebase uses Polars, NumPy, and SciPy for the main compute path.

## License

Apache-2.0. See [LICENSE](LICENSE).
