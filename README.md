# LOOPR

`loopr` is a domain-agnostic ranking library for event-based competition data.
It focuses on neutral tabular inputs, entity-level rankings derived from match
results, and exact leave-one-match-out analysis for the main log-odds engine.

The recommended public entrypoint is:

- `LOOPREngine` for the main ranking workflow
- `rank_entities(...)` for rankings
- `prepare_rank_inputs(...)` for validating and normalizing neutral input tables

`LOOPREngine` is currently an alias for `ExposureLogOddsEngine`.

## Installation

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

## Quick Start

```python
import polars as pl

from loopr import LOOPREngine

matches = pl.DataFrame(
    {
        "event_id": [1],
        "match_id": [10],
        "winner_id": [100],
        "loser_id": [200],
        "completed_at": [1_700_000_000],
    }
)

participants = pl.DataFrame(
    {
        "event_id": [1] * 8,
        "group_id": [100] * 4 + [200] * 4,
        "entity_id": [1, 2, 3, 4, 5, 6, 7, 8],
    }
)

engine = LOOPREngine()
rankings = engine.rank_entities(matches, participants)

print(rankings.select(["entity_id", "score"]).head())
```

## Recommended Defaults

For most users, the right starting point is:

- `LOOPREngine()`
- default configuration
- `appearances` included whenever available

Treat the output as a ranking signal, not a calibrated probability-like score.
Tune only after you have an external validation target or a clear operational
reason. For the opinionated defaults guide, see
[docs/defaults-and-recipes.md](docs/defaults-and-recipes.md). For the reusable
evaluation contract, see
[docs/validation-harness.md](docs/validation-harness.md).

## Public Input Shape

The main public `rank_entities(...)` path expects binary group-result inputs.

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
played in a match instead of assuming the full roster participated.

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

README is the general entrypoint. The advanced docs live under `docs/`.

- [docs/README.md](docs/README.md): advanced docs index
- [docs/input-patterns.md](docs/input-patterns.md): detailed schema guidance, participants, appearances, and normalization
- [docs/result-modes.md](docs/result-modes.md): binary group results vs positional results
- [docs/how-loopr-works.md](docs/how-loopr-works.md): deeper technical walkthrough of weighting, graph construction, and engine execution
- [docs/engines-and-configuration.md](docs/engines-and-configuration.md): engine comparison, outputs, and configuration knobs
- [docs/defaults-and-recipes.md](docs/defaults-and-recipes.md): recommended starting setup, conservative debugging baseline, and a minimal ablation grid
- [docs/validation-and-benchmarks.md](docs/validation-and-benchmarks.md): how to evaluate `loopr`, structure benchmark reports, and separate runtime evidence from model-quality evidence
- [docs/validation-harness.md](docs/validation-harness.md): the concrete contract for future reusable validation runs, reports, and benchmark artifacts
- [docs/ablations.md](docs/ablations.md): how to compare ingredients like appearances, decay, tournament influence, and tick-tock cleanly
- [docs/case-studies/README.md](docs/case-studies/README.md): concrete applied examples, including Sendou Plus and a Mario Kart positional-results case study
- [docs/analysis-and-diagnostics.md](docs/analysis-and-diagnostics.md): leave-one-match-out analysis, diagnostics, and benchmarking

## Common Next Steps

- Use `prepare_rank_inputs(...)` when you want schema validation or normalized
  internal column names before calling lower-level helpers.
- Pass `appearances` when match-level participation differs from the stored
  roster.
- Start with the default config and validate before tuning; see
  [docs/defaults-and-recipes.md](docs/defaults-and-recipes.md).
- Treat positional-result handling as an advanced/helper-level feature for now;
  see [docs/result-modes.md](docs/result-modes.md).

## Development

Run the test suite with:

```bash
pytest -q
```

The codebase uses Polars, NumPy, and SciPy for the main compute path.

## License

Apache-2.0. See [LICENSE](LICENSE).
