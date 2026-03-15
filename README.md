# LOOPR

`loopr` is a domain-agnostic ranking library for event-based competition data.
It is built around neutral tabular inputs, supports several ranking engines, and
includes exact leave-one-match-out impact analysis for the main log-odds engine.

The recommended public API is:

- `rank_entities(...)` for rankings
- `prepare_rank_inputs(...)` and the `normalize_*_schema(...)` helpers for schema adaptation
- `LOOPREngine` for the main exposure log-odds model

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

## Recommended Input Schema

`loopr` accepts neutral input tables and normalizes them internally.

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

If `appearances` are provided, the engines use them to determine who actually
played in a match. If `group_id` is missing, `loopr` infers it from the
participants table when possible.

The public API is neutral-schema-only. Inputs that use the older
`tournament` / `team` / `user` naming are no longer accepted by the supported
surface.

## Engine Overview

### `LOOPREngine` / `ExposureLogOddsEngine`

This is the main engine and the recommended default.

Characteristics:

- volume-neutral, exposure-based log-odds ranking
- optional time decay
- optional inactivity decay
- optional tick-tock active-entity filtering
- exact leave-one-match-out analysis support

`rank_entities(...)` returns a Polars `DataFrame` sorted by rank with:

- `entity_id`
- `player_rank`
- `score`
- `win_pr`
- `loss_pr`
- `exposure`

`player_rank` and `score` currently carry the same value.

### `TickTockEngine`

This engine iteratively estimates tournament influence and ratings in alternating
tick/tock steps.

`rank_entities(...)` returns:

- `entity_id`
- `score`

### `TTLEngine`

TTL combines tick-tock tournament influence updates with the log-odds backend.

`rank_entities(...)` returns at least:

- `entity_id`
- `score`
- `active`

It may also include:

- `win_pr`
- `loss_pr`
- `exposure`
- `quality_mass`

## Configuration

The main configuration types exported by the package are:

- `DecayConfig`
- `PageRankConfig`
- `EngineConfig`
- `TickTockConfig`
- `ExposureLogOddsConfig`

Example:

```python
from loopr import (
    DecayConfig,
    EngineConfig,
    ExposureLogOddsConfig,
    LOOPREngine,
    PageRankConfig,
    TickTockConfig,
)

config = ExposureLogOddsConfig(
    decay=DecayConfig(half_life_days=45.0),
    pagerank=PageRankConfig(alpha=0.85, tol=1e-8, max_iter=200),
    engine=EngineConfig(
        beta=1.0,
        min_exposure=2.0,
        score_decay_delay_days=30.0,
        score_decay_rate=0.01,
    ),
    tick_tock=TickTockConfig(max_ticks=5, convergence_tol=1e-4),
    lambda_mode="auto",
    fixed_lambda=None,
    use_tick_tock_active=True,
    apply_log_transform=True,
)

engine = LOOPREngine(config=config)
```

Important knobs for `LOOPREngine`:

- `decay.half_life_days`: recency weighting for match importance
- `engine.beta`: strength of tournament influence weighting
- `engine.min_exposure`: filter low-exposure entities from final output
- `engine.score_decay_delay_days` / `engine.score_decay_rate`: inactivity decay after ranking
- `lambda_mode` / `fixed_lambda`: score smoothing mode
- `use_tick_tock_active`: whether to derive active entities via tick-tock first

## Working With Appearances

When only a subset of a roster actually played in a match, pass an
`appearances` table:

```python
rankings = engine.rank_entities(
    matches,
    participants,
    appearances=appearances,
)
```

This matters because the log-odds engines distribute match exposure across the
entities that participated in the match, not just the full roster.

## Schema Utilities

The package exports `prepare_rank_inputs(...)` to validate neutral inputs and
rename them into the internal engine schema:

```python
from loopr import prepare_rank_inputs

prepared = prepare_rank_inputs(matches, participants, appearances)
```

`prepare_rank_inputs(...)` returns a `NormalizedRankingInputs` object with:

- `matches`
- `participants`
- `appearances`

## Leave-One-Match-Out Analysis

`LOOPREngine` includes exact leave-one-match-out analysis based on low-rank
PageRank updates.

```python
engine = LOOPREngine()
engine.rank_entities(matches, participants, appearances=appearances)

engine.prepare_loo_analyzer()

impact = engine.analyze_match_impact(match_id=10, entity_id=1)
print(impact["delta"]["score"])

entity_impacts = engine.analyze_entity_matches(
    entity_id=1,
    limit=20,
    include_teleport=True,
    parallel=True,
    max_workers=4,
)
print(entity_impacts.head())
```

Useful methods:

- `prepare_loo_analyzer(...)`
- `analyze_match_impact(match_id, entity_id, include_teleport=True)`
- `analyze_entity_matches(entity_id, limit=None, include_teleport=True, parallel=True, max_workers=4)`
- `get_loo_analyzer()`

`analyze_entity_matches(...)` returns a `DataFrame` sorted by absolute impact
with columns:

- `match_id`
- `is_win`
- `old_score`
- `new_score`
- `score_delta`
- `abs_delta`
- `win_pr_delta`
- `loss_pr_delta`

## Diagnostics

After a `LOOPREngine` run, the engine exposes:

- `last_result`: an `ExposureLogOddsResult`
- `last_stage_timings`: per-stage timing data for the most recent run
- `tournament_influence`: the influence weights used for that run

`last_result` includes arrays such as:

- `scores`
- `ids`
- `win_pagerank`
- `loss_pagerank`
- `teleport`
- `exposure`
- `lambda_used`
- `stage_timings`

## Benchmarking

A synthetic benchmark is included at:

- [benchmarks/benchmark_rank_entities.py](benchmarks/benchmark_rank_entities.py)

Example:

```bash
PYTHONPATH=src python benchmarks/benchmark_rank_entities.py \
  --events 40 \
  --teams-per-event 32 \
  --matches-per-event 160 \
  --roster-size 4 \
  --repeats 3
```

The script prints JSON with:

- workload size
- row counts
- min/mean/max runtime
- mean per-stage timings from `LOOPREngine.last_stage_timings`

## Development

Run the test suite with:

```bash
pytest -q
```

The codebase uses Polars, NumPy, and SciPy for the main compute path.

## License

Apache-2.0. See [LICENSE](LICENSE).
