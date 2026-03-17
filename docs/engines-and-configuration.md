# Engines And Configuration

This page compares the exported engines and collects the main configuration
knobs in one place.

For the fast-start overview, see [README.md](../README.md).
For the recommended starting setup, see
[defaults-and-recipes.md](defaults-and-recipes.md).

## Engine Selection

### `LOOPREngine` / `ExposureLogOddsEngine`

This is the recommended default.

Use it when you want:

- entity rankings from match exposure
- volume-neutral scoring
- optional time decay and inactivity decay
- exact leave-one-match-out analysis support

Typical output columns:

- `entity_id`
- `player_rank`
- `score`
- `win_pr`
- `loss_pr`
- `exposure`

### `TickTockEngine`

Use it when you specifically want iterative tournament-influence estimation as
the primary algorithm.

Typical output columns:

- `entity_id`
- `score`

### `TTLEngine`

Use it when you want tick-tock tournament influence updates combined with a
rating backend such as the log-odds backend.

Typical output columns include:

- `entity_id`
- `score`
- `active`
- sometimes `win_pr`, `loss_pr`, `exposure`, `quality_mass`

## Practical Selection Guide

Choose `LOOPREngine` when:

- you want the main recommended production path
- volume neutrality matters
- tournament sizes and participation volume vary a lot
- you want exact leave-one-match-out analysis support

Choose `TickTockEngine` when:

- you want explicit tournament influence outputs
- interpretability of the tournament-strength loop matters
- comparing a more direct PageRank-style engine is useful

Choose `TTLEngine` when:

- you want to experiment with hybrid outer-loop tournament influence and an
  inner rating backend
- you are comparing methodology rather than just looking for the default
  ranking path

## Exported Configuration Types

The main configuration dataclasses are:

- `DecayConfig`
- `PageRankConfig`
- `EngineConfig`
- `TickTockConfig`
- `ExposureLogOddsConfig`

## Example Configuration

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

## Important `LOOPREngine` Knobs

### Decay And Recency

- `decay.half_life_days`: controls recency weighting for match importance

### Tournament Influence And Exposure

- `engine.beta`: controls how strongly tournament influence affects weights
- `engine.min_exposure`: filters low-exposure entities from the final output

### Post-Ranking Inactivity

- `engine.score_decay_delay_days`
- `engine.score_decay_rate`

These apply inactivity decay after ranking, rather than changing the base match
weights themselves.

### Smoothing

- `lambda_mode`
- `fixed_lambda`

These affect score smoothing inside the log-odds flow.

### Active-Entity Resolution

- `use_tick_tock_active`

When enabled, `LOOPREngine` uses a tick-tock pass to determine active entities
and tournament influences before the main ranking pass.

## PageRank Settings

`PageRankConfig` exposes low-level numerical settings such as:

- `alpha`
- `tol`
- `max_iter`

These are usually secondary to the higher-level exposure and decay choices, but
they matter when tuning convergence behavior or numerical stability.

## Tick-Tock Settings

`TickTockConfig` contains the outer-loop controls for tick-tock behavior,
including:

- `max_ticks`
- `convergence_tol`
- `teleport_mode`
- `smoothing_mode`
- `influence_method`

When using `LOOPREngine`, a nested `tick_tock` config still matters because it
controls the active-entity / tournament-influence pass when that feature is
enabled.

## Related Reading

- [defaults-and-recipes.md](defaults-and-recipes.md) for the recommended
  starting setup and conservative debugging baseline
- [input-patterns.md](input-patterns.md) for input tables and normalization
- [analysis-and-diagnostics.md](analysis-and-diagnostics.md) for post-run
  diagnostics and leave-one-match-out analysis
