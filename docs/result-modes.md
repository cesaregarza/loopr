# Result Modes

This page describes the two result interpretations currently used when `loopr`
builds entity-level graph inputs from competition data.

For the general public entrypoint, start in [README.md](../README.md). For the
broader advanced docs map, see [the docs index](README.md).

## Binary Group Results

This is the main public path used by `rank_entities(...)`.

Input shape:

- one row per match
- `event_id`
- `match_id`
- `winner_id`
- `loser_id`
- optional `completed_at`, `created_at`, `walkover`

`winner_id` and `loser_id` are group identifiers. They may represent teams,
duos, relays, or any other grouping. `participants` provides the entity roster
for each group, and optional `appearances` narrows that roster down to who
actually played in a specific match.

This is why "team mode" still makes sense when the final ranking target is
individual players: the match outcome is recorded at the group level, then
expanded down to entities.

## Positional Results

This is the advanced/helper-level path for ordered finishes.

Input shape:

- one row per finisher within a result set
- `event_id`
- `match_id`
- `placement`
- exactly one of `entity_id` or `group_id`
- optional `completed_at`, `created_at`, `walkover`

Semantics:

- every better placement beats every worse placement
- ties create bidirectional peer edges
- self-edges are not created

Examples:

- `placement = [1, 2, 3, 4]` implies `1>2`, `1>3`, `1>4`, `2>3`, `2>4`, `3>4`
- `placement = [1, 1, 2]` implies both tied finishers beat each other and both
  beat the third-place finisher

When positional rows use `group_id`, `participants` and optional `appearances`
are applied in the same way as binary group results before pairwise expansion.

### When To Use Positional Results

Use positional results when the raw data is naturally an ordered finish list
rather than a single binary winner/loser match row.

Examples:

- race results
- placement-based tournament rounds
- ordered leaderboard slices that should imply pairwise wins

## Weighting Modes

Positional expansion currently supports two weighting policies:

- `pairwise_full`: every implied comparison gets the full base event weight
- `pairwise_average`: implied comparisons are scaled so the total expanded
  weight stays bounded across larger finish fields

## Current API Surface

Positional support is currently documented as an advanced/helper-level feature.

Available helper surfaces:

- `loopr.convert_matches_dataframe(..., result_mode="positional")`
- `loopr.core.build_player_edges(..., result_mode="positional")`
- `loopr.core.convert_matches_format(..., result_mode="positional")`

Current limitation:

- `rank_entities(...)` is still documented and supported around the binary
  group-result public input shape
- positional inputs are currently meant for lower-level prep/edge/conversion
  helpers rather than the main engine-level public flow

## Minimal Example

```python
import polars as pl

from loopr import convert_matches_dataframe

results = pl.DataFrame(
    {
        "event_id": [1, 1, 1],
        "match_id": [10, 10, 10],
        "entity_id": [101, 102, 103],
        "placement": [1, 2, 3],
        "completed_at": [1_700_000_000] * 3,
    }
)

expanded = convert_matches_dataframe(
    results,
    players=None,
    tournament_influence={},
    now_timestamp=1_700_000_000.0,
    decay_rate=0.0,
    result_mode="positional",
)

print(expanded.select(["winners", "losers", "share"]))
```

## Related Reading

- [input-patterns.md](input-patterns.md) for the main input tables and
  normalization helpers
- [engines-and-configuration.md](engines-and-configuration.md) for engine
  selection and config context
