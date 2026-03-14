# LOOPR

Standalone domain-agnostic ranking library extracted from `sendouq_analysis`.

## Input schema

`loopr` accepts neutral tabular inputs and normalizes them internally.

Matches:
- `event_id`
- `match_id`
- `winner_id`
- `loser_id`
- optional `completed_at`, `created_at`, `walkover`

Participants:
- `event_id`
- `group_id`
- `entity_id`

Appearances:
- `event_id`
- `match_id`
- optional `group_id`
- `entity_id`

Legacy `sendouq_analysis` column names are still accepted.

## Quick start

Canonical API: `rank_entities(...)`. Legacy `rank_players(...)` wrappers remain
available for compatibility, but the neutral entity/group/event naming is the
recommended surface for new callers.

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
```
