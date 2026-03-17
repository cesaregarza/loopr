# Input Patterns

This page expands on the neutral input shape used by the public `rank_entities`
flow and the schema-prep utilities around it.

For the high-level overview, start in the repository
[README.md](../README.md).

## Public Ranking Path

The recommended public path uses:

- `matches`: binary group results
- `participants`: entity membership for each group
- optional `appearances`: match-level participation overrides

This means `winner_id` and `loser_id` in a match row identify groups, while the
final ranking output still targets entities.

## Matches

Required columns:

- `event_id`
- `match_id`
- `winner_id`
- `loser_id`

Optional columns:

- `completed_at`
- `created_at`
- `walkover`

Usage notes:

- `completed_at` is the preferred event timestamp when available
- `created_at` can be used as a fallback timestamp
- `walkover` marks byes / non-played results that should not contribute normal
  competitive edges

## Participants

Required columns:

- `event_id`
- `group_id`
- `entity_id`

Each row says that one entity belongs to one group within one event.

Typical examples:

- a sports team roster for a tournament
- a doubles team pairing
- an esports match roster
- any grouping where match outcomes are stored at the group level but rankings
  are desired at the entity level

## Appearances

Optional per-match participation overrides:

- `event_id`
- `match_id`
- `entity_id`
- optional `group_id`

Use `appearances` when a match should be attributed to only part of the stored
roster.

Examples:

- substitutes
- bench players who did not enter
- temporary lineups within a larger registered team

If `group_id` is omitted, `loopr` attempts to infer it from the participants
table when possible.

## When To Pass `appearances`

Use `appearances` when you care about which entities actually played.

If you do not pass `appearances`, `loopr` assumes the full event-level group
roster participated in each match.

That difference matters for exposure and entity-level edge construction because
the ranking engines distribute match effects across the participating entities,
not just the abstract group identifier.

Using `appearances` therefore reduces noise and teammate inflation in the
entity-level graph: edges connect the entities who actually played, not every
registered roster member. If `appearances` are not available, roster fallback
keeps older or less detailed datasets usable.

## Schema Utilities

`prepare_rank_inputs(...)` validates neutral input tables and renames them into
the internal schema used by the engines.

```python
from loopr import prepare_rank_inputs

prepared = prepare_rank_inputs(matches, participants, appearances)
```

The returned `NormalizedRankingInputs` object contains:

- `matches`
- `participants`
- `appearances`

After normalization, the internal column names are:

- `tournament_id` instead of `event_id`
- `team_id` instead of `group_id`
- `user_id` instead of `entity_id`
- `winner_team_id` / `loser_team_id` instead of `winner_id` / `loser_id`

That is primarily useful when working with lower-level helpers or debugging the
internal preparation pipeline.

## Neutral Schema Boundary

The supported public surface expects the neutral schema shown above.

Inputs that already use older internal names such as `tournament_id`,
`team_id`, or `user_id` are not the main documented public path, even though
those names still exist internally after normalization.

## Related Reading

- [result-modes.md](result-modes.md) for binary group results vs positional
  results
- [engines-and-configuration.md](engines-and-configuration.md) for engine
  selection and config knobs
