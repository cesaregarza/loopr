# How LOOPR Works Under The Hood

This page is the deeper technical walk-through of the ranking pipeline.

It is not intended to define a stable internal API contract. It explains the
current implementation shape well enough to reason about behavior, debug
results, or extend the library.

For the public-facing starting point, use [README.md](../README.md). For the
advanced docs index, use [README.md](README.md).

## Big Picture

At a high level, `loopr` does four things:

1. Normalize neutral input tables into the internal schema
2. Turn binary group-result rows into entity-level winner/loser relationships
3. Build graph artifacts from those relationships
4. Run one of the ranking engines on the resulting graph

The implementation is organized so that the expensive graph-preparation work is
shared between multiple engines.

## Stage 1: Normalize Inputs

The public API starts from neutral column names such as:

- `event_id`
- `group_id`
- `entity_id`
- `winner_id`
- `loser_id`

`prepare_rank_inputs(...)` validates those inputs and renames them into the
internal schema used by the engines:

- `tournament_id`
- `team_id`
- `user_id`
- `winner_team_id`
- `loser_team_id`

That normalization boundary is important because almost everything deeper in the
pipeline assumes the internal names.

## Stage 2: Turn Results Into Weighted Entity Interactions

This is the most important conceptual step in the codebase.

The engines do not rank abstract match rows directly. They first convert result
rows into resolved entity-level winner/loser lists with per-result weights.

### 2A. Attach Timestamps And Base Weights

`prepare_weighted_matches(...)` does the first weighting pass.

It:

- filter out rows that cannot produce competitive outcomes
- filter byes / walkovers when `walkover` or `is_bye` is present
- choose a timestamp using `completed_at` / `last_game_finished_at`, then
  `created_at` / `match_created_at`, then `now`
- compute time decay
- optionally multiply by tournament influence raised to `beta`

Conceptually:

- if `beta == 0`, weight is just time decay
- otherwise, weight is time decay times tournament-strength adjustment

### 2B. Resolve Who Actually Participated

For the main public binary group-result path:

- `winner_team_id` and `loser_team_id` identify groups
- `participants` defines the roster for each group in each event
- `appearances`, when present, overrides the full roster for that specific match

This is why the public path can still produce individual rankings even when the
raw result data is group-shaped.

The resolved representation is:

- `match_id`
- `tournament_id`
- `winners`: list of entity IDs
- `losers`: list of entity IDs
- `weight`
- `ts`

When share-aware graph prep is requested, it also includes:

- `winner_count`
- `loser_count`
- `share = weight / (winner_count * loser_count)`

That `share` is the per winner-loser pair mass used by the exposure-style
engines.

### 2C. Two Different Edge Masses Exist

This is easy to miss and explains a lot of later behavior.

`loopr` tracks two closely related but different notions of mass:

- `weight`: the resolved match/event contribution before pair explosion
- `share`: the per winner-loser pair contribution for exposure-style graphs

For a 4-vs-4 resolved match:

- `weight` is the total resolved match weight
- `share` is `weight / 16`

That distinction matters because:

- the exposure log-odds flow uses pairwise `share`
- the row-edge / tick-tock flow aggregates loser-to-winner `weight`

## Stage 3: Build Shared Graph Artifacts

Once `loopr` has resolved matches, it builds reusable graph inputs.

The core shared artifacts are:

- `entity_metrics`
- `pair_edges`
- `node_ids`
- `node_to_idx`
- `index_mapping`

### Entity Metrics

`aggregate_entity_metrics(...)` sums per-entity quantities across both winners
and losers:

- total `share`
- total `weight`
- max `ts`

These metrics are reused for teleport construction, exposure reporting, and
last-activity tracking.

### Pair Edges

`_build_exposure_pair_edges(...)` explodes `winners` and `losers` into all
winner-loser pairs and sums `share` by pair.

This produces a pair table of:

- `winner_id`
- `loser_id`
- `share`

It is the compact pairwise graph input for exposure log-odds ranking.

### Node Sets

`merged_node_ids(...)` takes the entity IDs that actually appeared in resolved
matches and optionally unions them with an externally supplied active set.

That is important in the main engine because active-entity selection can come
from a prior tick-tock pass, while substitutions or actually appearing players
still need to survive into the graph if they were present in the results.

## Stage 4: Exposure Log-Odds Engine

`LOOPREngine` / `ExposureLogOddsEngine` is the main ranking path.

### 4A. Optional Tick-Tock Pass For Active Entities

If `use_tick_tock_active=True`, the engine first runs a tick-tock pass to get:

- an active entity set
- tournament influence values

If disabled, all entities in `participants` are treated as active and tournament
influence defaults to uniform.

### 4B. Prepare Exposure Graph Inputs

The engine then calls `prepare_exposure_graph(...)`, which returns the resolved
match table plus the reusable graph artifacts described above.

If the resolved matches are empty after filtering and roster/appearance
resolution, the engine returns an empty DataFrame.

### 4C. Build Teleport And Adjacency

The log-odds engine uses:

- teleport vector `rho` from aggregated `share`
- adjacency built from winner-loser pair `share`

`teleport_from_share(...)` takes the per-entity summed `share`, adds a tiny
epsilon, and normalizes it.

`build_exposure_triplets(...)` converts pair edges into sparse COO triplets with:

- row = winner index
- column = loser index
- data = pair `share`

That becomes the win adjacency matrix.

### 4D. Run Two PageRanks

The log-odds flow runs two PageRanks with the same teleport vector:

- win PageRank on the win adjacency
- loss PageRank on the transpose of that adjacency

This is the core modeling trick:

- high win PageRank means you receive mass from entities you beat
- high loss PageRank means you receive mass from entities that beat you

Using the same teleport vector in both directions is important. It means the
final ratio is anchored to the same exposure prior on both the win side and the
loss side, which is a big part of why the score behaves more like conversion
quality relative to exposure than raw match volume.

### 4E. Smooth And Convert To Scores

The engine computes a smoothing value `lambda`.

In auto mode, it targets a small fraction of the median win PageRank relative
to the median teleport mass. Then it computes:

- `win_smooth = win_pr + lambda * rho`
- `loss_smooth = loss_pr + lambda * rho`

The default score is:

- `log(win_smooth / loss_smooth)`

If log transform is disabled, the raw ratio is used instead.

### 4F. Post-Processing

After score computation, the engine may:

- apply inactivity decay based on last activity timestamps
- compute user-facing `exposure` from aggregated `weight`
- filter low-exposure entities using `engine.min_exposure`

The final result is sorted descending by score.

## Stage 5: Tick-Tock Engine

`TickTockEngine` uses a different graph semantics than the exposure log-odds
engine.

Instead of pairwise `share`, it works from loser-to-winner row edges aggregated
from resolved matches.

### 5A. Build Row Edges

`prepare_row_edge_inputs(...)` resolves matches and then
`_build_row_edge_dataframe(...)` explodes winner and loser lists into:

- `loser_user_id`
- `winner_user_id`
- `weight_sum`

These edges represent total loss mass from one entity toward another.

### 5B. Normalize Outgoing Loss Mass

Tick-tock then:

1. computes denominators with a smoothing strategy
2. normalizes each loser row so outgoing edge weights sum appropriately
3. converts the result to sparse triplets

This produces a row-stochastic style graph for PageRank.

### 5C. Tick

The “tick” step computes entity PageRank on the normalized row graph using the
configured teleport mode.

### 5D. Tock

The “tock” step re-estimates tournament influence from the participants who
actually appeared in the resolved matches.

Conceptually:

- collect participants by tournament from resolved matches
- aggregate participant quality/rating into tournament influence
- normalize tournament influence back to mean 1.0
- repeat until convergence or `max_ticks`

Those tournament influence values can then feed later weighting passes.

## Stage 6: TTL

`TTLEngine` wraps a tick-tock style outer loop around a rating backend.

In practice:

- the outer loop updates tournament influence
- the inner backend computes the rating result for the current influence values

So TTL keeps the anti-gaming / tournament-strength iteration structure while
delegating the inner ranking computation to a backend such as log-odds.

## Diagnostics And Stored State

After ranking, the main engine stores several useful objects:

- `last_result`
- `last_stage_timings`
- `tournament_influence`
- `_converted_matches_df`

That stored state is what makes the analysis and diagnostics features practical
without rerunning everything from scratch each time.

## Leave-One-Match-Out Analysis

The exact leave-one-match-out machinery sits on top of the prepared graph state
from the main engine.

The rough idea is:

- run the main ranking pipeline once
- keep the graph artifacts and factorization-friendly state
- answer “what if this match were removed?” queries using low-rank graph updates

That is why the library can provide exact per-match impact analysis without
recomputing the entire ranking from zero for every query.

For the actual equations behind the exact low-rank update and the newer
perturbation approximation used in benchmarking, see
[mathematical-machinery.md](mathematical-machinery.md).

## Mental Model Summary

If you want the shortest deep mental model, it is this:

- public inputs describe group results and rosters
- preparation resolves those into entity winner/loser lists plus weights
- exposure log-odds builds pairwise win/loss graph mass from `share`
- tick-tock builds normalized loser-to-winner transition mass from `weight`
- tournament influence changes the base weights before graph construction
- diagnostics and LOO analysis reuse the prepared graph state

## Related Reading

- [input-patterns.md](input-patterns.md) for the public input boundary
- [engines-and-configuration.md](engines-and-configuration.md) for engine
  selection and tuning
- [mathematical-machinery.md](mathematical-machinery.md) for the scoring and
  LOO equations
- [analysis-and-diagnostics.md](analysis-and-diagnostics.md) for the public
  analysis-facing features
