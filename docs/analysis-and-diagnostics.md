# Analysis And Diagnostics

This page collects the post-ranking analysis features and runtime diagnostics
available around the main engine.

## Leave-One-Match-Out Analysis

`LOOPREngine` supports exact leave-one-match-out analysis based on low-rank
PageRank updates.

For the derivation and the benchmarked perturbation approximation, see
[mathematical-machinery.md](mathematical-machinery.md).

Typical flow:

```python
from loopr import LOOPREngine

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

Behavior notes:

- `analyze_match_impact(...)` is an exact single-match leave-one-out update.
- `analyze_entity_matches(..., limit=None)` exact-evaluates every match involving
  that entity.
- `analyze_entity_matches(..., limit=K)` can use a fast flux-based pre-ranking
  step internally to choose which matches to exact-evaluate, but the returned
  rows still contain exact LOO deltas.

`analyze_entity_matches(...)` returns a `DataFrame` sorted by absolute impact
with columns such as:

- `match_id`
- `is_win`
- `old_score`
- `new_score`
- `score_delta`
- `abs_delta`
- `win_pr_delta`
- `loss_pr_delta`

## Runtime Diagnostics

After a `LOOPREngine` run, the engine exposes:

- `last_result`
- `last_stage_timings`
- `tournament_influence`

`last_result` is an `ExposureLogOddsResult` and includes arrays such as:

- `scores`
- `ids`
- `win_pagerank`
- `loss_pagerank`
- `teleport`
- `exposure`
- `lambda_used`
- `stage_timings`

These are mainly useful for inspection, profiling, or deeper debugging after a
ranking run.

## Benchmarking

A synthetic benchmark script is included at:

- [`benchmarks/benchmark_rank_entities.py`](../benchmarks/benchmark_rank_entities.py)
- [`benchmarks/benchmark_loo.py`](../benchmarks/benchmark_loo.py)

Example:

```bash
PYTHONPATH=src python benchmarks/benchmark_rank_entities.py \
  --events 40 \
  --teams-per-event 32 \
  --matches-per-event 160 \
  --roster-size 4 \
  --repeats 3
```

```bash
PYTHONPATH=src python benchmarks/benchmark_loo.py \
  --dataset sendou_window \
  --limit-tournaments 100 \
  --variants exact_combined exact_separate perturb_2 perturb_4 \
  --entity-limits 20 50 \
  --repeats 3
```

The script prints JSON with:

- workload size
- row counts
- min / mean / max runtime
- mean per-stage timings from `LOOPREngine.last_stage_timings`

The LOO benchmark additionally reports:

- analyzer preparation time
- per-variant exact or approximate single-impact runtime
- per-variant per-entity batch-analysis runtime for the requested limits
- overlap and delta-correlation against the exact combined baseline
- approximate cache footprint

For the broader evaluation story, including how to structure external
validation and ablation reports, see
[validation-and-benchmarks.md](validation-and-benchmarks.md) and
[ablations.md](ablations.md).

## Development Checks

For general development validation, run:

```bash
pytest -q
```

## Related Reading

- [engines-and-configuration.md](engines-and-configuration.md) for engine
  selection and configuration
- [input-patterns.md](input-patterns.md) for the input tables feeding the
  ranking pipeline
