# Analysis And Diagnostics

This page collects the post-ranking analysis features and runtime diagnostics
available around the main engine.

## Leave-One-Match-Out Analysis

`LOOPREngine` supports exact leave-one-match-out analysis based on low-rank
PageRank updates.

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
- min / mean / max runtime
- mean per-stage timings from `LOOPREngine.last_stage_timings`

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
