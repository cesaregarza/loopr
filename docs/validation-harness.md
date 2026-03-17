# Validation Harness Spec

This page defines the reusable validation workflow that `loopr` should be able
to support, even though the repository does not yet ship a generic runner.

The point is to remove ambiguity around what "serious evaluation" means for the
library and for projects built on top of it.

For the higher-level evidence overview, see
[validation-and-benchmarks.md](validation-and-benchmarks.md).

## Purpose

A validation harness for `loopr` should make it easy to answer the same
questions across datasets and model variants:

- does the ranking signal align with the real downstream target
- what do the defaults buy relative to simpler baselines
- which ingredients change quality, stability, and runtime
- what evidence is strong enough to justify tuning

This page is a contract for future tooling, not a claim that the runner already
exists.

## A Validation Run Should Contain

Every run should be organized around four things:

1. one dataset bundle
2. one split and filter policy
3. one baseline configuration plus named variants
4. one metric and report package

If those four things are not frozen together, the results are hard to compare.

## Dataset Bundle Contract

The core bundle is domain-agnostic:

- `matches`
- `participants`
- optional `appearances`

Optional evaluation-side tables:

- external labels or votes
- held-out outcome targets
- metadata for slices such as event type, field size, roster size, or activity
  level

The harness should treat these as separate layers:

- ranking inputs build the signal
- evaluation targets judge the signal
- slice metadata explains where the signal is strong or weak

## Split And Filter Policy Contract

Every report should explicitly freeze:

- the time cutoff or snapshot rule
- leakage-prevention rules
- the active-entity policy
- row inclusion and exclusion filters

Typical policies include:

- frozen-cutoff evaluation
- rolling snapshot evaluation
- held-out future outcomes
- active-only evaluation windows

These should be written down alongside the results, not left implicit in code.

## Variant Contract

Each validation run should have:

- one named baseline
- a small set of named variants
- optional non-`loopr` baselines for context

For `loopr`, the recommended baseline is:

- `LOOPREngine`
- default `ExposureLogOddsConfig`
- `appearances` included when available

The default variant grid should stay small:

- no time decay
- `beta=0`
- `use_tick_tock_active=False`
- roster fallback instead of `appearances` when both are available

That grid directly tests the main coupled assumptions the docs call out.

## Output Contract

Every run should emit at least:

- one machine-readable JSON summary
- one markdown report
- one compact headline table
- one stratified table by meaningful slices
- one recorded config snapshot

Optional but high value:

- exported ranking snapshots per variant
- bootstrap samples or confidence intervals
- stage timings and memory summaries

## Minimum Metric Groups

The harness should support at least four metric families.

### Quality

Examples:

- ROC AUC
- log loss
- rank correlation with an external target

### Stability

Examples:

- snapshot-to-snapshot Spearman
- top-K overlap
- bootstrap variation

### Runtime

Examples:

- total wall-clock runtime
- `last_stage_timings`
- scaling with match count or roster size

### Coverage

Examples:

- ranked entity count
- active-only entity count
- evaluation-row coverage after joins and filters

## Suggested Repository Layout

The harness should follow a clean separation of concerns:

- `benchmarks/`
  - synthetic throughput scripts and implementation-cost benchmarks
- `docs/validation-harness.md`
  - the reusable evaluation contract
- `docs/ablations.md`
  - variant design guidance
- `docs/case-studies/`
  - domain-specific evidence reports
- `artifacts/` or `reports/`
  - generated outputs from validation runs, typically gitignored or published
    elsewhere

The repo does not need checked-in benchmark datasets to adopt this structure.

## Example Conceptual Run Spec

This is a documentation-level example of the shape a future runner should
accept.

```yaml
run_name: sendou_plus_active_window
dataset_bundle:
  matches: data/matches.parquet
  participants: data/participants.parquet
  appearances: data/appearances.parquet
  external_targets: data/plus_votes.parquet
split_policy:
  type: frozen_cutoff
  cutoff_ts: 2024-06-01T00:00:00Z
  active_threshold_days: 90
baseline:
  engine: LOOPREngine
  config: default
variants:
  - name: no_decay
  - name: beta_0
  - name: no_tick_tock_active
  - name: roster_fallback
metrics:
  - roc_auc_macro
  - roc_auc_micro
  - spearman_top_100
  - runtime_total
```

This is not a current CLI format. It is the shape future tooling should target.

## First Reference Example: Sendou Plus

The first worked example for this contract should remain
[case-studies/sendou-plus.md](case-studies/sendou-plus.md).

Mapped onto the harness, that case study looks like:

- ranking inputs: tournament results plus rosters and appearances
- external target: Plus peer-vote outcomes
- split policy: temporal freeze before the vote snapshot cutoff
- filter policy: active-only window
- baseline: default LOOPR-style outcomes-only signal
- comparison set: OpenSkill and simple activity baselines
- outputs: AUC table, ablation snapshot, and qualitative interpretation

That makes it a good exemplar because it uses an external human signal rather
than just an internal ranking metric.

## Related Reading

- [defaults-and-recipes.md](defaults-and-recipes.md)
- [validation-and-benchmarks.md](validation-and-benchmarks.md)
- [ablations.md](ablations.md)
- [case-studies/sendou-plus.md](case-studies/sendou-plus.md)
