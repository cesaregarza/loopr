# Ablations

This page describes how to think about ablations for `loopr`.

The point of an ablation is not to produce a giant table for its own sake. It
is to isolate which ingredients actually matter for ranking quality, stability,
or runtime.

For the public-facing docs, start in [README.md](../README.md). For the docs
index, see [README.md](README.md).

## What An Ablation Should Answer

A good ablation asks a narrow question:

- What does recency weighting buy us?
- What changes when tournament influence is enabled?
- Does `appearances` materially improve the graph over roster fallback?
- How much does the tick-tock active set change outputs or runtime?
- What is the tradeoff between simpler PageRank-style ranking and exposure
  log-odds ranking?

Bad ablations change too many things at once and then over-interpret the result.

## The Ingredients Worth Isolating

For `loopr`, the main ingredient families are:

### Result Preparation

- roster fallback only
- `appearances`-aware resolution

### Weighting

- no time decay
- time decay enabled
- tournament influence disabled (`beta = 0`)
- tournament influence enabled

### Graph Construction

- row-edge / tick-tock style graph
- exposure pair-edge / log-odds style graph

### Score Construction

- win-only or simpler PageRank-style ranking
- exposure log-odds with shared teleport
- smoothing enabled vs disabled or fixed vs auto

### Active-Set Logic

- all participants active
- tick-tock-derived active set

## Example Ablation Table Shape

The Sendou-oriented reports are useful partly because the table shape is simple
and repeatable. A domain-agnostic `loopr` ablation table should usually look
something like:

| Variant | Primary quality metric | Stability metric | Runtime metric |
|---|---:|---:|---:|
| Baseline simple model | ... | ... | ... |
| `loopr` base | ... | ... | ... |
| `loopr` + time decay | ... | ... | ... |
| `loopr` + appearances | ... | ... | ... |
| `loopr` + tournament influence | ... | ... | ... |
| `loopr` + tick-tock active set | ... | ... | ... |

The exact metrics depend on the domain, but the structure should stay compact.

## Recommended Metrics Per Column

### Quality

Pick one primary metric that reflects the real downstream goal:

- held-out match log loss
- held-out match AUC
- agreement with expert labels
- rank correlation with a trusted external target

### Stability

Examples:

- rank correlation between adjacent snapshots
- top-K overlap
- score variance under bootstrap or resampling

### Cost

Examples:

- total runtime
- stage timing breakdown
- memory use

## High-Value Ablation Comparisons

If time is limited, these are usually the most informative:

### `appearances` vs roster fallback

This tests whether real participation data matters enough to justify collecting
it.

Expected pattern:

- cleaner edges
- less teammate inflation
- better entity attribution

### time decay on vs off

This tests whether recency helps in the target domain.

Expected pattern:

- better short-horizon predictive alignment in dynamic ecosystems
- possibly lower long-term stability if the domain is very noisy

### tournament influence on vs off

This tests whether event-strength weighting improves results or just adds
complexity.

Expected pattern:

- better resistance to low-quality-volume farming in some ecosystems
- sometimes little effect in flatter or already well-mixed competition pools

### tick-tock active set on vs off

This tests whether the active-set logic improves the meaningful ranking pool or
just adds runtime.

Expected pattern:

- more sensible active output set in long-tail historical datasets
- measurable runtime increase

### log-odds vs simpler PageRank-style baselines

This is often the most important conceptual ablation because it tests whether
the exposure-aware dual-PageRank design is actually buying volume neutrality and
better external validity.

## Default Ablation Grid

If you want one compact ablation table around the recommended setup, start with
this grid:

| Variant | Change from recommended baseline | Main question |
|---|---|---|
| baseline | default `LOOPREngine` with `appearances` when available | what does the recommended path do |
| no decay | `half_life_days=0.0` | does recency help enough to justify extra sensitivity |
| `beta=0` | tournament influence disabled | is event-strength weighting buying signal or circularity |
| no tick-tock active set | `use_tick_tock_active=False` | does the active-set logic improve the output enough to justify runtime |
| roster fallback | omit `appearances` | how much does real per-match participation matter |

This is the minimum table that tests the most coupled ideas in the default
story.

## Report Hygiene

To keep ablations interpretable:

- change one ingredient family at a time when possible
- freeze the data window and filters across variants
- report confidence intervals or resampling uncertainty
- keep metric definitions identical across rows
- include at least one simpler baseline, not just close cousins of the main
  model

## What `loopr` Ships Today

`loopr` does not yet ship a full generic ablation runner.

What it does ship that makes ablations easier:

- configurable decay
- configurable tournament influence weighting
- optional tick-tock active-set logic
- synthetic throughput benchmarking
- detailed runtime diagnostics

So the current expectation is:

- the core library provides the ingredients
- the integrating project builds the evaluation harness around them

## Related Reading

- [defaults-and-recipes.md](defaults-and-recipes.md) for the recommended
  baseline that these ablations should branch from
- [validation-harness.md](validation-harness.md) for the reusable evaluation
  workflow this page plugs into
- [validation-and-benchmarks.md](validation-and-benchmarks.md) for the broader
  evaluation stack
- [engines-and-configuration.md](engines-and-configuration.md) for the knobs
  you can vary
- [how-loopr-works.md](how-loopr-works.md) for where those ingredients enter
  the pipeline
