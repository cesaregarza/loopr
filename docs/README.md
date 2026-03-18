# Advanced Docs

The repository `README.md` is the general entrypoint for `loopr`.

This `docs/` directory is the deeper reference for users who already
understand the main public ranking path and want more detail about input
patterns, engine behavior, and analysis features.

## Guide

- [input-patterns.md](input-patterns.md)
  - detailed schema guidance for `matches`, `participants`, and `appearances`
  - when to use `prepare_rank_inputs(...)`
  - how roster-level and appearance-level participation differ
- [how-loopr-works.md](how-loopr-works.md)
  - deeper technical walk-through of normalization, weighting, graph prep, and engine execution
  - explains `weight` vs `share`, pair edges vs row edges, and how the engines diverge
- [engines-and-configuration.md](engines-and-configuration.md)
  - when to use `LOOPREngine`, `TickTockEngine`, or `TTLEngine`
  - output columns and major configuration knobs
  - example configuration setup
- [defaults-and-recipes.md](defaults-and-recipes.md)
  - recommended starting configuration for most users
  - conservative debugging baseline and minimal ablation grid
  - guidance on what to tune last
- [validation-and-benchmarks.md](validation-and-benchmarks.md)
  - overview of the evidence stack around correctness, speed, and model quality
  - what the repo ships today and what it does not yet ship
- [validation-harness.md](validation-harness.md)
  - reusable contract for future evaluation runs, variants, and report outputs
  - domain-agnostic workflow with Sendou Plus as the first reference example
- [ablations.md](ablations.md)
  - how to compare modeling ingredients cleanly
  - suggested ablation families, metrics, table shapes, and a default grid
- [case-studies/README.md](case-studies/README.md)
  - concrete domain examples using `loopr`
  - each case study is structured as context first, then results
- [analysis-and-diagnostics.md](analysis-and-diagnostics.md)
  - leave-one-match-out analysis workflow
  - diagnostics exposed after ranking runs
  - benchmark script and output expectations

## Public vs Advanced Surface

The main documented public flow remains:

1. Build neutral-schema `matches` and `participants`
2. Optionally add `appearances`
3. Call top-level `rank_entities(...)`
4. Drop down to `LOOPREngine` only when you need diagnostics or analysis state
