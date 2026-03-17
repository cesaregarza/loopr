# Validation And Benchmarks

This page is the evidence-oriented overview for `loopr`.

The goal is not just to explain how to run the library, but how to separate
implementation confidence, runtime cost, and real-world model usefulness.

For the public entrypoint, start in [README.md](../README.md). For the advanced
docs index, see [README.md](README.md).

## What This Page Is For

There are at least four different questions people mean by "does the model
work?":

1. Does the implementation behave correctly?
2. Is it numerically stable?
3. Is it fast enough?
4. Does it produce rankings that align with the thing I care about?

Those questions need different evidence. This page is the map to that evidence.

## The Current `loopr` Evidence Surface

Today, the repository directly ships:

- a broad automated test suite for correctness and regressions
- a synthetic throughput benchmark in
  [`benchmarks/benchmark_rank_entities.py`](../benchmarks/benchmark_rank_entities.py)
- runtime diagnostics exposed through `LOOPREngine.last_stage_timings`
- exact leave-one-match-out tooling for local explanation and sensitivity checks

What the repository does **not** yet ship as a reusable generic harness:

- a packaged external-validation runner
- a cross-validation framework for predictive scoring
- a canned ablation runner that compares multiple model variants end to end

That means `loopr` can already support serious evaluation, but the integrating
project still has to assemble some of the evidence workflow itself.

## The Four Evidence Layers

### 1. Correctness

Use the test suite to establish that the implementation behaves as intended on:

- basic ranking scenarios
- edge construction semantics
- empty or filtered inputs
- appearance overrides
- deterministic reruns

This is the minimum bar before empirical interpretation.

### 2. Implementation Benchmarks

Use the synthetic benchmark script to answer:

- how runtime scales with more events, matches, and roster size
- how much `use_tick_tock_active` changes runtime
- which stages dominate wall-clock time

Example:

```bash
PYTHONPATH=src python benchmarks/benchmark_rank_entities.py \
  --events 40 \
  --teams-per-event 32 \
  --matches-per-event 160 \
  --roster-size 4 \
  --repeats 3
```

This is the right benchmark for implementation cost, not model quality.

### 3. Predictive Or External Validation

This is where a domain project has to do its own work.

Typical targets:

- held-out match prediction
- future event prediction
- alignment with expert ratings or human votes
- agreement with downstream tiers, divisions, or seeding outcomes

This is the layer that decides whether the ranking signal generalizes.

### 4. Sensitivity And Explainability

Use:

- `last_stage_timings`
- `last_result`
- `prepare_loo_analyzer(...)`
- `analyze_match_impact(...)`
- `analyze_entity_matches(...)`

These tools help explain:

- which matches drive a ranking
- whether surprising results come from sparse exposure
- whether tournament influence or appearance resolution is materially changing
  outcomes

## Where Each Kind Of Evidence Lives

Keep these concerns separated:

- implementation benchmarks: runtime and scaling
- ablations: what each modeling ingredient buys you
- predictive validation: does the signal predict something external
- operational diagnostics: stage timings, failure modes, memory, stability

That separation is one of the most useful lessons from the benchmark-heavy
documentation in `sendouq_analysis`.

## Recommended Workflow

For a serious `loopr` deployment, the recommended progression is:

1. start from the recommended defaults in
   [defaults-and-recipes.md](defaults-and-recipes.md)
2. run the built-in test suite and synthetic benchmark
3. define a validation run using the contract in
   [validation-harness.md](validation-harness.md)
4. compare the default against a small ablation grid from
   [ablations.md](ablations.md)
5. publish at least one case-study-style report if the model will be used in a
   real product decision

## Recommended Report Shape

A serious evaluation report should usually include:

- one-paragraph conclusion
- one compact headline metric table
- data window and filter rules
- config snapshot
- one stratified breakdown table
- one limitations section

If the ranking is later thresholded or converted into classes, add calibration
and threshold analysis separately rather than pretending the ranking score is
already a probability.

## Related Reading

- [defaults-and-recipes.md](defaults-and-recipes.md) for the starting setup
- [validation-harness.md](validation-harness.md) for the future reusable
  evaluation contract
- [ablations.md](ablations.md) for ingredient-level comparison design
- [case-studies/README.md](case-studies/README.md) for concrete applied examples
- [analysis-and-diagnostics.md](analysis-and-diagnostics.md) for the current
  built-in diagnostics
- [how-loopr-works.md](how-loopr-works.md) for the technical pipeline beneath
  the benchmark outputs
