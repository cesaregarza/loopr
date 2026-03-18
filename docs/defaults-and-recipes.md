# Defaults And Recipes

This page is the shortest path from "the library looks powerful" to "I know
what to run first."

For the public entrypoint, start in [README.md](../README.md). For the broader
engine and config reference, see [engines-and-configuration.md](engines-and-configuration.md).

## Recommended Default

For most real datasets, the recommended starting point is:

- `LOOPREngine`
- the default `ExposureLogOddsConfig`
- `appearances` included whenever you have them
- no tuning until you have external validation or a clear operational reason

That default is a good first pass because it already includes:

- exposure-aware scoring rather than raw volume accumulation
- time decay
- tick-tock-assisted active-set and tournament-influence resolution
- exact leave-one-match-out analysis support after ranking

What this default does **not** mean:

- the score is not a calibrated probability
- tournament influence should not be treated as automatically correct
- roster fallback is not equivalent to true per-match participation

Treat the default output as a ranking signal. Validate it before tuning it or
turning it into product thresholds.

## Default Interpretation Rules

These are the main modeling assumptions to keep in mind when using the default
path:

- `appearances` are strongly preferred over roster fallback when real lineups
  matter
- `use_tick_tock_active=True` keeps inactive historical entities from defining
  the output set
- `engine.beta=1.0` means tournament influence is active by default and should
  be checked against a `beta=0` ablation
- `decay.half_life_days=30.0` means recent results matter more than old ones

If any of those assumptions feel questionable in your domain, the answer is
usually to run a focused ablation rather than immediately inventing a new
configuration.

## Recipe: Smallest Viable Setup

Use this when you want the fewest moving parts on a clean binary-results
dataset.

```python
from loopr import LOOPREngine

engine = LOOPREngine()
rankings = engine.rank_entities(matches, participants)
```

Use this when:

- you want the main recommended path immediately
- you do not yet have `appearances`
- you are still checking data quality and schema fit

## Recipe: Recommended Production Baseline

Use this as the first serious baseline before tuning.

```python
from loopr import ExposureLogOddsConfig, LOOPREngine

config = ExposureLogOddsConfig()
engine = LOOPREngine(config=config)
rankings = engine.rank_entities(
    matches,
    participants,
    appearances=appearances,
)
```

Expect this baseline to be accompanied by:

- one external or predictive validation target
- one simpler baseline outside `loopr`
- one compact ablation table

This is the baseline that later variants should be compared against.

## Recipe: Conservative Debugging Baseline

Use this when you want to isolate the effect of the more coupled ingredients.

```python
from loopr import ExposureLogOddsConfig, LOOPREngine
from loopr.core import DecayConfig, EngineConfig

config = ExposureLogOddsConfig(
    decay=DecayConfig(half_life_days=0.0),
    engine=EngineConfig(beta=0.0),
    use_tick_tock_active=False,
)

engine = LOOPREngine(config=config)
rankings = engine.rank_entities(matches, participants, appearances=appearances)
```

This is not the recommended production setup. It is useful because it removes:

- recency weighting
- tournament-influence weighting
- tick-tock active-set resolution

That makes it easier to answer "is the surprising behavior coming from the core
graph or from the added weighting logic?"

## What To Tune Last

New users often reach for these too early:

- `engine.beta`
- `decay.half_life_days`
- `engine.min_exposure`
- `score_decay_delay_days` and `score_decay_rate`

Recommended order:

1. verify the data shape and add `appearances` if possible
2. establish one external validation target
3. compare the default against a simpler baseline
4. run a small ablation grid
5. only then tune knobs

`engine.min_exposure` is usually an output hygiene setting, not the first lever
to use for model quality.

## Minimum Ablation Grid

If you only run a few comparisons, start here:

| Variant | Why it matters |
|---|---|
| default `LOOPREngine` | main recommended baseline |
| `beta=0` | tests tournament-influence circularity risk |
| no time decay | tests whether recency is helping |
| no tick-tock active set | tests whether the active-set logic is buying enough |
| no `appearances` or roster fallback only | tests whether per-match participation materially changes attribution |

For more on ablation structure, see [ablations.md](ablations.md).

## Related Reading

- [engines-and-configuration.md](engines-and-configuration.md)
- [validation-harness.md](validation-harness.md)
- [validation-and-benchmarks.md](validation-and-benchmarks.md)
- [case-studies/sendou-plus.md](case-studies/sendou-plus.md)
